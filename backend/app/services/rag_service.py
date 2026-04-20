"""FAISS + sentence-transformers retrieval (aligned with eval/inference.py + rag/retriever.py)."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple

from ..core.config import Settings

logger = logging.getLogger(__name__)


def retrieve_rag_context(vectorstore: Any, user_query: str, k: int) -> str:
    """Dense similarity search over the FAISS index; join top-k chunks."""
    retrieved_docs = vectorstore.similarity_search(user_query, k=k)
    context_chunks = [doc.page_content for doc in retrieved_docs]
    return "\n\n".join(context_chunks)


def merge_rag_and_client_context(rag_text: str, client_context: Optional[str]) -> Optional[str]:
    """Prefer retrieved passages first, then optional user-supplied clinical context."""
    rag_stripped = (rag_text or "").strip()
    client_stripped = (client_context or "").strip()
    if rag_stripped and client_stripped:
        return f"{rag_stripped}\n\n{client_stripped}"
    if rag_stripped:
        return rag_stripped
    if client_stripped:
        return client_stripped
    return None


def load_rag_vectorstore(settings: Settings) -> Tuple[Optional[Any], Optional[str]]:
    """Load LangChain FAISS index; return (vectorstore, error_message). None, None if RAG disabled."""
    if not settings.rag_enabled:
        return None, None

    index_path = settings.resolved_rag_index_path()
    if not index_path.exists():
        msg = f"RAG index not found: {index_path}"
        logger.warning(msg)
        return None, msg

    try:
        import torch
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:
        msg = f"RAG dependencies missing: {exc!s}"
        logger.error(msg)
        return None, msg

    device_kw: Optional[Dict[str, str]] = None
    if torch.backends.mps.is_available():
        device_kw = {"device": "mps"}
    elif torch.cuda.is_available():
        device_kw = {"device": "cuda"}
    else:
        device_kw = None

    emb_kwargs: Dict[str, Any] = {"model_name": settings.rag_embedding_model}
    if device_kw is not None:
        emb_kwargs["model_kwargs"] = device_kw
    embedding_model = HuggingFaceEmbeddings(**emb_kwargs)
    start_time = time.time()
    vectorstore = FAISS.load_local(
        str(index_path),
        embedding_model,
        allow_dangerous_deserialization=True,
    )

    try:
        import faiss

        cpu_index = vectorstore.index
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        vectorstore.index = gpu_index
        logger.info("RAG FAISS index moved to GPU")
    except Exception as exc:  # noqa: BLE001 — CPU fallback is fine
        logger.info("RAG FAISS kept on CPU (GPU index unavailable): %s", exc)

    logger.info("RAG vectorstore loaded in %.2fs from %s", time.time() - start_time, index_path)
    return vectorstore, None
