"""Load PEFT model once and run constrained generation (thread-safe).

Heavy deps (torch, transformers, peft) are imported inside `load` / `generate`
so API tests can run without a CUDA stack installed.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.config import Settings
from .prompts import (
    Role,
    build_chat_messages,
    build_user_content,
    stub_chat_answer,
    system_prompt_for_role,
)
from .rag_service import (
    load_rag_vectorstore,
    merge_rag_and_client_context,
    retrieve_rag_context,
)

logger = logging.getLogger(__name__)


def _torch_dtype(name: str) -> Any:
    import torch

    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    return torch.float32


class LLMService:
    """Holds tokenizer + AutoPeft model; `generate` is serialized with a lock."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
        self._load_error: Optional[str] = None
        self._adapter_path: Path = settings.resolved_adapter_path()
        self._lock = threading.Lock()
        self._rag_vectorstore: Optional[Any] = None
        self._rag_load_error: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        if self._settings.mock_generation:
            return True
        return self._model is not None and self._tokenizer is not None

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def load(self) -> None:
        """Load adapter + tokenizer into VRAM/RAM. Safe to call once at startup."""
        self._load_error = None
        if self._settings.mock_generation:
            logger.info("CLINIQ_MOCK_GENERATION: skipping PEFT load (stub replies only)")
            return

        import torch
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer, BitsAndBytesConfig
        adapter_path = self._adapter_path
        if not adapter_path.exists():
            self._load_error = f"Adapter path does not exist: {adapter_path}"
            logger.error(self._load_error)
            return

        cfg_path = adapter_path / "adapter_config.json"
        if not cfg_path.is_file():
            self._load_error = f"No adapter_config.json under {adapter_path}"
            logger.error(self._load_error)
            return

        token = self._settings.hf_token
        kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
        }
        if token:
            kwargs["token"] = token

        bnb_config = None
        if self._settings.load_in_4bit:
            dtype = torch.bfloat16 if self._settings.torch_dtype == "bfloat16" else torch.float16
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
            kwargs["quantization_config"] = bnb_config

        kwargs["torch_dtype"] = _torch_dtype(self._settings.torch_dtype)
        kwargs["device_map"] = self._settings.device_map

        try:
            logger.info("Loading PEFT model from %s", adapter_path)
            self._model = AutoPeftModelForCausalLM.from_pretrained(
                str(adapter_path),
                **kwargs,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(adapter_path),
                **kwargs,
            )
        except Exception as exc:  # noqa: BLE001 — surface any load failure to operators
            self._model = None
            self._tokenizer = None
            self._load_error = f"Model load failed: {exc!s}"
            logger.exception("Failed to load model")

        if self._tokenizer is not None:
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
                self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
            if not hasattr(self._tokenizer, "pad_token_id") or self._tokenizer.pad_token_id is None:
                self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

    def load_rag(self) -> None:
        """Load optional FAISS retriever (eval/inference.py flow). Safe to skip if disabled or missing index."""
        self._rag_load_error = None
        self._rag_vectorstore = None
        if not self._settings.rag_enabled:
            return
        if self._settings.mock_generation:
            logger.info("RAG: skipped (mock generation)")
            return

        store, err = load_rag_vectorstore(self._settings)
        self._rag_vectorstore = store
        self._rag_load_error = err
        if store is not None:
            logger.info("RAG ready (top_k=%s)", self._settings.rag_top_k)
        elif err:
            logger.warning("RAG not available: %s", err)

    def model_card(self) -> Dict[str, Any]:
        """Non-secret summary for GET /api/v1/model."""
        real_weights = self._model is not None and self._tokenizer is not None
        card: Dict[str, Any] = {
            "loaded": self.is_loaded,
            "real_weights_loaded": real_weights,
            "mock_generation": self._settings.mock_generation,
            "adapter_path": str(self._adapter_path),
            "device_map": self._settings.device_map,
            "load_in_4bit": self._settings.load_in_4bit,
            "torch_dtype": self._settings.torch_dtype,
        }
        if self._load_error:
            card["error"] = self._load_error
        cfg_path = self._adapter_path / "adapter_config.json"
        if cfg_path.is_file():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                card["base_model_name_or_path"] = cfg.get("base_model_name_or_path")
            except (OSError, json.JSONDecodeError):
                card["base_model_name_or_path"] = None
        card["rag_enabled"] = self._settings.rag_enabled
        card["rag_index_path"] = str(self._settings.resolved_rag_index_path())
        card["rag_top_k"] = self._settings.rag_top_k
        card["rag_retriever_ready"] = self._rag_vectorstore is not None
        if self._rag_load_error:
            card["rag_error"] = self._rag_load_error
        return card

    def _build_prompt_string(self, role: Role, message: str, context: Optional[str]) -> str:
        """Render prompt: prefer native chat template; else Llama-3-style manual template (DPO user prefix)."""
        messages = build_chat_messages(role, message, context)
        tok = self._tokenizer
        if tok is None:
            raise RuntimeError("Tokenizer not loaded")

        if getattr(tok, "chat_template", None):
            try:
                return tok.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("apply_chat_template failed (%s); using fallback layout", exc)

        return self._fallback_prompt_no_template(role, message, context)

    def _fallback_prompt_no_template(self, role: Role, message: str, context: Optional[str]) -> str:
        """Match legacy `eval/inference.py` Llama-3-style layout (no separate system header)."""
        system = system_prompt_for_role(role)
        user_block = build_user_content(role, message, context)
        combined_user = f"{system}\n\n{user_block}"
        return (
            f"<|start_header_id|>user<|end_header_id|>\n"
            f"{combined_user}\n"
            f"<|eot_id|>\n\n"
            f"<|start_header_id|>assistant<|end_header_id|>\n"
        )

    def generate(
        self,
        role: Role,
        message: str,
        context: Optional[str],
    ) -> str:
        """Blocking generation; caller should run inside a thread pool."""
        if self._settings.mock_generation:
            with self._lock:
                return stub_chat_answer(role, message, context)

        import torch

        if not self.is_loaded or self._model is None or self._tokenizer is None:
            raise RuntimeError(self._load_error or "Model is not loaded")

        tokenizer = self._tokenizer
        model = self._model

        with self._lock:
            merged_context = context
            if self._settings.rag_enabled and self._rag_vectorstore is not None:
                rag_text = retrieve_rag_context(
                    self._rag_vectorstore,
                    message,
                    int(self._settings.rag_top_k),
                )
                merged_context = merge_rag_and_client_context(rag_text, context)

            if self._settings.rag_log_merged_preview:
                if merged_context:
                    preview = merged_context[:240].replace("\n", "\\n")
                    logger.info("Merged context for prompt: chars=%s preview=%s", len(merged_context), preview)
                else:
                    logger.info("Merged context for prompt: empty")

            prompt = self._build_prompt_string(role, message, merged_context)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            input_length = int(inputs.input_ids.shape[1])

            eot_token_id: Optional[int] = None
            try:
                encoded_eot = tokenizer.encode("<|eot_id|>", add_special_tokens=False)
                if encoded_eot:
                    eot_token_id = encoded_eot[0]
            except Exception:  # noqa: BLE001
                eot_token_id = None

            eos_token_id = tokenizer.eos_token_id
            stop_ids: List[int] = []
            if eot_token_id is not None and eot_token_id != eos_token_id:
                stop_ids.append(eot_token_id)
            if eos_token_id is not None:
                stop_ids.append(eos_token_id)
            final_eos = stop_ids[0] if stop_ids else tokenizer.eos_token_id

            temperature = float(self._settings.temperature)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=int(self._settings.max_new_tokens),
                    temperature=temperature,
                    top_p=float(self._settings.top_p),
                    do_sample=temperature > 0.0,
                    pad_token_id=tokenizer.pad_token_id
                    if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id,
                    eos_token_id=final_eos,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                )

            generated_tokens = outputs[0][input_length:]
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
            answer = tokenizer.decode(generated_tokens, skip_special_tokens=False)

        answer = answer.replace("<|eot_id|>", "").strip()
        answer = answer.replace("<|end_of_text|>", "").strip()
        if "<|start_header_id|>" in answer:
            answer = answer.split("<|start_header_id|>")[0].strip()

        if not answer or len(answer.strip()) < 5:
            marker = "<|start_header_id|>assistant<|end_header_id|>\n"
            if marker in full_response:
                answer = full_response.split(marker)[-1].strip()
                answer = answer.replace("<|eot_id|>", "").strip()
                if "<|start_header_id|>" in answer:
                    answer = answer.split("<|start_header_id|>")[0].strip()

        return answer
