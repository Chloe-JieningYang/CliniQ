from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path
import time
import torch

# RAG retrieval
def rag_retrieval(retriver, user_query: str, k: int = 1) -> str:
    retrieved_docs = retriver.similarity_search(user_query, k=k)
    context_chunks = [doc.page_content for doc in retrieved_docs]
    context = "\n\n".join(context_chunks)
    return context

def load_rag_retriever(index_path: str = "/home/ubuntu/CliniQ/rag/faiss_index"):
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    import faiss
    import time
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "mps"} if torch.backends.mps.is_available() else {"device": "cuda"} if torch.cuda.is_available() else None
    )
    start_time = time.time()
    vectorstore = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
    
    # # Convert CPU index to GPU index to speed up retrieval
    # cpu_index = vectorstore.index
    # res = faiss.StandardGpuResources()
    # gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    # vectorstore.index = gpu_index
    
    print(f"Rag retriever loaded successfully after {time.time() - start_time:.2f} seconds!")
    return vectorstore

if __name__ == "__main__":

    # Load pre-built faiss index into vector store
    base_dir = Path(__file__).resolve().parent.parent 
    index_path = base_dir / "faiss_index"
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "mps"} if torch.backends.mps.is_available() else {"device": "cuda"} if torch.cuda.is_available() else None
    )
    vectorstore = FAISS.load_local(str(index_path), embedding_model, allow_dangerous_deserialization=True)

    # Example usage
    query = "Asymmetrical molecules lack uniform charge distribution, often featuring lone pairs on the central atom or different terminal atoms."

    start_time = time.time()
    context = rag_retrieval(vectorstore, query)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds to retrieve context")

    print("Query:", query)
    print("Context:", context)