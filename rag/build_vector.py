from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import datetime
import tiktoken
import torch
import time

# Define token length function
enc = tiktoken.get_encoding("cl100k_base")
def token_len(text):
    return len(enc.encode(text))

# Generator to stream lines
def stream_chunks(file_paths):
    for path in file_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if text:
                    yield text

# Batch generator
def batch_generator(iterable, batch_size=128):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def count_nonempty_lines(file_paths):
    """Stream files once; count lines that would be yielded by stream_chunks (O(1) memory)."""
    n = 0
    for path in file_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    n += 1
    return n


def _ascii_bar(done: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "[" + "?" * width + "]"
    filled = min(width, int(width * done / total))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _format_duration(seconds: float) -> str:
    if seconds <= 0:
        return "0s"
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s"


if __name__ == "__main__":
    batch_size = 128

    # Setup embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "mps"} if torch.backends.mps.is_available() else {"device": "cuda"} if torch.cuda.is_available() else None,
        encode_kwargs={"batch_size": batch_size}
    )
    print(f"Embedding model device: {embedding_model._client.device}")

    # Define index path
    base_dir = Path(__file__).resolve().parent
    index_path = base_dir / "faiss_index"

    # Define document paths
    folder = "documents"
    dir_path = Path(base_dir) / folder
    file_paths = [str(p) for p in dir_path.glob("*.txt")]
    print(f"Converting documents {file_paths} to vector store")

    n_lines = count_nonempty_lines(file_paths)
    num_batches = (n_lines + batch_size - 1) // batch_size if n_lines else 0
    print(
        f"Corpus: {n_lines} non-empty lines → {num_batches} batches (batch_size={batch_size})",
        flush=True,
    )
    if num_batches == 0:
        print("Nothing to index; exiting.")
        raise SystemExit(0)

    # Define text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=token_len,
        separators=["\n\n", "\n", ". ", "! ", "? ", " "],  # sentence-aware priority
    )

    # Build vector store
    vectorstore = None
    start_time = time.time()
    for batch_num, batch in enumerate(
        batch_generator(stream_chunks(file_paths), batch_size=batch_size), start=1
    ):
        docs = text_splitter.create_documents(batch)
        if vectorstore is None:
            vectorstore = FAISS.from_documents(docs, embedding_model)
        else:
            vectorstore.add_documents(docs)
        
        # Print progress
        if batch_num % 100 == 0:
            elapsed = time.time() - start_time
            pct = 100.0 * batch_num / num_batches
            remaining_s = (elapsed / batch_num) * (num_batches - batch_num)
            now = datetime.datetime.now()
            finish_at = now + datetime.timedelta(seconds=remaining_s)
            finish_fmt = (
                "%H:%M:%S" if finish_at.date() == now.date() else "%b %d %H:%M"
            )
            bar = _ascii_bar(batch_num, num_batches)
            line = (
                f"\r{bar} {pct:5.1f}%  "
                f"batch {batch_num}/{num_batches}  "
                f"({_format_duration(remaining_s)} left)"
            )
            # Pad so a shorter line clears remnants of a longer previous line
            print(f"{line:<120}", end="", flush=True)
    
    print()
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds to build vector store")

    # Save index
    vectorstore.save_local(str(index_path))
    print(f"Vector store saved to {index_path}")