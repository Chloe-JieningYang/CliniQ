# CliniQ

Medical question-answering stack built on **Llama-3.1-8B-Instruct**: **supervised fine-tuning (SFT)** on MedAlpaca data, **direct preference optimization (DPO)** to separate doctor vs patient personas, optional **retrieval-augmented generation (RAG)** over a curated knowledge mix, and a small **FastAPI + React** runtime for demos.

### Training & alignment (high level)

- **SFT**: MedAlpaca-style supervision on **Llama-3.1-8B-Instruct** → **predictions (1)** → **Evaluation 1** (e.g. **USMLE-style accuracy**).  
- **DPO**: preference learning so outputs match **doctor vs patient** personas; judge model (e.g. **Qwen2.5-7B**) for **LLM-as-Judge** and **persona alignment** on **predictions (2)**.  
- **RAG + final eval**: multi-source **knowledge base** → **FAISS** retrieval → combine with model → **USMLE-style** (and other) final checks.

### RAG backend (runtime)

- **Startup**: load **DPO-tuned PEFT + tokenizer**; if RAG on, load **embeddings + vector index** and cache retriever.  
- **Per request**: **question + audience + optional client context** → **top‑k similarity search** (skip if RAG off) → **merge + prompt** → **generate answer**.

---

## 1. Training pipeline (SFT → DPO → RAG & evaluation)

**Stage A — Supervised fine-tuning (SFT)**  
- **Data**: MedAlpaca-style medical instruction data.  
- **Model**: **Llama-3.1-8B-Instruct** with efficient adaptation (e.g. LoRA + 4-bit in this repo).  
- **Output**: domain-adapted **predictions (1)**.  
- **Evaluation 1**: **USMLE-style accuracy** (and other automated metrics you configure).

**Stage B — Preference refinement (DPO)**  
- **Preference data**: pairs built from **doctor-oriented prompts/inputs**, with **preferred = doctor-style answers** and **rejected = patient-style (or weaker) answers** so the model learns who it is speaking *as* and *to*.  
- **Method**: **Direct Preference Optimization (DPO)** on top of the SFT checkpoint.  
- **Output**: **predictions (2)** aligned to the intended personas.  
- **Evaluation**: **LLM-as-judge** (e.g. **Qwen2.5-7B** in the project figure) plus **persona alignment** checks.

**Stage C — Retrieval & final evaluation**  
- **Corpus**: multiple medical / exam-style sources (e.g. `medical_meadow_*`, health advice, flashcards, **MedGA**, **MMMLU**, **PubMed** / causal QA-style material) combined into a **knowledge base**.  
- **Index**: embeddings + **FAISS** (or equivalent) for **retrieved context** at query time.  
- **Inference**: **DPO model + retrieved context** → final answers.  
- **Final evaluation**: **USMLE accuracy** (and any additional benchmarks you wire in).

Implementation entry points in this repo include `train/finetune_lora.py` (SFT), `dpo/prepare_train_data.py` & `dpo/train_dpo.py` (DPO), `eval/run_hf_model_eval.py` & `dpo/evaluate_dpo.py` (metrics / judge-style eval), and `rag/build_vector.py` (offline index).

---

## 2. RAG backend pipeline (serving)

At **startup**:

1. Load the **DPO-tuned PEFT adapter + tokenizer** (base LLM + LoRA).  
2. If RAG is enabled: load the **sentence embedding model**, **deserialize the vector index**, and **cache the retriever**.

On **each chat request**:

1. Receive **question**, **audience** (doctor vs patient), and **optional client context**.  
2. If RAG is on: **similarity search** for top passages; otherwise skip retrieval.  
3. **Merge** retrieved passages with client context and **build the prompt**.  
4. **Run the language model** and **return the answer**.

Configured via environment variables (see `.env.example` and `backend/app/core/config.py`). Optional editable diagram source: `docs/rag-backend-pipeline.drawio`.

---

## 3. Repository layout (selected)

```
CliniQ/
├── docs/
│   ├── rag-backend-pipeline.drawio   # Optional diagram source
│   └── technical_report.md           # Detailed methods & hyperparameters
├── train/finetune_lora.py             # SFT (LoRA)
├── dpo/                               # DPO data prep, training, ablations, judge eval
├── eval/                              # MediQA / PubMedQA style generation + n-gram & BERT metrics
├── rag/                               # Offline FAISS index build + retriever helpers
├── backend/                           # FastAPI service (PEFT + optional RAG)
├── web/                               # Vite + React UI
├── requirements.txt
├── Dockerfile
└── dpo_model/                         # Default adapter path for the API (when present)
```

---

## 4. Requirements

- **NVIDIA GPU** recommended for training and for GPU RAG embeddings (CPU fallback exists for parts of RAG).  
- **CUDA / PyTorch**: this repo targets **PyTorch 2.7.1 + cu118** wheels in `requirements.txt`; match your **driver** to the CUDA user runtime you install.  
- **Hugging Face token** for gated models (Llama, etc.): set `HF_TOKEN` in `.env` (see `.env.example`).  
- **Docker** (optional): `Dockerfile` installs Python 3.10 + CUDA 11.8 runtime + pinned torch.

---

## 5. Quick start (minimal)

### Hugging Face access

Create `.env` in the repo root:

```bash
cp .env.example .env
# edit HF_TOKEN=...
```

### Python environment

Use a virtual environment, install PyTorch **with the cu118 index** if you need GPU, then the rest:

```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### SFT

```bash
python train/finetune_lora.py
```

Checkpoints default under `medical_lora_output/` (see script for `output_dir`).

### DPO (after SFT adapter exists)

```bash
python dpo/prepare_train_data.py   # build preference JSON (modes documented in dpo/)
python dpo/train_dpo.py            # TRL DPOTrainer on chosen pairs
```

### Evaluation

```bash
python eval/run_hf_model_eval.py --model_id <hf_id_or_local_path> [--peft] [--load_in_4bit]
python dpo/evaluate_dpo.py         # LLM-as-judge style comparisons when configured
```

### Offline RAG index

```bash
python rag/build_vector.py         # see rag/README.md for corpus paths & env
```

### API + UI (optional)

```bash
# Terminal A — from repo root, with venv active
cd backend && uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

# Terminal B
cd web && npm install && npm run dev
```

---

## 6. Further reading

- **`docs/technical_report.md`** — bilingual technical write-up (SFT / DPO / ablations / metrics).  
- **`final report.pdf`** — course / project narrative (open locally; PDF text extraction is not bundled in-repo).  
- **DPO & TRL**: [Direct Preference Optimization paper](https://huggingface.co/papers/2305.18290), [TRL docs](https://huggingface.co/docs/trl).

---

## 7. Troubleshooting (short)

- **CUDA driver vs wheel mismatch**: install a PyTorch build that matches your driver, or upgrade the host driver.  
- **OOM**: lower batch size / sequence length, enable 4-bit loading where supported.  
- **RAG / Triton compile errors on API startup**: install Python dev headers (`Python.h`) on the host if bitsandbytes/triton JIT is triggered.
