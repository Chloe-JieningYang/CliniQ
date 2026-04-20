# (Optional) Data Indexing & Storage

Note: The data used in our experiment is already indexed and stored under `faiss/`. Below is the steps to reproduce our data preparation process.

1. Get raw JSON data from [MedAlpaca](https://huggingface.co/medalpaca/datasets) on Hugging Face store and them under the `data` folder in this directory. For example, we use the following files to build our RAG vectorstore:
   - medical_meadow_health_advice_only.json
   - medical_meadow_medical_flashcards.json
   - medical_meadow_medqa.json
   - medical_meadow_mmmlu.json
   - medical_meadow_pubmed_causal.json

2. Parse raw json data into line-separated documents in `.txt` format. An example code is provided in `load.py`

3. Run `python build_vector.py`, which will chunk the documents and create a Faiss index of their embeddings. This will create a sub-folder `faiss-index`. You may find the exact index we used in our model inference already uploaded under this sub-folder.

# RAG Retrieval

1. Run `python retriever.py` to test if the retriever is working, or directly run from `eval/inference.py` to generate CliniQ results with contexts retrieved from RAG.