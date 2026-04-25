# NVIDIA CUDA 11.8 runtime; PyTorch cu118 wheels bundle their own CUDA user libs.
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt /app/requirements.txt

# Install PyTorch first (CUDA 11.8 build)
RUN pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118

# Install project dependencies from requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy rest of project files
COPY . /app

# Verify installation (optional, for debugging)
RUN python -c "import torch; print(torch.__version__); import transformers; print(transformers.__version__); import trl; print(trl.__version__)"

# Run from project root (/app). Examples:
#   python train/finetune_lora.py
#   python eval/inference.py
#   python eval/run_hf_model_eval.py --model_id <hf_model_id> [--peft] [--load_in_4bit]
# CMD ["python", "train/finetune_lora.py"]
