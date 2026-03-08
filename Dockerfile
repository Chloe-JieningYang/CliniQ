# Based on NVIDIA CUDA base image (CUDA 12.1)
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy project files to container
COPY . /app

# Install PyTorch first (CUDA 12.1 version)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Hugging Face libraries and TRL
RUN pip install --no-cache-dir transformers datasets accelerate trl peft bitsandbytes huggingface_hub python-dotenv tensorboard

# Optional: Install Jupyter (if notebook support is needed)
RUN pip install --no-cache-dir jupyter

# Verify installation (optional, for debugging)
RUN python -c "import torch; print(torch.__version__); import transformers; print(transformers.__version__); import trl; print(trl.__version__)"

# Default startup command (modify according to actual script)
# CMD ["python", "your_finetune_script.py"]
