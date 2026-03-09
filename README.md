# CliniQ

Medical Q&A LLM Fine-tuning Project - Using LoRA to fine-tune Llama models

## Project Overview

This project uses LoRA (Low-Rank Adaptation) technology to fine-tune Llama models (e.g., Llama-3.1-8B-Instruct) specifically for medical Q&A tasks. The dataset used is `medalpaca/medical_meadow_mediqa`.

## Requirements

- NVIDIA GPU (L4 or higher recommended, at least 8GB VRAM)
- CUDA 12.1
- Docker (recommended)

## Quick Start

### 1. Build Docker Image

```bash
docker build -t cliniq-env .
```

### 2. Get Hugging Face Token

**Important**: Llama models are gated models and require a Hugging Face Token for access.

1. **Visit Hugging Face Website**:
   - Register/Login: https://huggingface.co/join
   - Visit Token settings: https://huggingface.co/settings/tokens

2. **Create Token**:
   - Click "New token"
   - Select "Read" permission (sufficient for downloading models)
   - Copy the generated token (format: `hf_xxxxxxxxxxxxx`)

3. **Request Model Access**:
   - Visit the model page you want to use (e.g., https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
   - Click "Agree and access repository" to accept terms
   - Wait for approval (usually within minutes)

### 3. Configure Token

You can set the token in three ways (in order of priority):

**Method 1: Using .env file (Recommended)**
```bash
# Create .env file in project root
echo "HF_TOKEN=your_token_here" > .env
```

**Method 2: Environment variable**
```bash
export HF_TOKEN=your_token_here
```

**Method 3: Pass when starting Docker container**
```bash
docker run -e HF_TOKEN=your_token_here ...
```

### 4. Start Container

```bash
# If using .env file, it will be automatically loaded
docker run --gpus all -it --rm -v $(pwd):/app cliniq-env bash

# Or pass token directly
docker run --gpus all -it --rm \
  -v $(pwd):/app \
  -e HF_TOKEN=your_token_here \
  cliniq-env bash
```

### 5. Run Fine-tuning

Inside the container, execute:

```bash
python finetune_lora.py
```

The training process will:
- Automatically download the specified Llama model (configured in `finetune_lora.py`)
- Load `medalpaca/medical_meadow_mediqa` dataset
- Perform efficient fine-tuning using 4-bit quantization + LoRA
- Save model to `./medical_lora_output/final_model`

### 6. Inference with Fine-tuned Model

```bash
python inference.py [model_path]
```

If no path is specified, defaults to `./medical_lora_output/final_model`

## Configuration

### Training Parameters

You can modify the following parameters in `finetune_lora.py`:

- `lora_r`: LoRA rank (default: 8)
- `lora_alpha`: LoRA scaling factor (default: 16)
- `num_train_epochs`: Number of training epochs (default: 3)
- `per_device_train_batch_size`: Batch size (default: 4)
- `learning_rate`: Learning rate (default: 2e-4)
- `max_seq_length`: Maximum sequence length (default: 512)

### Memory Optimization

- Uses 4-bit quantization (NF4)
- Uses 8-bit optimizer (paged_adamw_8bit)
- Uses LoRA to train only a small number of parameters

On L4 GPU (24GB), fine-tuning with 4-bit quantization typically requires:
- Small models (1-3B): ~4-6GB VRAM
- Medium models (7-13B): ~8-12GB VRAM
- Large models (30B+): May require multiple GPUs or larger VRAM

## Project Structure

```
cliniQ/
├── Dockerfile              # Docker environment configuration
├── finetune_lora.py        # LoRA fine-tuning main script
├── inference.py            # Inference script
├── README.md               # Project documentation
├── .env                    # Environment variables (create from .env.example)
├── .env.example            # Example environment file template
└── medical_lora_output/    # Training output directory (generated after training)
    └── final_model/        # Final model save location
```

## Important Notes

1. **Hugging Face Token (Required)**:
   - Llama models are gated models, **must** provide Token to download
   - **Recommended**: Create `.env` file with `HF_TOKEN=your_token_here`
   - Alternative: Set environment variable `export HF_TOKEN=your_token_here`
   - Or pass when starting container: `docker run -e HF_TOKEN=your_token_here ...`
   - If Token is not set, script will prompt and provide guidance
   - **Note**: `.env` file is automatically ignored by git (see `.gitignore`)

2. **Dataset Format**: `medical_meadow_mediqa` dataset should contain `instruction` and `output` fields

3. **Model Saving**: After training completes, model is saved in `./medical_lora_output/final_model/`, including:
   - LoRA adapter weights
   - Tokenizer files
   - Configuration files

## Monitoring Training with TensorBoard

During training, loss metrics are automatically logged to TensorBoard. To visualize training progress:

1. **Start TensorBoard** (in a separate terminal or after training):
   ```bash
   tensorboard --logdir ./medical_lora_output/logs
   ```

2. **Access TensorBoard**:
   - Open your browser and go to: `http://localhost:6006`
   - You'll see real-time plots of:
     - Training loss
     - Validation loss
     - Learning rate
     - Other training metrics

3. **From Docker container**:
   ```bash
   # Inside container or from host
   tensorboard --logdir /app/medical_lora_output/logs --host 0.0.0.0 --port 6006
   ```
   Then access via `http://your-server-ip:6006`

## Troubleshooting

### CUDA Related Errors
- Ensure using `--gpus all` when starting container
- Check if NVIDIA driver version supports CUDA 12.1

### Out of Memory
- Reduce `per_device_train_batch_size`
- Increase `gradient_accumulation_steps`
- Reduce `max_seq_length`

### Model Download Failed
- Check network connection
- Confirm Hugging Face Token is set (if required)

## Reference Resources

- [PEFT Documentation](https://huggingface.co/docs/peft)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [Llama Models](https://huggingface.co/meta-llama)
- [Medical Meadow Dataset](https://huggingface.co/datasets/medalpaca/medical_meadow_mediqa)
