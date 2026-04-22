# How to run the script

## Prepare training data for DPO

<!-- Generate rule-based pairs:

```bash
python prepare_train_data.py \
    --mode rule \
    --input_path ./train_set/medical_meadow_small.json \
    --max_samples 5000 \
    --output_path ./train_set/dpo_pairs_rule.json
```

Generate model-based pairs:

```bash
python prepare_train_data.py \
    --mode model \
    --input_path ./train_set/medical_meadow_small.json \
    --output_path ./train_set/dpo_pairs_llama3_both.json \
    --model_path "./sft_model" \
    --max_samples 5000 
```

Generate model-based (rejected only) pairs:

```bash
python prepare_train_data.py \
    --mode model-rejected \
    --input_path ./train_set/medical_meadow_small.json \
    --output_path ./train_set/dpo_pairs_llama3_rejected.json \
    --model_path "./sft_model" \
    --max_samples 5000 
``` -->

Prepare the raw data to DPO format:

```bash
python prepare_train_data.py
```

## Train DPO

```bash
export HF_TOKEN=<you-token>
```

```bash
python train_dpo.py \
    --model "meta-llama/Llama-3.1-8B-Instruct" \
    --sft_adapter_path "../sft_model" \
    --data_path "./train_set/dpo_train_data.jsonl" \
    --output_dir "../dpo_model" \
    --train_in_4bit True \
    --bf16 True \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_target_modules "['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']" \
    --beta 0.3 \
    --learning_rate 1e-6 \
    --global_batch_size 64 
```

## Prepare evaluation data for DPO

```bash
python prepare_eval_data.py
```

## Evaluate DPO model

```bash
python evaluate_dpo.py \
    --mode pairwise \
    --model_a ../sft_model \
    --model_b ../dpo_model \
    --judge_model Qwen/Qwen2.5-7B-Instruct \
    --input_path ./eval_set/patient_eval.json \
    --output_path ./patient_eval_results.json \
    --hf_token ___ \
    --load_in_4bit True \
    --batch_size 8
```

Eval with responses already generated:

```bash
python evaluate_dpo.py \
    --mode pairwise \
    --judge_model Qwen/Qwen2.5-7B-Instruct \
    --model_a ../sft_model \
    --model_b ../dpo_model \
    --input_path ./eval_set/doctor_responses.json \
    --checkpoint_path ./eval_result/qwen/doctor_checkpoint.json \
    --output_path ./eval_result/qwen/doctor_eval_results.json \
    --hf_token ___
```

```bash
python evaluate_dpo.py \
    --mode pairwise \
    --judge_model meta-llama/Llama-3.1-8B-Instruct \
    --model_a ../sft_model \
    --model_b ../dpo_model \
    --input_path ./eval_set/patient_responses.json \
    --checkpoint_path ./eval_result/llama/patient_checkpoint.json \
    --output_path ./eval_result/qwen/patient_eval_results.json \
    --hf_token ___
```

Judges:

- meta-llama/Llama-3.1-8B-Instruct
- Qwen/Qwen2.5-7B-Instruct

Test the dpo evaluation judge model:

```bash
python evaluate_dpo.py \
    --input_path test_judge.json \
    --model_a ./any_folder \
    --model_b ./any_folder \
    --mode pairwise \
    --judge_model Qwen/Qwen2.5-7B-Instruct \
    --hf_token <your-hf-token>
```