#!/bin/bash
#
# DPO Ablation Experiment: Temperature × Beta grid search
#
# Phase 1: Generate preference pairs at different rejected temperatures
# Phase 2: Train DPO with different beta values for each dataset
# Phase 3: Summarize results
#
# Usage:
#   bash run_ablation.sh                    # full grid (3×3 = 9 experiments)
#   bash run_ablation.sh --quick            # sequential ablation (3+3 = 6 experiments)
#   TEMPS="0.8 1.4" BETAS="0.1" bash run_ablation.sh  # custom grid

set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────────
BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
SFT_ADAPTER="${SFT_ADAPTER:-./sft_adaptor}"
INPUT_DATA="${INPUT_DATA:-./dpo/train_set/medical_meadow_small.json}"
EVAL_DATA="${EVAL_DATA:-./dpo/mediqa_eval_ready.json}"

# Generation mode: "model" (both sides generated) or "model-rejected" (gold chosen + model rejected)
GEN_MODE="${GEN_MODE:-model}"

# Ablation grid
TEMPS="${TEMPS:-0.8 1.2 1.6}"
BETAS="${BETAS:-0.05 0.1 0.2}"

# Dataset size for ablation (smaller = faster iteration)
MAX_SAMPLES="${MAX_SAMPLES:-1000}"

# Training hyperparameters (fixed across ablation)
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GLOBAL_BATCH="${GLOBAL_BATCH:-32}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_MODULES="${LORA_MODULES:-[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-4}"

# Output directories
DATA_DIR="./dpo/train_set/ablation"
OUTPUT_BASE="./output/ablation"
RESULTS_FILE="${OUTPUT_BASE}/ablation_results.json"

# ─── Parse flags ──────────────────────────────────────────────────────────────
QUICK_MODE=false
for arg in "$@"; do
    case $arg in
        --quick) QUICK_MODE=true ;;
    esac
done

# ─── Helpers ──────────────────────────────────────────────────────────────────
timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

log() { echo "[$(timestamp)] $*"; }

ensure_dir() { mkdir -p "$1"; }

# ─── Phase 1: Generate preference pairs ──────────────────────────────────────
generate_data() {
    local temp=$1
    local output_file="${DATA_DIR}/dpo_pairs_${GEN_MODE}_temp${temp}.json"

    if [ -f "$output_file" ]; then
        log "SKIP data generation for temp=${temp} (file exists: ${output_file})"
        return 0
    fi

    log "START data generation: mode=${GEN_MODE}, rejected_temp=${temp}, samples=${MAX_SAMPLES}"

    python dpo/prepare_train_data.py \
        --mode "${GEN_MODE}" \
        --input_path "${INPUT_DATA}" \
        --output_path "${output_file}" \
        --model_path "${SFT_ADAPTER}" \
        --base_model_id "${BASE_MODEL}" \
        --max_samples "${MAX_SAMPLES}" \
        --rejected_temperature "${temp}" \
        --batch_size "${GEN_BATCH_SIZE}" \
        --load_in_4bit True

    log "DONE data generation for temp=${temp} → ${output_file}"
}

# ─── Phase 2: Train DPO ──────────────────────────────────────────────────────
train_dpo() {
    local temp=$1
    local beta=$2
    local data_file="${DATA_DIR}/dpo_pairs_${GEN_MODE}_temp${temp}.json"
    local output_dir="${OUTPUT_BASE}/dpo_temp${temp}_beta${beta}"

    if [ -d "${output_dir}" ] && [ -f "${output_dir}/adapter_config.json" ]; then
        log "SKIP training for temp=${temp}, beta=${beta} (output exists: ${output_dir})"
        return 0
    fi

    if [ ! -f "${data_file}" ]; then
        log "ERROR: data file not found: ${data_file}"
        return 1
    fi

    log "START DPO training: temp=${temp}, beta=${beta}"

    python dpo/train_dpo.py \
        --model "${BASE_MODEL}" \
        --sft_adapter_path "${SFT_ADAPTER}" \
        --data_path "${data_file}" \
        --output_dir "${output_dir}" \
        --train_in_4bit True \
        --bf16 True \
        --beta "${beta}" \
        --learning_rate "${LEARNING_RATE}" \
        --num_epochs "${NUM_EPOCHS}" \
        --per_device_batch_size "${BATCH_SIZE}" \
        --global_batch_size "${GLOBAL_BATCH}" \
        --lora_r "${LORA_R}" \
        --lora_alpha "${LORA_ALPHA}" \
        --lora_target_modules "${LORA_MODULES}" \
        --max_length "${MAX_LENGTH}" \
        --eval_steps 50 \
        --use_wandb False

    log "DONE DPO training: temp=${temp}, beta=${beta} → ${output_dir}"
}

# ─── Main ─────────────────────────────────────────────────────────────────────
main() {
    log "============================================================"
    log "  DPO Ablation Experiment"
    log "  Mode: ${GEN_MODE}"
    log "  Temperatures: ${TEMPS}"
    log "  Betas: ${BETAS}"
    log "  Samples: ${MAX_SAMPLES}"
    log "  Quick mode: ${QUICK_MODE}"
    log "============================================================"

    ensure_dir "${DATA_DIR}"
    ensure_dir "${OUTPUT_BASE}"

    local temps_arr=(${TEMPS})
    local betas_arr=(${BETAS})

    # Phase 1: Generate data at each temperature
    log "── Phase 1: Data Generation ──"
    for temp in "${temps_arr[@]}"; do
        generate_data "${temp}"
    done

    # Phase 2: Train DPO
    log "── Phase 2: DPO Training ──"

    if [ "${QUICK_MODE}" = true ]; then
        # Sequential ablation:
        #   Step A: fix beta=0.1, vary temperature → find best temp
        #   Step B: fix best temp (middle), vary beta → find best beta
        local default_beta="0.1"
        log "Quick mode: Phase 2a — fixing beta=${default_beta}, varying temperature"
        for temp in "${temps_arr[@]}"; do
            train_dpo "${temp}" "${default_beta}"
        done

        local middle_temp="${temps_arr[1]}"
        log "Quick mode: Phase 2b — fixing temp=${middle_temp}, varying beta"
        for beta in "${betas_arr[@]}"; do
            if [ "${beta}" = "${default_beta}" ]; then
                continue
            fi
            train_dpo "${middle_temp}" "${beta}"
        done
    else
        # Full grid search
        for temp in "${temps_arr[@]}"; do
            for beta in "${betas_arr[@]}"; do
                train_dpo "${temp}" "${beta}"
            done
        done
    fi

    # Phase 3: Summarize
    log "── Phase 3: Collecting Results ──"
    python dpo/analyze_ablation.py \
        --output_base "${OUTPUT_BASE}" \
        --results_file "${RESULTS_FILE}"

    log "============================================================"
    log "  Ablation complete! Results: ${RESULTS_FILE}"
    log "============================================================"
}

main
