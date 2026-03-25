# CliniQ 技术报告 / Technical Report

## 基于 LoRA 微调与 DPO 偏好优化的医学问答大模型

## Medical QA with LoRA Fine-tuning and DPO Preference Optimization

---

## 1. 项目概述 / Project Overview

CliniQ 是一个面向医学问答（Medical QA）的大语言模型微调项目。基于 Meta 的 Llama-3.1-8B-Instruct，在单张 NVIDIA L4 GPU（23GB VRAM）上实现了从监督微调（SFT）到直接偏好优化（DPO）再到多维度评估的完整 alignment pipeline。

CliniQ is a medical question-answering fine-tuning project built on Meta's Llama-3.1-8B-Instruct. It implements a complete alignment pipeline — from supervised fine-tuning (SFT) through Direct Preference Optimization (DPO) to multi-dimensional evaluation — on a single NVIDIA L4 GPU (23GB VRAM).

### 1.1 技术栈 / Technology Stack

| 组件 / Component | 技术选型 / Technology |
|------|---------|
| 基座模型 / Base Model | Llama-3.1-8B-Instruct |
| 微调方法 / Fine-tuning | LoRA (Low-Rank Adaptation) |
| 量化 / Quantization | 4-bit NF4 (bitsandbytes) |
| 训练框架 / Training Framework | TRL (SFTTrainer / DPOTrainer) |
| 参数高效微调 / PEFT | Hugging Face PEFT |
| 运行环境 / Runtime | Docker + CUDA 12.1 |
| 评估 / Evaluation | BERTScore / BLEU / ROUGE + LLM-as-Judge |

### 1.2 数据集 / Datasets

| 数据集 / Dataset | 用途 / Usage | 规模 / Size |
|--------|------|------|
| medalpaca/medical_meadow_mediqa | SFT 训练 + 自动评估 / SFT training + automated eval | ~2,200 |
| medalpaca/medical_meadow_wikidoc_patient_information | DPO 偏好对构建 / DPO pair construction | ~5,900 |
| qiaojin/PubMedQA | 额外评估 / Additional eval (optional) | ~200 |

---

## 2. Pipeline 总览 / Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  阶段 1 / Stage 1: SFT（监督微调 / Supervised Fine-Tuning）          │
│  train/finetune_lora.py                                             │
│  Llama-3.1-8B + 4-bit NF4 + LoRA (r=8, alpha=16)                  │
│  数据 / Data: MediQA  |  输出 / Output: sft_adaptor/               │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  阶段 2 / Stage 2: DPO 偏好对构建 / Preference Pair Construction     │
│  dpo/prepare_train_data.py                                          │
│  三种模式 / Three modes: rule / model / model-rejected              │
│  输出 / Output: dpo/train_set/dpo_pairs_*.json                      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  阶段 3 / Stage 3: DPO 训练 / DPO Training                          │
│  dpo/train_dpo.py                                                   │
│  TRL DPOTrainer + LoRA (r=16, alpha=32) + sigmoid loss             │
│  输出 / Output: output/dpo_final/                                   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  阶段 4 / Stage 4: 评估 / Evaluation                                │
│  A. 自动指标 / Automated: eval/run_hf_model_eval.py                 │
│     BERTScore / BLEU / ROUGE                                        │
│  B. LLM-as-Judge: dpo/evaluate_dpo.py                               │
│     Pairwise (SFT vs DPO) + Absolute                               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. 阶段 1：监督微调 / Stage 1: Supervised Fine-Tuning (SFT)

### 3.1 训练配置 / Training Configuration

| 超参数 / Hyperparameter | 值 / Value |
|--------|-----|
| 基座模型 / Base Model | meta-llama/Llama-3.1-8B-Instruct |
| LoRA rank (r) | 8 |
| LoRA alpha | 16 (缩放因子 / scaling factor alpha/r = 2) |
| LoRA dropout | 0.05 |
| LoRA 目标模块 / target modules | q_proj, k_proj, v_proj, o_proj |
| 量化 / Quantization | 4-bit NF4, double quantization |
| 优化器 / Optimizer | paged_adamw_8bit |
| 学习率 / Learning rate | 2e-4 (cosine scheduler) |
| Warmup steps | 100 |
| Batch size | 2 (梯度累积 / grad accum 8, 有效 / effective = 16) |
| Epochs | 3 |
| 最大序列长度 / Max seq length | 512 |
| 精度 / Precision | bf16 |

### 3.2 Prompt 格式 / Prompt Format

采用 Llama 3 原生对话格式。 / Uses the native Llama 3 chat template.

```
<|start_header_id|>user<|end_header_id|>
{instruction}
Context: {input}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{output}
<|eot_id|>
```

**关键设计 / Key Design**：仅对 assistant 回答部分计算 loss（通过 label masking 将 prompt 部分设为 -100），避免模型学习 prompt 模板噪声。

Loss is computed only on assistant response tokens. Prompt tokens are masked with label=-100 to prevent learning template artifacts.

### 3.3 LoRA 参数量分析 / LoRA Parameter Analysis

Llama-3.1-8B 使用 GQA（Grouped Query Attention），K/V head 数量为 8（非 32）。

Llama-3.1-8B uses GQA with 8 KV heads (not 32).

| 模块 / Module | 权重维度 / Shape | LoRA 参数 / Params (r=8) |
|------|---------|----------------|
| q_proj | 4096 × 4096 | 8 × (4096+4096) = 65,536 |
| k_proj | 4096 × 1024 | 8 × (4096+1024) = 40,960 |
| v_proj | 4096 × 1024 | 8 × (4096+1024) = 40,960 |
| o_proj | 4096 × 4096 | 8 × (4096+4096) = 65,536 |
| **每层合计 / Per layer** | | **212,992** |
| **全模型 / Total (32 layers)** | | **6,815,744 (~6.8M)** |

可训练参数占比 / Trainable ratio: 6.8M / 8,030M ≈ **0.085%**

---

## 4. 阶段 2：DPO 偏好对构建 / Stage 2: Preference Pair Construction

### 4.1 三种数据构建模式 / Three Data Construction Modes

#### 模式 1 / Mode 1: Rule-based（基于规则退化 / Deterministic Degradation）

- **chosen** = 数据集原始标注 / original gold annotation
- **rejected** = 对原始答案进行确定性退化 / degraded version of gold answer

四种退化手段 / Four degradation strategies:

| 策略 / Strategy | 说明 / Description |
|------|------|
| 截断 / Truncation | 保留 30~60% 内容 / Keep 30-60% of content |
| 句序打乱 / Shuffle | 保留首尾句，中间随机排列 / Randomize middle sentences |
| 模糊前缀 / Hedging | 加入不确定表述 / Prepend "I'm not entirely sure, but..." |
| 删除细节 / Drop details | 移除含数值/剂量的句子 / Strip sentences with dosages/numbers |

每条数据随机叠加 1-2 种退化。生成 **5,000** 条偏好对。

Each sample applies 1-2 random degradations. Generated **5,000** preference pairs.

#### 模式 2 / Mode 2: Model-based（模型生成两端 / Both Sides Generated）

- **chosen** = SFT 模型在 temperature=0.0（greedy）下的输出 / greedy decoding output
- **rejected** = SFT 模型在 temperature=1.4（高随机性）下的输出 / high-temperature sample

通过温度差制造质量差距。 / Creates quality gap via temperature difference.

#### 模式 3 / Mode 3: Model-rejected（推荐 / Recommended）

- **chosen** = 数据集原始标注（不生成）/ gold annotation (no generation)
- **rejected** = SFT 模型在 temperature=1.4 下采样 / model sample at high temp

兼顾 chosen 的高质量和 rejected 的真实性。生成 **2,000** 条用于正式训练。

Combines high-quality human annotations for chosen with realistic model failures for rejected. Generated **2,000** pairs for production training.

### 4.2 数据质量控制 / Data Quality Control

- rejected 长度低于 10 词的样本被丢弃 / Rejected shorter than 10 words discarded
- chosen 与 rejected 完全相同的样本被跳过 / Identical pairs skipped
- 清理模型生成伪影 / Model artifacts cleaned

---

## 5. 阶段 3：DPO 训练 / Stage 3: DPO Training

### 5.1 训练配置 / Training Configuration

| 超参数 / Hyperparameter | 值 / Value |
|--------|-----|
| 基座 / Base | Llama-3.1-8B-Instruct + SFT LoRA adapter |
| DPO loss 类型 / type | sigmoid（原始 DPO / original DPO） |
| Beta (KL 惩罚系数 / penalty) | 0.1 |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 (缩放因子 / scaling = 2) |
| LoRA 目标模块 / targets | q/k/v/o_proj + gate/up/down_proj (7 个 / total) |
| 学习率 / Learning rate | 1e-6 |
| 调度器 / Scheduler | cosine |
| 有效 Batch / Effective batch | 64 |
| Epochs | 1 |
| 最大长度 / Max length | 1024 |
| 量化 / Quantization | 4-bit NF4 |

### 5.2 Prompt 格式 / Prompt Format

DPO 阶段采用 medAlpaca 的 Alpaca 格式。 / DPO uses the medAlpaca Alpaca format.

```
Below is an instruction that describes a task, paired with an input that
provides further context. Write a response that appropriately completes
the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
```

### 5.3 Reference Model 处理 / Reference Model Handling

利用 TRL 的内置机制：在 PeftModel 上自动禁用 LoRA adapter 来获取 reference policy 的 logits，无需加载第二份模型。

Leverages TRL's built-in mechanism: automatically disables LoRA adapters to derive reference policy logits, eliminating the need for a separate model copy.

### 5.4 训练指标 / Training Metrics (dpo_final, 120 steps)

| 指标 / Metric | 值 / Value |
|------|-----|
| 最终训练 Loss / Final train loss | 0.6798 |
| 最佳验证 Loss / Best eval loss | 0.6904 |
| 训练 Reward Margin / Train reward margin | 0.0281 |
| 训练 Reward Accuracy / Train reward acc | 66.7% |
| 验证 Reward Margin / Eval reward margin | 0.0050 |
| 验证 Reward Accuracy / Eval reward acc | 50.0% |

---

## 6. 阶段 4：评估 / Stage 4: Evaluation

### 6.1 自动指标 / Automated Metrics

使用 `eval/run_hf_model_eval.py` 计算。 / Computed via `eval/run_hf_model_eval.py`.

| 指标 / Metric | 说明 / Description |
|------|------|
| BERTScore (P/R/F1) | 基于 BERT 嵌入的语义相似度 / Semantic similarity via BERT embeddings |
| BLEU | n-gram 精确率 / n-gram precision (sacrebleu) |
| ROUGE-1/2/L | n-gram 召回率 / n-gram recall with stemming |

评估数据 / Eval data：MediQA 验证集 (10% split) + PubMedQA (可选 / optional)

### 6.2 LLM-as-Judge 评估 / LLM-as-Judge Evaluation

使用远程 Judge 模型做 pairwise 对比。 / Uses remote judge model for pairwise comparison.

**评估维度 / Dimensions (1-5)：**

| 维度 / Dimension | 含义 / Description |
|------|------|
| Accuracy | 医学事实准确性 / Medical factual correctness |
| Completeness | 信息完整性 / Information coverage |
| Clarity | 表达清晰度与结构 / Expression quality and structure |
| Safety | 安全警告与注意事项 / Safety warnings and precautions |

**去偏策略 / Debiasing**：每对回答做两轮评判（swap positions），两次结果一致才计为有效，否则记为 TIE。

Each pair is judged twice with swapped positions. A verdict counts only when both rounds agree; otherwise TIE.

### 6.3 Pairwise 评估结果 / Pairwise Results (Judge: Llama-3.1-8B-Instruct)

| 维度 / Dimension | SFT | SFT + DPO | 提升 / Improvement (%) |
|------|-----|-----------|---------|
| Accuracy | 4.55 | **4.65** | +2.20% |
| Completeness | 4.20 | **4.35** | +3.57% |
| Clarity | **4.65** | 4.58 | -1.51% |
| Safety | **4.53** | 4.47 | -1.32% |
| **Total** | **17.93** | **18.05** | **+0.67%** |

**分析 / Analysis**：

DPO 在 Accuracy（+2.20%）和 Completeness（+3.57%）上有明确提升，说明偏好优化帮助模型生成了更准确、更完整的医学回答。Clarity 和 Safety 略有下降，可能与 DPO 训练倾向生成更长回答导致结构略散有关。总分提升 +0.67%，DPO 带来了净正向效果。

DPO shows clear gains in Accuracy (+2.20%) and Completeness (+3.57%), indicating preference optimization helps generate more factually correct and comprehensive medical responses. Clarity and Safety show slight decreases, possibly because DPO-trained models tend to produce longer, slightly less structured responses. Overall net positive improvement of +0.67%.

---

## 7. 消融实验 / Ablation Study

### 7.1 实验设计 / Experimental Design

对两个关键超参数做 3×3 网格搜索。 / 3×3 grid search over two key hyperparameters.

| 维度 / Dimension | 消融值 / Values | 直觉 / Rationale |
|------|--------|------|
| Rejected 温度 / temperature | 0.8 / 1.2 / 1.6 | 控制偏好对质量差距 / Controls quality gap |
| DPO Beta | 0.05 / 0.1 / 0.2 | 控制 KL 约束强度 / Controls KL constraint |

- 温度越低 → rejected 质量越高 → 差距小 → DPO 信号弱但精细
- 温度越高 → rejected 质量越低 → 差距大 → 信号强但可能学到伪特征
- Beta 越小 → 允许偏离 reference policy 更多 → 更激进
- Beta 越大 → 约束更强 → 更保守

Lower temp → smaller gap → weaker but finer signal. Higher temp → larger gap → stronger but noisier. Smaller beta → more aggressive. Larger beta → more conservative.

每组使用 200 条偏好对，model 模式生成。 / 200 pairs per experiment, model mode.

### 7.2 数据构建方式对比 / Data Construction Comparison

通过训练过程中的 reward 指标对比三种策略。 / Comparing via training reward metrics.

**Reward Margin（chosen 与 rejected 的隐式奖励差 / implicit reward gap）：**

- 三种方式的 margin 均随训练递增 → DPO 有效 / All three rise → DPO works
- rule-mode 上升最快 → 模型快速学会区分规则退化 / Fastest rise → learns shallow patterns quickly
- model-mode 上升更慢但更稳定 → 信号更真实 / Slower but more realistic signal

**Reward Accuracy（正确偏好 chosen 的比例 / fraction preferring chosen）：**

- rule-mode 在 0.4 epoch 即达 ~0.6 → 偏好对太容易区分 / Too easy to distinguish
- model-mode 上升更缓慢 → 偏好对更难但信号更有价值 / Harder pairs, more valuable signal
- 最终三者收敛到 ~0.8 / All converge to ~0.8

**结论 / Conclusion**：Rule-based 数据让模型学会了"不截断""不加模糊前缀"等浅层特征，而非医学质量的深层差异。Model-based 数据虽然训练更慢，但提供了更有意义的偏好信号。

Rule-based data teaches shallow features ("don't truncate," "don't hedge") rather than deep medical quality differences. Model-based data provides more meaningful preference signals despite slower training.

---

## 8. 硬件与资源 / Hardware and Resources

| 资源 / Resource | 配置 / Configuration |
|------|------|
| GPU | NVIDIA L4, 23GB VRAM |
| 系统内存 / RAM | 15GB |
| 环境 / Runtime | Docker (nvidia/cuda:12.1.0-runtime-ubuntu22.04) |
| SFT 训练 / training | ~2-3 小时 / hours (3 epochs) |
| DPO 数据生成 / data gen | ~30-60 分钟 / min (1,000 samples, 4-bit) |
| DPO 训练 / training | ~20-30 分钟 / min (1 epoch) |

---

## 9. 项目结构 / Project Structure

```
CliniQ/
├── Dockerfile                          # Docker 环境 / environment
├── requirements.txt                    # Python 依赖 / dependencies
├── .env                                # HuggingFace Token
├── train/
│   └── finetune_lora.py                # SFT LoRA 微调 / fine-tuning
├── eval/
│   ├── inference.py                    # 交互式推理 / interactive inference
│   ├── run_hf_model_eval.py            # 自动评估 / automated eval (BERTScore/BLEU/ROUGE)
│   └── test_model.py                   # 冒烟测试 / smoke testing
├── dpo/
│   ├── prepare_train_data.py           # 偏好对生成 / pair generation (rule/model/model-rejected)
│   ├── train_dpo.py                    # DPO 训练 / training (TRL DPOTrainer)
│   ├── prepare_eval_data.py            # 评估数据准备 / eval data prep
│   ├── evaluate_dpo.py                 # LLM-as-Judge 评估 / evaluation
│   ├── run_ablation.sh                 # 消融实验 / ablation orchestrator
│   ├── analyze_ablation.py             # 消融分析 / ablation analysis
│   └── train_set/
│       ├── medical_meadow_small.json   # 原始数据 / source (~5,900)
│       ├── dpo_pairs_rule.json         # Rule-based 偏好对 / pairs (5,000)
│       ├── dpo_pairs_model_rejected_full.json  # Model-rejected 偏好对 / pairs (2,000)
│       └── ablation/                   # 消融数据 / ablation data (200 each)
├── sft_adaptor/                        # SFT LoRA adapter 权重 / weights
└── output/
    ├── dpo_final/                      # 最终 DPO 模型 / final model
    └── ablation/                       # 消融实验模型 / ablation models (9 runs)
```

---

## 10. 局限性与改进方向 / Limitations and Future Work

| 方面 / Aspect | 当前局限 / Limitation | 改进方向 / Improvement |
|------|---------|---------|
| 偏好数据 / Preference data | Rule-based 退化过于人工；model-based 依赖温度差 / Rule degradation is artificial; model-based relies on temp gap | 使用 GPT-4 或专业医生标注 / Use GPT-4 or professional medical annotations |
| 评估 / Evaluation | Judge 与被评估模型共享底座 / Judge shares base model (self-preference bias) | 不同架构 judge + 人类评估交叉验证 / Different-arch judges + human eval cross-validation |
| Prompt 不一致 / Mismatch | SFT 用 Llama 3 chat，DPO 用 Alpaca / SFT uses Llama 3 chat, DPO uses Alpaca | 统一格式 / Unify prompt format |
| 安全性 / Safety | Safety 维度略有下降 / Safety slightly decreases | 增加 safety-specific 偏好对 / Add safety-specific preference pairs |
| 规模 / Scale | 单卡 L4，数据量有限 / Single L4, limited data | 多卡 + 更大数据集 / Multi-GPU + larger datasets |
| DPO 变体 / Variants | 仅用原始 DPO / Only original sigmoid DPO | 尝试 SimPO / ORPO / Iterative DPO / Explore alternatives |
