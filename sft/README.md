# SFT Fine-tuning

This directory contains scripts for fine-tuning LLMs on our generated SFT data.

**Supported Models:**
- Qwen2.5 (7B, 14B, 72B) → `finetune_qwen.py`
- Llama 3.1 (8B, 70B) → `finetune_llama.py`

## Setup

Install required dependencies:

```bash
pip install transformers torch peft accelerate
```

For 4-bit quantization:
```bash
pip install bitsandbytes
```

For Flash Attention 2 (optional, for even faster training):
```bash
pip install flash-attn --no-build-isolation
```
Note: The Llama script uses PyTorch SDPA by default, which requires no extra installation.

For logging with Weights & Biases:
```bash
pip install wandb
wandb login  # Enter your API key
```

## Fine-tuning

Use the `finetune_qwen.py` script to fine-tune a Qwen model:

### Qwen2.5-7B (2x H100)

```bash
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 sft/finetune_qwen.py \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    --train-file data/sft/sft_n2823_with_reasoning.jsonl \
    --output-dir models/qwen2.5-7b-sft-v2 \
    --num-epochs 1 \
    --batch-size 2 \
    --gradient-accumulation-steps 4 \
    --learning-rate 2e-5 \
    --max-length 6000 \
    --lora-r 32 \
    --lora-alpha 64 \
    --eval-steps 25 \
    --save-steps 50 \
    --use-bf16 \
    --report-to wandb \
    --wandb-project qwen-sft \
    --run-name "qwen2.5-7b-reasoning-v2"
```

### Qwen2.5-14B (2x H100)

```bash
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 sft/finetune_qwen.py \
    --model-name Qwen/Qwen2.5-14B-Instruct \
    --train-file data/sft/sft_n2823_with_reasoning.jsonl \
    --output-dir models/qwen2.5-14b-sft-v2 \
    --num-epochs 1 \
    --batch-size 1 \
    --gradient-accumulation-steps 8 \
    --learning-rate 1e-5 \
    --max-length 6000 \
    --lora-r 32 \
    --lora-alpha 64 \
    --eval-steps 25 \
    --save-steps 50 \
    --use-bf16 \
    --report-to wandb \
    --wandb-project qwen-sft \
    --run-name "qwen2.5-14b-reasoning-v2"
```

### Qwen2.5-7B (4x H100)

```bash
accelerate launch --num_processes 4 sft/finetune_qwen.py \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    --train-file data/sft/sft_n2823_with_reasoning.jsonl \
    --output-dir models/qwen2.5-7b-sft-v2 \
    --num-epochs 1 \
    --batch-size 2 \
    --gradient-accumulation-steps 2 \
    --learning-rate 2e-5 \
    --max-length 6000 \
    --lora-r 32 \
    --lora-alpha 64 \
    --eval-steps 25 \
    --save-steps 50 \
    --use-bf16 \
    --report-to wandb \
    --wandb-project qwen-sft \
    --run-name "qwen2.5-7b-reasoning-v2"
```

### Qwen2.5-14B (4x H100)

```bash
accelerate launch --num_processes 4 sft/finetune_qwen.py \
    --model-name Qwen/Qwen2.5-14B-Instruct \
    --train-file data/sft/sft_n2823_with_reasoning.jsonl \
    --output-dir models/qwen2.5-14b-sft-v2 \
    --num-epochs 1 \
    --batch-size 1 \
    --gradient-accumulation-steps 4 \
    --learning-rate 1e-5 \
    --max-length 6000 \
    --lora-r 32 \
    --lora-alpha 64 \
    --eval-steps 25 \
    --save-steps 50 \
    --use-bf16 \
    --report-to wandb \
    --wandb-project qwen-sft \
    --run-name "qwen2.5-14b-reasoning-v2"
```

### Single GPU (Limited VRAM)

For single GPU with limited VRAM:

```bash
python sft/finetune_qwen.py \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    --train-file data/sft/sft_n2823_with_reasoning.jsonl \
    --output-dir models/qwen2.5-7b-sft-v2 \
    --num-epochs 1 \
    --batch-size 1 \
    --gradient-accumulation-steps 16 \
    --learning-rate 2e-5 \
    --max-length 6000 \
    --lora-r 16 \
    --eval-steps 25 \
    --save-steps 50 \
    --load-in-4bit \
    --use-bf16
```

### Recommended Parameters for ~2800 Samples

Our SFT data (v2) has the following characteristics:
- **Samples**: 2823
- **Avg tokens per sample**: ~4900 (prompt: ~3200, completion: ~1700)
- **Max tokens**: ~5500
- **P95 tokens**: ~5200

**Epochs**: 1 epoch is recommended. With ~2800 samples, more epochs risk overfitting.
Save checkpoints frequently and pick the best one based on eval loss.

#### Qwen2.5-7B

| Setup | batch | grad_accum | effective_batch | steps/epoch | lora_r | lr |
|-------|-------|------------|-----------------|-------------|--------|----|
| 2x H100 | 2 | 4 | 16 | ~176 | 32 | 2e-5 |
| 4x H100 | 2 | 2 | 16 | ~176 | 32 | 2e-5 |
| 1x 24GB | 1 | 16 | 16 | ~176 | 16 | 2e-5 |

#### Qwen2.5-14B

| Setup | batch | grad_accum | effective_batch | steps/epoch | lora_r | lr |
|-------|-------|------------|-----------------|-------------|--------|----|
| 2x H100 | 1 | 8 | 16 | ~176 | 32 | 1e-5 |
| 4x H100 | 1 | 4 | 16 | ~176 | 32 | 1e-5 |

**Key settings:**
- `--num-epochs 1`: Avoid overfitting on small SFT data
- `--max-length 6000`: Data max is ~5500, need headroom
- `--lora-r 32`: Higher rank for better capacity
- `--learning-rate 1e-5`: Lower LR for 14B (larger model needs gentler updates)
- `--use-bf16`: Native BF16 on H100 for speed + quality
- No `--load-in-4bit`: Full precision on H100 for best results

**Training stats (all setups, 1 epoch):**
- Effective batch size: 16
- Steps per epoch: ~176
- Total steps: ~176
- Warmup steps: ~18 (10% default)

---

## Llama Fine-tuning

Use `finetune_llama.py` to fine-tune Llama 3.1 models.

### Llama-3.1-8B (2x H100)

```bash
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 sft/finetune_llama.py \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    --train-file data/sft/sft_n2823_stepwise_cot.jsonl \
    --output-dir models/llama-3.1-8b-sft-v1 \
    --num-epochs 1 \
    --batch-size 2 \
    --gradient-accumulation-steps 4 \
    --learning-rate 2e-5 \
    --max-length 6000 \
    --lora-r 32 \
    --lora-alpha 64 \
    --eval-steps 25 \
    --save-steps 50 \
    --report-to wandb \
    --wandb-project llama-sft \
    --run-name "llama-3.1-8b-v1"
```

### Llama-3.1-8B with QLoRA (Single GPU)

For limited VRAM, use 4-bit quantization:

```bash
python sft/finetune_llama.py \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    --train-file data/sft/sft_n2823_stepwise_cot.jsonl \
    --output-dir models/llama-3.1-8b-sft-qlora \
    --num-epochs 1 \
    --batch-size 1 \
    --gradient-accumulation-steps 16 \
    --learning-rate 2e-5 \
    --max-length 6000 \
    --lora-r 16 \
    --lora-alpha 32 \
    --eval-steps 25 \
    --save-steps 50 \
    --load-in-4bit
```

### Llama vs Qwen Default Settings

| Setting | Qwen | Llama |
|---------|------|-------|
| Max length | 2048 | 4096 |
| Batch size | 4 | 2 |
| Gradient accum | 4 | 8 |
| LoRA rank/alpha | 8/16 | 16/32 |
| Precision | FP16 | BF16 |
| Attention | default | SDPA |
| LR scheduler | linear | cosine |

### Recommended Parameters for Llama-3.1-8B (~2800 Samples)

| Setup | batch | grad_accum | effective_batch | steps/epoch | lora_r | lr |
|-------|-------|------------|-----------------|-------------|--------|----|
| 2x H100 | 2 | 4 | 16 | ~176 | 32 | 2e-5 |
| 4x H100 | 2 | 2 | 16 | ~176 | 32 | 2e-5 |
| 1x 24GB (QLoRA) | 1 | 16 | 16 | ~176 | 16 | 2e-5 |

### Llama-specific Options

- `--attn-implementation`: Attention implementation (`sdpa` default, `flash_attention_2`, or `eager`)
- `--no-bf16`: Disable BF16 training (BF16 is default for Llama)

---

## Parameters

**Common parameters (both scripts):**

| Parameter | Description | Qwen Default | Llama Default |
|-----------|-------------|--------------|---------------|
| `--model-name` | HuggingFace model name | `Qwen/Qwen2.5-7B-Instruct` | `meta-llama/Llama-3.1-8B-Instruct` |
| `--train-file` | Path to training data (JSON/JSONL) | required | required |
| `--output-dir` | Output directory for saved model | required | required |
| `--num-epochs` | Number of training epochs | 3 | 3 |
| `--batch-size` | Per-device batch size | 4 | 2 |
| `--gradient-accumulation-steps` | Gradient accumulation | 4 | 8 |
| `--learning-rate` | Learning rate | 2e-5 | 2e-5 |
| `--max-length` | Maximum sequence length | 2048 | 4096 |
| `--lora-r` | LoRA rank | 8 | 16 |
| `--lora-alpha` | LoRA alpha | 16 | 32 |
| `--warmup-ratio` | Warmup ratio | 0.1 | 0.1 |
| `--eval-steps` | Evaluate every N steps | None | None |
| `--save-steps` | Save checkpoint every N steps | 500 | 500 |
| `--save-total-limit` | Max checkpoints to keep (None = all) | 3 | 3 |
| `--load-in-4bit` | Use 4-bit quantization (QLoRA) | False | False |
| `--load-in-8bit` | Use 8-bit quantization | False | False |
| `--use-bf16` | Use BF16 training | False | True |
| `--use-fp16` | Use FP16 training | False | False |
| `--report-to` | Logging (`none`/`wandb`/`tensorboard`) | none | none |
| `--run-name` | Name for the training run | None | None |
| `--wandb-project` | Wandb project name | qwen-sft | llama-sft |
| `--resume-from-checkpoint` | Path to checkpoint to resume from | None | None |

**Llama-specific parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--attn-implementation` | Attention implementation (`sdpa`, `flash_attention_2`, `eager`) | sdpa |
| `--no-bf16` | Disable BF16 training | False (BF16 enabled) |

## Data Format

The training data should be in JSONL format with each line containing:

```json
{
  "prompt": "The inspiring papers and research question...",
  "completion": "The reasoning process and final paper...",
  "tree_id": "tree_xxx",
  "root_paper_title": "Paper Title"
}
```

## Memory Requirements

Base model memory (without training data):

| Model | 4-bit | Full Precision |
|-------|-------|----------------|
| Qwen2.5-7B | ~14GB | ~28GB |
| Qwen2.5-14B | ~28GB | ~56GB |
| Qwen2.5-72B | ~144GB | ~288GB |
| Llama-3.1-8B | ~16GB | ~32GB |
| Llama-3.1-70B | ~140GB | ~280GB |

### H100 (94GB) - Recommended Settings

| Model | Precision | batch_size | max_length | Est. VRAM |
|-------|-----------|------------|------------|-----------|
| Qwen2.5-7B | BF16 | 4 | 6000 | ~45GB |
| Qwen2.5-14B | BF16 | 2 | 6000 | ~70GB |
| Qwen2.5-72B | 4-bit | 1 | 6000 | ~80GB |
| Llama-3.1-8B | BF16 | 4 | 6000 | ~50GB |
| Llama-3.1-70B | 4-bit | 1 | 6000 | ~85GB |

With H100s, skip quantization (`--load-in-4bit`) for better training quality.

### Limited VRAM (24-40GB)

With long sequences (6000 tokens), memory usage increases significantly:
- Qwen2.5-7B + 4-bit + batch_size=1 + 6000 tokens: ~24-32GB VRAM
- Llama-3.1-8B + 4-bit + batch_size=1 + 6000 tokens: ~26-34GB VRAM
- Use `--load-in-4bit` and `--batch-size 1`

For very limited VRAM, consider:
- Reducing `--max-length` (may truncate data)
- Using gradient checkpointing (enabled by default)
- Using smaller models (Qwen2.5-3B, Llama-3.2-3B)
