# Learning to Predict Future-Aligned Research Proposals with Language Models

Code and data for the paper [*Learning to Predict Future-Aligned Research Proposals with Language Models*](https://arxiv.org/abs/2603.27146).

We train language models to generate research proposals that align with what *future* papers actually publish, using a synthesized dataset of (target paper, inspiring papers) tuples drawn from past venues, and evaluate generations via retrieval + LLM-as-judge against held-out post-cutoff papers from NeurIPS'25 / ICML'25 / ICLR'25.

## Repository Structure

```
.
├── data/                 # SFT data (n=2823) + held-out eval data (n=819)
├── data_synthesis/       # Build training trees and (prompt, completion) pairs from past papers
├── sft/                  # LoRA fine-tuning for Qwen2.5 and Llama-3.1
├── inference/            # Generate proposals from base or fine-tuned models
├── evaluation/           # Retrieval + LLM-as-judge alignment scoring
├── analysis/             # Ablations and prediction-level analyses
├── human_evaluation/     # Pairwise human comparison interface and analysis
├── utils/                # Shared API / paper-filter / arxiv helpers
└── pyproject.toml
```

See [`data/README.md`](data/README.md) for the data schema and provenance.

## Setup

Requires Python ≥ 3.10. From the repo root:

```bash
pip install -e .
```

Environment variables:

| Variable | Used by | Required? |
|---|---|---|
| `OPENAI_API_KEY` | data synthesis, evaluation, prompting baselines | Yes for those steps |
| `S2_API_KEY` | Semantic Scholar lookups in `data_synthesis/` | Optional (higher rate limits) |

## Data

Shipped under `data/`:

- `data/sft/sft_n2823_{direct,with_reasoning,stepwise_cot}.jsonl` — 2,823 SFT instances in three completion formats
- `data/test_set/test_set_n819.jsonl` — 819 held-out evaluation instances
- `data/{sft,test_set}/structured_papers.json` — per-paper structured fields used as the LLM-judge corpus

All four prompt JSONLs are self-contained: each `prompt` already embeds the full inspiring-paper context, so consumers do not need to consult `structured_papers.json` for inference. Counts and schema are documented in [`data/README.md`](data/README.md).

## Reproducing the Pipeline

### 1. Fine-tuning

Fine-tune Qwen2.5-14B (LoRA) on the stepwise-CoT variant:

```bash
python sft/finetune_qwen.py \
    --model-name Qwen/Qwen2.5-14B-Instruct \
    --train-file data/sft/sft_n2823_stepwise_cot.jsonl \
    --output-dir models/qwen2.5-14b-sft-stepwise-cot
```

`sft/finetune_llama.py` is the analogue for Llama-3.1-8B-Instruct.

### 2. Inference

Generate proposals on the test set with a fine-tuned LoRA adapter:

```bash
python inference/generate_qwen_predictions.py \
    --model Qwen/Qwen2.5-14B-Instruct \
    --adapter-path models/qwen2.5-14b-sft-stepwise-cot \
    --local \
    --test-file data/test_set/test_set_n819.jsonl \
    --output predictions/qwen-14b-stepwise-cot.json
```

Drop `--adapter-path` to run the base model. Use `inference/prompting.py` for API-only prompting baselines (e.g. GPT-4.1).

### 3. Evaluation

Retrieve the closest matches in the held-out corpus, then have an LLM judge alignment along hypothesis / method / novelty / experiments:

```bash
python evaluation/evaluate.py \
    --predictions predictions/qwen-14b-stepwise-cot.json \
    --corpus data/test_set/structured_papers.json \
    --retriever embedding --subfield-scores \
    --output evaluation/results/eval_qwen-14b-stepwise-cot.json
```

`evaluation/validate_robustness.py` re-runs evaluation across retriever / judge configurations.

### 4. Building the Data from Scratch (optional)

The shipped JSONLs are sufficient to reproduce all training and evaluation results in the paper. If you instead want to rebuild the dataset end-to-end:

```bash
# Construct paper trees for a venue (downloads PDFs, structures content, picks inspirations)
python data_synthesis/pipeline.py --conference neurips_2024

# Convert trees to (prompt, completion) SFT pairs
python data_synthesis/one_layer_sft.py --input data/trees/neurips_2024.json --output data/sft/

# Build the held-out test set from post-cutoff venues
python data_synthesis/build_test_data.py --output data/test_set/
```

## Analysis and Human Evaluation

`analysis/` contains the ablations reported in the paper (number of inspiring papers, citation-type filtering) and prediction-level diagnostics. `human_evaluation/` ships the blind side-by-side comparison server, batch splitter, and majority-vote / Wilson-CI aggregator.

## Citation

```bibtex
@article{wang2026future,
  title   = {Learning to Predict Future-Aligned Research Proposals with Language Models},
  author  = {Wang, Heng and Jiang, Pengcheng and Sun, Jiashuo and Shi, Zhiyi and Yu, Haofei and Han, Jiawei and Ji, Heng},
  journal = {arXiv preprint arXiv:2603.27146},
  year    = {2026},
  url     = {https://arxiv.org/abs/2603.27146},
}
```
