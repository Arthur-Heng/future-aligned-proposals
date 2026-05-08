# Data

SFT training data and held-out evaluation data used in the paper.

## Layout

```
data/
├── sft/
│   ├── sft_n2823_direct.jsonl           # 2823 lines,  47 MB
│   ├── sft_n2823_with_reasoning.jsonl   # 2823 lines,  60 MB  (CoT)
│   ├── sft_n2823_stepwise_cot.jsonl     # 2823 lines,  59 MB
│   └── structured_papers.json           # 9,466 papers, 35 MB
└── test_set/
    ├── test_set_n819.jsonl              #  819 lines, 8.6 MB
    └── structured_papers.json           # 3,497 papers, 13 MB
```

Each instance pairs one target paper with ~5 inspiring papers (`5×n` for the vast majority;
a small tail at 3–4). Counted with multiplicity (an inspiring paper used in N instances
counts N times — they are shown to the model N times during training):

- SFT: 2,823 targets + 14,109 inspiring paper uses = **16,932 paper uses**
- Eval: 819 targets + 4,084 inspiring paper uses = **4,903 paper uses**
- Total: **21,835 paper uses** across 3,642 instances

Source venues: NeurIPS / ICML / ICLR 2021–2024 for SFT, NeurIPS'25 / ICML'25 / ICLR'25 for evaluation.

## SFT Training Data (`sft/`)

2,823 training instances, available in three completion formats. All three share the same
2,823 (target, inspiring-papers) tuples; only the completion style differs.

| File | Format | Description |
|---|---|---|
| `sft_n2823_direct.jsonl` | `direct` | Completion is the structured paper (hypothesis, method, novelty, experiments) only |
| `sft_n2823_with_reasoning.jsonl` | `cot` | Completion is gap-analysis reasoning followed by the structured paper |
| `sft_n2823_stepwise_cot.jsonl` | `stepwise_cot` | Completion is explicit step-by-step reasoning (one step per field) followed by the structured paper |

### Schema

`direct`:
```json
{
  "prompt": "...",
  "completion": "...",
  "root_paper_id": "<sha-hash>",
  "root_title": "...",
  "research_question": "...",
  "num_inspiring": 5,
  "reasoning_cost": 0.02,
  "gap_analysis": "..."
}
```

`with_reasoning` and `stepwise_cot` (same shape, fields wrapped in `metadata`):
```json
{
  "prompt": "...",
  "completion": "...",
  "metadata": {
    "tree_id": "tree_<sha>",
    "root_paper_id": "<sha-hash>",
    "root_title": "...",
    "research_question": "...",
    "num_inspiring": 5,
    "reasoning_cost": 0.02,
    "format": "stepwise_cot"          // stepwise_cot only
  }
}
```

The `prompt` field already embeds the inspiring-paper context, so training pipelines do
not need to consult `structured_papers.json` directly.

## Evaluation Data (`test_set/`)

`test_set_n819.jsonl` — 819 held-out test instances drawn from NeurIPS'25,
ICML'25, and ICLR'25 (post-training-cutoff papers). Same schema as the SFT
`with_reasoning`/`stepwise_cot` formats: `{prompt, completion, metadata}` with metadata
fields `{tree_id, root_paper_id, root_title, research_question, num_inspiring}`.

The `completion` is the ground-truth paper structure used as the gold reference for
the LLM-judge evaluation.

## Structured Papers (`*/structured_papers.json`)

Per-paper structured metadata: `{paper_id → record}` with each record containing:

```json
{
  "paper_id": "<sha-hash>",
  "title": "...",
  "year": 2024,
  "research_question": "...",
  "hypothesis": "...",
  "proposed_method": "...",
  "experiment_details": "...",
  "novelty_claims": "..."
}
```

`sft/structured_papers.json` indexes the distinct papers referenced (target + inspiring)
across the 2,823 SFT instances; `test_set/structured_papers.json` does the same for
the 819 evaluation instances. The JSON files dedup by `paper_id` because the same
paper may serve as inspiration for many target papers — the `prompt` strings in the
JSONLs already contain the full per-instance content, so this lookup table is just
provided for convenience (e.g. building the LLM-judge corpus, or reconstructing the
inspiring set per instance via the `tree_id`). All 819 evaluation root paper IDs
resolve in `test_set/structured_papers.json`; 190 inspiring-paper IDs referenced by
SFT instances failed structured extraction and are absent from `sft/structured_papers.json`,
but their content is already present inline inside the corresponding `prompt` strings,
so training is unaffected.
