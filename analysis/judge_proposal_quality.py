#!/usr/bin/env python3
"""
Multi-Dimensional LLM Judge for Proposal Quality.

Evaluates proposals on 3 dimensions (1-5 scale):
1. Resource Validity: Are datasets/baselines/models real and factually grounded?
2. Task-Method Consistency: Does the proposed method address the stated task/hypothesis?
3. Task-Experiment Consistency: Do metrics/datasets/baselines match the task type?
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.api import call_chat_completion


JUDGE_PROMPT = """You are a critical expert reviewer evaluating a research proposal. Be strict and skeptical. Score on 3 dimensions (1-5 scale).

=== RESEARCH PROPOSAL ===
{proposal}

=== EVALUATION DIMENSIONS (BE CRITICAL - average proposals should score 3) ===

1. **Resource Validity** (1-5): Are the mentioned datasets, benchmarks, baseline models, and tools REAL and correctly named?
   - 5: ALL resources verified as real, correctly named, appropriate for the task (rare - requires specific well-known resources)
   - 4: Most resources appear real, maybe 1-2 minor naming issues or obscure references
   - 3: Mix of clearly real resources and some that are generic/vague or cannot be verified
   - 2: Several resources appear fabricated, have wrong names, or don't exist
   - 1: Most resources are clearly hallucinated or incorrectly described
   
   RED FLAGS: Generic dataset names without specifics, made-up benchmark names, baseline models that don't exist

2. **Task-Method Consistency** (1-5): Does the method ACTUALLY solve the stated problem? Is there logical coherence?
   - 5: Method directly addresses every aspect of the research question with clear logical connection (rare)
   - 4: Method addresses the main task but may have minor logical gaps
   - 3: Method partially addresses the task, some components seem disconnected or tangential
   - 2: Weak connection between method and stated task, or method solves a different problem
   - 1: Method does not logically address the stated task
   
   RED FLAGS: Method components that don't connect to the hypothesis, solving a different problem than stated

3. **Task-Experiment Consistency** (1-5): Do experiments actually validate the claims? Are metrics appropriate?
   - 5: Experiments perfectly designed to test the hypothesis with appropriate metrics (rare)
   - 4: Experiments mostly appropriate, minor gaps in validation coverage
   - 3: Some experiments don't match the task type, or metrics are partially inappropriate
   - 2: Significant mismatch - experiments test different things than claimed
   - 1: Experiments completely inappropriate for the stated task
   
   RED FLAGS: Wrong metrics for the task type, missing key experiments, baselines from wrong domain

IMPORTANT: Be critical. Most proposals have flaws. A score of 5 should be rare. Average proposals score around 3.

Respond with ONLY a JSON object:
{{
  "resource_validity": {{"score": 1-5, "justification": "specific issues found or why it's good"}},
  "task_method_consistency": {{"score": 1-5, "justification": "specific logical gaps or strengths"}},
  "task_experiment_consistency": {{"score": 1-5, "justification": "specific mismatches or why appropriate"}}
}}"""


def strip_reasoning_from_proposal(text: str) -> str:
    """
    Extract only the proposal content, removing reasoning process.
    Copied from evaluation/evaluate.py for consistency.
    """
    if not text:
        return text

    # Handle corrupted outputs that echo the chat template
    if 'for reasoning sections' in text[:500] or '\nuser\n' in text[:600]:
        assistant_match = re.search(r'\nassistant\s*\n', text)
        if assistant_match:
            text = text[assistant_match.end():]

    # Remove step-wise CoT reasoning blocks (### Step N: ...)
    def _strip_step_blocks(t):
        return re.sub(
            r'###\s*Step\s*\d+[^\n]*\n'
            r'(?:(?!##\s|###\s*(?!Step)|Research Question[:\s]|Hypothesis[:\s]|Proposed Method[:\s\+]|Novelty Claims?[:\s]|Experiment Details?[:\s]).*\n)*',
            '', t
        )

    # Check for step-wise CoT markers
    step_markers = re.findall(r'###\s*Step\s*\d', text)
    if len(step_markers) >= 2:
        cleaned = _strip_step_blocks(text)
        # Strip leading text before first proposal header
        proposal_start = re.search(
            r'(##\s*Proposed Research|##\s*Research Question|'
            r'##\s*Proposed Method|##\s*Novel Research|'
            r'##\s*Proposal|\*\*Research Question)',
            cleaned, re.IGNORECASE
        )
        if proposal_start:
            cleaned = cleaned[proposal_start.start():]
        return cleaned.strip()

    # Try to find "## Proposed Research" section
    proposal_match = re.search(
        r'(##\s*Proposed Research.*)',
        text, re.DOTALL | re.IGNORECASE
    )
    if proposal_match:
        return proposal_match.group(1).strip()

    # Try other proposal headers
    for pattern in [r'##\s*Research Question', r'\*\*Research Question', r'Research Question:']:
        match = re.search(f'({pattern}.*)', text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return text.strip()


def load_predictions(path: str) -> List[Dict]:
    """Load predictions from JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)
    return data.get('predictions', data)


def load_predictions_for_test_set(
    test_set_path: str,
    predictions_dirs: List[str],
    model_variant: str = "qwen-14b-tuned-stepwise-cot-v4"
) -> List[Dict]:
    """Load predictions matching the 819 test set from multiple directories."""
    
    # Mapping from batch1 names to batch2 names (batch2 uses 'sft' instead of 'tuned')
    variant_mappings = {
        "qwen-14b-tuned-stepwise-cot-v4": ["qwen-14b-tuned-stepwise-cot-v4", "qwen-14b-sft-stepwise-cot-v4"],
        "qwen-14b-tuned-cot-v4": ["qwen-14b-tuned-cot-v4", "qwen-14b-sft-cot-v4"],
        "qwen-14b-tuned-ncot-v4": ["qwen-14b-tuned-ncot-v4", "qwen-14b-sft-ncot-v4"],
        "qwen-7b-sft-stepwise-cot-v4": ["qwen-7b-sft-stepwise-cot-v4"],
        "qwen-7b-sft-ncot-v4": ["qwen-7b-sft-ncot-v4"],
        "llama-8b-sft-stepwise-cot-v4": ["llama-8b-sft-stepwise-cot-v4"],
        "llama-8b-sft-ncot-v4": ["llama-8b-sft-ncot-v4"],
    }
    
    # Get possible variant names
    variant_names = variant_mappings.get(model_variant, [model_variant])
    
    # Load test set tree_ids
    test_tree_ids = set()
    with open(test_set_path) as f:
        for line in f:
            item = json.loads(line)
            metadata = item.get('metadata', {})
            if isinstance(metadata, str):
                metadata = json.loads(metadata.replace("'", '"'))
            test_tree_ids.add(metadata.get('tree_id', ''))
    
    print(f"Test set has {len(test_tree_ids)} unique tree_ids")
    
    # Collect predictions from all directories
    all_predictions = {}
    for pred_dir in predictions_dirs:
        found = False
        for variant_name in variant_names:
            pred_file = os.path.join(pred_dir, f"{variant_name}.json")
            if os.path.exists(pred_file):
                found = True
                with open(pred_file) as f:
                    data = json.load(f)
                
                count = 0
                for pred in data.get('predictions', []):
                    tree_id = pred.get('tree_id', '')
                    if tree_id in test_tree_ids and tree_id not in all_predictions:
                        all_predictions[tree_id] = pred
                        count += 1
                
                print(f"  {pred_file}: loaded {count} predictions")
                break
        
        if not found:
            print(f"  Warning: no matching file found in {pred_dir}")
    
    print(f"Total: {len(all_predictions)} predictions matching test set")
    
    return list(all_predictions.values())


def judge_proposal(
    proposal: str,
    model: str = "gpt-4.1-mini",
    dry_run: bool = False
) -> Tuple[Dict, float]:
    """Judge a single proposal on all dimensions."""
    
    prompt = JUDGE_PROMPT.format(proposal=proposal)
    
    if dry_run:
        return {
            "resource_validity": {"score": 3, "justification": "[DRY RUN]"},
            "task_method_consistency": {"score": 3, "justification": "[DRY RUN]"},
            "task_experiment_consistency": {"score": 3, "justification": "[DRY RUN]"}
        }, 0
    
    messages = [{"role": "user", "content": prompt}]
    response, cost = call_chat_completion(
        messages=messages,
        model=model,
        temperature=0.0,
        max_tokens=500
    )
    
    try:
        result = json.loads(response)
        return result, cost
    except json.JSONDecodeError:
        # Try to extract scores from malformed response
        scores = {}
        for dim in ['resource_validity', 'task_method_consistency', 'task_experiment_consistency']:
            match = re.search(rf'"{dim}".*?"score":\s*(\d)', response)
            scores[dim] = {
                "score": int(match.group(1)) if match else 3,
                "justification": "Parse error"
            }
        return scores, cost


def judge_sample(
    sample: Dict,
    sample_idx: int,
    model: str,
    dry_run: bool
) -> Tuple[int, Dict, float]:
    """Judge a single sample. Returns (sample_idx, result, cost)."""
    raw_prediction = sample.get('prediction', '')
    
    # Strip reasoning to get only the proposal content
    proposal = strip_reasoning_from_proposal(raw_prediction)
    
    scores, cost = judge_proposal(proposal, model, dry_run)
    
    result = {
        "id": sample.get('id', sample_idx),
        "tree_id": sample.get('tree_id', ''),
        "root_paper_id": sample.get('root_paper_id', ''),
        "scores": scores
    }
    
    return sample_idx, result, cost


def run_evaluation(
    predictions: List[Dict],
    output_path: str,
    model: str = "gpt-4.1-mini",
    num_workers: int = 8,
    dry_run: bool = False,
    limit: Optional[int] = None
):
    """Run evaluation on all predictions."""
    
    if limit:
        predictions = predictions[:limit]
    
    all_results = [None] * len(predictions)
    total_cost = 0
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(judge_sample, sample, idx, model, dry_run): idx
            for idx, sample in enumerate(predictions)
        }
        
        with tqdm(total=len(futures), desc="Judging") as pbar:
            for future in as_completed(futures):
                idx, result, cost = future.result()
                all_results[idx] = result
                total_cost += cost
                pbar.update(1)
                pbar.set_postfix(cost=f"${total_cost:.4f}")
    
    # Calculate statistics
    dims = ['resource_validity', 'task_method_consistency', 'task_experiment_consistency']
    stats = {dim: [] for dim in dims}
    
    for result in all_results:
        if result and 'scores' in result:
            for dim in dims:
                if dim in result['scores']:
                    stats[dim].append(result['scores'][dim]['score'])
    
    summary = {
        "model": model,
        "num_samples": len(predictions),
        "total_cost": total_cost,
        "dimension_stats": {}
    }
    
    for dim in dims:
        scores = stats[dim]
        if scores:
            summary["dimension_stats"][dim] = {
                "mean": sum(scores) / len(scores),
                "min": min(scores),
                "max": max(scores),
                "distribution": {i: scores.count(i) for i in range(1, 6)}
            }
    
    output_data = {
        "summary": summary,
        "results": all_results
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nSaved to {output_path}")
    print(f"Total cost: ${total_cost:.4f}")
    
    print(f"\n=== Dimension Scores (Mean) ===")
    for dim in dims:
        if dim in summary["dimension_stats"]:
            mean = summary["dimension_stats"][dim]["mean"]
            print(f"  {dim}: {mean:.2f}")
    
    return output_data


def estimate_cost(predictions: List[Dict], model: str = "gpt-4.1-mini"):
    """Estimate the cost without running."""
    total_samples = len(predictions)
    
    # Estimate tokens per proposal (after stripping reasoning)
    total_chars = 0
    for p in predictions:
        stripped = strip_reasoning_from_proposal(p.get('prediction', ''))
        total_chars += len(stripped)
    avg_proposal_chars = total_chars / total_samples
    input_tokens_per_sample = avg_proposal_chars / 4 + 600  # prompt template
    output_tokens_per_sample = 300  # JSON response
    
    total_input = total_samples * input_tokens_per_sample
    total_output = total_samples * output_tokens_per_sample
    
    pricing = {
        "gpt-4.1": (2.0, 8.0),
        "gpt-4.1-mini": (0.4, 1.6),
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4o": (2.50, 10.0)
    }
    
    in_price, out_price = pricing.get(model, (0.4, 1.6))
    cost = (total_input * in_price + total_output * out_price) / 1_000_000
    
    print(f"\n=== Cost Estimate for {model} ===")
    print(f"Total samples: {total_samples}")
    print(f"Avg proposal length: {int(avg_proposal_chars):,} chars")
    print(f"Estimated input tokens: {int(total_input):,}")
    print(f"Estimated output tokens: {int(total_output):,}")
    print(f"Estimated cost: ${cost:.2f}")
    
    return cost


def main():
    parser = argparse.ArgumentParser(description="Multi-dimensional LLM judge for proposal quality")
    parser.add_argument(
        "--predictions", "-p",
        default=None,
        help="Path to predictions JSON file (single file mode)"
    )
    parser.add_argument(
        "--test-set", "-t",
        default=None,
        help="Path to test set JSONL (combined mode: loads from predictions_full dirs)"
    )
    parser.add_argument(
        "--model-variant", "-mv",
        default="qwen-14b-tuned-stepwise-cot-v4",
        help="Model variant name for combined mode"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output path for evaluation results"
    )
    parser.add_argument(
        "--model", "-m",
        default="gpt-4.1-mini",
        choices=["gpt-4.1", "gpt-4.1-mini", "gpt-4o-mini", "gpt-4o"],
        help="Model to use for judging"
    )
    parser.add_argument(
        "--num-workers", "-w",
        type=int,
        default=8,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit number of samples to evaluate"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Estimate cost without making API calls"
    )
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Only show cost estimate, don't run"
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt"
    )
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent
    
    # Load predictions
    if args.test_set:
        # Combined mode: load from predictions_full directories
        test_path = args.test_set if os.path.isabs(args.test_set) else str(base_dir / args.test_set)
        pred_dirs = [
            str(base_dir / "baselines/predictions_full"),
            str(base_dir / "baselines/predictions_full_batch2")
        ]
        print(f"Loading predictions for test set: {test_path}")
        predictions = load_predictions_for_test_set(test_path, pred_dirs, args.model_variant)
        
        if args.output:
            output_path = args.output if os.path.isabs(args.output) else str(base_dir / args.output)
        else:
            output_path = str(base_dir / f"analyze/judge_{args.model_variant}_819.json")
    elif args.predictions:
        # Single file mode
        pred_path = args.predictions if os.path.isabs(args.predictions) else str(base_dir / args.predictions)
        print(f"Loading predictions from {pred_path}...")
        predictions = load_predictions(pred_path)
        
        if args.output:
            output_path = args.output if os.path.isabs(args.output) else str(base_dir / args.output)
        else:
            output_path = pred_path.replace('.json', '_judged.json')
    else:
        print("Error: Must specify either --predictions or --test-set")
        return
    
    if args.limit:
        predictions = predictions[:args.limit]
    
    print(f"Loaded {len(predictions)} predictions")
    
    cost = estimate_cost(predictions, args.model)
    
    if args.estimate_only:
        return
    
    if not args.dry_run and not args.yes:
        confirm = input(f"\nProceed with evaluation? (estimated cost: ${cost:.2f}) [y/N]: ")
        if confirm.lower() != 'y':
            print("Aborted.")
            return
    
    run_evaluation(
        predictions=predictions,
        output_path=output_path,
        model=args.model,
        num_workers=args.num_workers,
        dry_run=args.dry_run,
        limit=args.limit
    )


if __name__ == "__main__":
    main()
