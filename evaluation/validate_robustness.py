#!/usr/bin/env python3
"""
Validate robustness of evaluation metrics across different configurations.

Tests robustness to:
1. Retrieval depth (top-k): 5, 10, 20
2. Judge model: gpt-4.1-mini, gpt-4.1
3. Retriever type: bm25, embedding

Usage:
    python evaluation/validate_robustness.py --num-samples 100
    python evaluation/validate_robustness.py --num-samples 100 --skip-expensive
"""

import os
import sys
import json
import argparse
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from evaluation.evaluate import (
    load_corpus,
    load_predictions,
    BM25Retriever,
    EmbeddingRetriever,
    evaluate_prediction,
    compute_aggregate_metrics,
    strip_reasoning_from_proposal,
)


@dataclass
class RobustnessConfig:
    """Configuration for a single robustness test."""
    name: str
    retriever_type: str
    top_k: int
    judge_model: str
    embedding_model: str = "text-embedding-3-large"
    fast_judge: bool = False  # If True, skip reasoning in judge output


# Fast judge prompt (score only, no reasoning - saves ~60% output tokens)
FAST_JUDGE_PROMPT = """Rate semantic similarity between a research proposal and paper on 0-10 scale.

## Proposal:
{proposal}

## Paper:
{candidate_summary}

Scale: 0=unrelated, 5=related, 7=very similar, 10=identical
Respond with ONLY: {{"score": <number>}}"""


# Default configurations to test
DEFAULT_CONFIGS = [
    # Baseline (matches original evaluation settings)
    RobustnessConfig("baseline", "embedding", 10, "gpt-4.1-mini", "text-embedding-3-large"),
    
    # Vary top-k
    RobustnessConfig("top_k_5", "embedding", 5, "gpt-4.1-mini", "text-embedding-3-large"),
    
    # Vary embedding model
    RobustnessConfig("embed_small", "embedding", 10, "gpt-4.1-mini", "text-embedding-3-small"),
    
    # Vary judge model
    RobustnessConfig("judge_gpt4o_mini", "embedding", 10, "gpt-4o-mini", "text-embedding-3-large"),
]

# Cheap config for quick verification
CHEAP_CONFIGS = [
    RobustnessConfig("top_k_5", "embedding", 5, "gpt-4.1-mini", "text-embedding-3-large"),
]


# Pricing per 1M tokens (as of 2026)
PRICING = {
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "text-embedding-3-large": {"input": 0.13},
    "text-embedding-3-small": {"input": 0.02},
}


def estimate_cost(
    num_samples: int,
    num_models: int,
    configs: List[RobustnessConfig],
    judge_top_n: int = 5,
) -> Dict[str, float]:
    """Estimate cost for robustness validation."""
    
    # Average tokens per judge call
    avg_input_tokens = 1500  # proposal + retrieved paper
    avg_output_tokens = 50   # score + brief justification
    
    costs = {}
    total_cost = 0.0
    
    for config in configs:
        # Judge calls: num_samples * num_models * judge_top_n
        num_judge_calls = num_samples * num_models * judge_top_n
        
        judge_price = PRICING.get(config.judge_model, PRICING["gpt-4.1-mini"])
        input_cost = (num_judge_calls * avg_input_tokens / 1_000_000) * judge_price["input"]
        output_cost = (num_judge_calls * avg_output_tokens / 1_000_000) * judge_price["output"]
        config_cost = input_cost + output_cost
        
        costs[config.name] = {
            "judge_calls": num_judge_calls,
            "judge_model": config.judge_model,
            "cost_usd": config_cost
        }
        total_cost += config_cost
    
    costs["total"] = total_cost
    return costs


def sample_predictions(predictions_path: str, num_samples: int, seed: int = 42) -> Tuple[List[Dict], List[int]]:
    """Sample a subset of predictions. Returns (sampled_predictions, selected_indices)."""
    predictions = load_predictions(predictions_path)
    random.seed(seed)
    
    if len(predictions) <= num_samples:
        return predictions, list(range(len(predictions)))
    
    indices = sorted(random.sample(range(len(predictions)), num_samples))
    sampled = [predictions[i] for i in indices]
    return sampled, indices


def run_evaluation(
    predictions: List[Dict],
    corpus: List[Dict],
    config: RobustnessConfig,
    num_workers: int = 8,
) -> Dict:
    """Run evaluation with a specific configuration."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    # Initialize retriever
    if config.retriever_type == "bm25":
        retriever = BM25Retriever(corpus, text_field="structured")
    else:
        # Use model-specific cache path
        # IMPORTANT: text-embedding-3-large should use the original cache to match
        # the original evaluation (which used unlabeled corpus_embeddings.npy)
        if config.embedding_model == "text-embedding-3-large":
            # Use labeled cache that matches original evaluation dimensions
            cache_path = "evaluation/cache/text_embedding_3_large_corpus_embeddings.npy"
        else:
            cache_name = config.embedding_model.replace("-", "_").replace("/", "_")
            cache_path = f"evaluation/cache/{cache_name}_corpus_embeddings.npy"
        retriever = EmbeddingRetriever(
            corpus,
            model=config.embedding_model,
            text_field="structured",
            cache_path=cache_path
        )
    
    # Strip reasoning from predictions
    for pred in predictions:
        original = pred.get('prediction', '')
        pred['prediction'] = strip_reasoning_from_proposal(original)
    
    def eval_single(pred):
        """Evaluate a single prediction."""
        return evaluate_prediction(
            prediction=pred,
            retriever=retriever,
            corpus=corpus,
            top_k=config.top_k,
            judge_model=config.judge_model,
            judge_top_n=5,
            include_subfields=True,
        )
    
    # Parallel evaluation
    results = [None] * len(predictions)
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_idx = {
            executor.submit(eval_single, pred): idx
            for idx, pred in enumerate(predictions)
        }
        
        for future in tqdm(as_completed(future_to_idx), total=len(predictions), desc=f"Eval ({config.name})"):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.warning(f"Error evaluating prediction {idx}: {e}")
    
    # Filter out None results
    results = [r for r in results if r is not None]
    
    # Compute metrics
    metrics = compute_aggregate_metrics(results, include_subfields=True)
    
    return {
        "config": asdict(config),
        "num_samples": len(results),
        "metrics": metrics,
        "individual_scores": [r.max_llm_score for r in results]
    }


def compute_correlation(scores1: List[float], scores2: List[float]) -> Tuple[float, float]:
    """Compute Pearson and Spearman correlation between two score lists."""
    from scipy import stats
    
    # Need at least 2 samples for correlation
    if len(scores1) < 2 or len(scores2) < 2:
        return float('nan'), float('nan')
    
    # Check for constant arrays (no variance)
    if np.std(scores1) == 0 or np.std(scores2) == 0:
        return float('nan'), float('nan')
    
    pearson_r, pearson_p = stats.pearsonr(scores1, scores2)
    spearman_r, spearman_p = stats.spearmanr(scores1, scores2)
    
    return pearson_r, spearman_r


def run_robustness_validation(
    predictions_paths: Dict[str, str],
    corpus_paths: List[str],
    output_dir: str,
    num_samples: int = 100,
    configs: Optional[List[RobustnessConfig]] = None,
    seed: int = 42,
    num_workers: int = 8,
    estimate_only: bool = False,
):
    """
    Run robustness validation across multiple configurations.
    
    Args:
        predictions_paths: Dict mapping model name to predictions file path
        corpus_paths: List of corpus file paths
        output_dir: Directory to save results
        num_samples: Number of samples to evaluate per model
        num_workers: Number of parallel workers for evaluation
        estimate_only: If True, only show cost estimate and exit
        configs: List of configurations to test (defaults to DEFAULT_CONFIGS)
        seed: Random seed for sampling
    """
    configs = configs or DEFAULT_CONFIGS
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Robustness validation with {num_samples} samples per model")
    logger.info(f"Models: {list(predictions_paths.keys())}")
    logger.info(f"Configurations: {[c.name for c in configs]}")
    
    # Estimate cost
    num_models = len(predictions_paths)
    cost_estimate = estimate_cost(num_samples, num_models, configs)
    
    logger.info(f"\n{'='*60}")
    logger.info("COST ESTIMATION")
    logger.info(f"{'='*60}")
    for config_name, info in cost_estimate.items():
        if config_name == "total":
            continue
        logger.info(f"  {config_name}: {info['judge_calls']} calls ({info['judge_model']}) = ${info['cost_usd']:.2f}")
    logger.info(f"  TOTAL ESTIMATED COST: ${cost_estimate['total']:.2f}")
    logger.info(f"{'='*60}\n")
    
    if estimate_only:
        logger.info("Estimate only mode - exiting without running evaluation")
        return cost_estimate
    
    # Load corpus (with titles, matching original evaluation)
    logger.info("Loading corpus...")
    titles_paths = [
        "data/accepted_papers/NeurIPS.cc_2025.json",
        "data/accepted_papers/ICML.cc_2025.json",
        "data/accepted_papers/ICLR.cc_2025.json",
    ]
    existing_titles = [t for t in titles_paths if os.path.exists(t)]
    corpus = load_corpus(corpus_paths, titles_path=existing_titles if existing_titles else None)
    logger.info(f"Corpus size: {len(corpus)}")
    
    # Sample predictions for each model
    logger.info(f"Sampling {num_samples} predictions per model...")
    sampled_predictions = {}
    sample_indices = {}
    for model_name, path in predictions_paths.items():
        sampled, indices = sample_predictions(path, num_samples, seed)
        sampled_predictions[model_name] = sampled
        sample_indices[model_name] = indices
        logger.info(f"  {model_name}: {len(sampled)} samples (from {len(load_predictions(path))} total)")
    
    # Save sample indices for reproducibility / running remaining samples later
    sample_info_path = output_path / "sample_indices.json"
    sample_info = {
        "seed": seed,
        "num_samples": num_samples,
        "models": {
            model_name: {
                "source_file": predictions_paths[model_name],
                "total_predictions": len(load_predictions(predictions_paths[model_name])),
                "sampled_count": len(indices),
                "sampled_indices": indices,
                "sampled_tree_ids": [p.get("tree_id", "") for p in sampled_predictions[model_name]],
            }
            for model_name, indices in sample_indices.items()
        }
    }
    with open(sample_info_path, 'w') as f:
        json.dump(sample_info, f, indent=2)
    logger.info(f"Saved sample indices to {sample_info_path}")
    
    # Run evaluation for each config x model combination
    all_results = {}
    
    for config in configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Configuration: {config.name}")
        logger.info(f"  Retriever: {config.retriever_type}, Top-k: {config.top_k}, Judge: {config.judge_model}")
        logger.info(f"{'='*60}")
        
        config_results = {}
        
        for model_name, predictions in sampled_predictions.items():
            logger.info(f"\nEvaluating {model_name}...")
            
            # Deep copy predictions to avoid mutation issues
            predictions_copy = [dict(p) for p in predictions]
            
            result = run_evaluation(
                predictions=predictions_copy,
                corpus=corpus,
                config=config,
                num_workers=num_workers,
            )
            
            config_results[model_name] = result
            logger.info(f"  avg_max_llm_score: {result['metrics']['avg_max_llm_score']:.3f}")
        
        all_results[config.name] = config_results
    
    # Analyze robustness
    logger.info(f"\n{'='*60}")
    logger.info("ROBUSTNESS ANALYSIS")
    logger.info(f"{'='*60}")
    
    # Print results table
    model_names = list(predictions_paths.keys())
    
    print("\n### Scores by Configuration")
    print(f"{'Config':<25}", end="")
    for model in model_names:
        print(f"{model:<20}", end="")
    print()
    print("-" * (25 + 20 * len(model_names)))
    
    for config_name, config_results in all_results.items():
        print(f"{config_name:<25}", end="")
        for model in model_names:
            score = config_results[model]['metrics']['avg_max_llm_score']
            print(f"{score:<20.3f}", end="")
        print()
    
    # Correlation analysis with baseline (per model)
    print("\n### Correlation with Baseline Configuration (per model)")
    baseline_config = "baseline"
    if baseline_config in all_results:
        for model in model_names:
            print(f"\n{model}:")
            print(f"  {'Config':<20} {'Pearson r':<12} {'Spearman r':<12}")
            print(f"  {'-'*44}")
            
            baseline_scores = all_results[baseline_config][model]['individual_scores']
            
            for config_name, config_results in all_results.items():
                if config_name == baseline_config:
                    continue
                
                config_scores = config_results[model]['individual_scores']
                
                if len(baseline_scores) >= 2 and len(config_scores) >= 2:
                    pearson_r, spearman_r = compute_correlation(baseline_scores, config_scores)
                    if np.isnan(pearson_r):
                        print(f"  {config_name:<20} {'N/A':<12} {'N/A':<12}")
                    else:
                        print(f"  {config_name:<20} {pearson_r:<12.3f} {spearman_r:<12.3f}")
                else:
                    print(f"  {config_name:<20} {'N/A (< 2 samples)':<24}")
    
    # Ranking consistency
    print("\n### Ranking Consistency")
    print("(Do relative model rankings stay the same?)")
    
    rankings = {}
    for config_name, config_results in all_results.items():
        scores = [(model, config_results[model]['metrics']['avg_max_llm_score']) 
                  for model in model_names]
        scores.sort(key=lambda x: x[1], reverse=True)
        ranking = [model for model, _ in scores]
        rankings[config_name] = ranking
        print(f"{config_name:<20}: {' > '.join(ranking)}")
    
    # Check if rankings are consistent
    baseline_ranking = rankings.get("baseline", [])
    consistent = all(rankings[c] == baseline_ranking for c in rankings)
    print(f"\nRankings consistent across all configs: {'✓ YES' if consistent else '✗ NO'}")
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = output_path / f"robustness_validation_{timestamp}.json"
    
    # Convert individual_scores to serializable format
    serializable_results = {}
    for config_name, config_results in all_results.items():
        serializable_results[config_name] = {}
        for model_name, result in config_results.items():
            serializable_results[config_name][model_name] = {
                "config": result["config"],
                "num_samples": result["num_samples"],
                "metrics": result["metrics"],
                "individual_scores": result["individual_scores"]
            }
    
    output_data = {
        "validation_config": {
            "num_samples": num_samples,
            "seed": seed,
            "models": list(predictions_paths.keys()),
            "configurations": [asdict(c) for c in configs]
        },
        "results": serializable_results,
        "summary": {
            "rankings_consistent": consistent,
            "rankings_by_config": rankings
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_file}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Validate robustness of evaluation metrics")
    
    parser.add_argument(
        "--predictions",
        nargs='+',
        default=[
            "analyze/predictions_819/qwen-14b-stepwise-cot.json",
            "analyze/predictions_819/qwen-14b-ncot.json",
        ],
        help="Paths to prediction files (model name inferred from filename)"
    )
    parser.add_argument(
        "--corpus",
        nargs='+',
        default=[
            "data/paper_structured/neurips_2025.json",
            "data/paper_structured/icml_2025.json",
            "data/paper_structured/iclr_2025.json"
        ],
        help="Corpus JSON files"
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation/robustness",
        help="Output directory for results"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples per model (default: 100)"
    )
    parser.add_argument(
        "--skip-expensive",
        action="store_true",
        help="Skip expensive configurations (e.g., gpt-4.1 judge)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling"
    )
    parser.add_argument(
        "--num-workers", "-w",
        type=int,
        default=8,
        help="Number of parallel workers for evaluation (default: 8)"
    )
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Only estimate cost, don't run evaluation"
    )
    
    args = parser.parse_args()
    
    # Build predictions paths dict
    predictions_paths = {}
    for path in args.predictions:
        # Extract model name from filename
        name = Path(path).stem.replace('predictions_', '').replace('_819', '')
        predictions_paths[name] = path
    
    # Select configs
    configs = CHEAP_CONFIGS if args.skip_expensive else DEFAULT_CONFIGS
    
    # Run validation
    run_robustness_validation(
        predictions_paths=predictions_paths,
        corpus_paths=args.corpus,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        configs=configs,
        seed=args.seed,
        num_workers=args.num_workers,
        estimate_only=args.estimate_only,
    )


if __name__ == "__main__":
    main()
