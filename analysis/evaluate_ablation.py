#!/usr/bin/env python3
"""
Evaluate ablation experiments with comparison to baseline.

For each ablation config (e.g., no_background):
1. Evaluate the ablation predictions
2. Find corresponding baseline predictions for the SAME samples
3. Compute scores for both on the same subset
4. Filter to samples that HAVE the removed citation type (so ablation matters)
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Set
from dataclasses import asdict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from evaluation.evaluate import (
    load_corpus,
    BM25Retriever,
    EmbeddingRetriever,
    evaluate_prediction,
    compute_aggregate_metrics,
    strip_reasoning_from_proposal,
    EvaluationResult,
)


def load_citation_types(path: str = "analyze/citation_types_full.json") -> Dict[str, Set[str]]:
    """Load citation types and return mapping of tree_id -> set of citation types."""
    with open(path, 'r') as f:
        data = json.load(f)
    
    result = {}
    for sample in data['samples']:
        tree_id = sample['tree_id']
        types = set()
        for paper in sample['papers']:
            types.update(paper.get('citation_types', []))
        result[tree_id] = types
    
    return result


def result_to_dict(result: EvaluationResult) -> Dict:
    """Convert EvaluationResult dataclass to JSON-serializable dict."""
    d = asdict(result)
    # Handle None values and nested dataclasses
    if d.get('subfield_scores'):
        d['subfield_scores'] = [asdict(s) if s else None for s in result.subfield_scores or []]
    if d.get('max_subfield_scores'):
        d['max_subfield_scores'] = asdict(result.max_subfield_scores) if result.max_subfield_scores else None
    if d.get('avg_subfield_scores'):
        d['avg_subfield_scores'] = asdict(result.avg_subfield_scores) if result.avg_subfield_scores else None
    if d.get('true_root_subfield_scores'):
        d['true_root_subfield_scores'] = asdict(result.true_root_subfield_scores) if result.true_root_subfield_scores else None
    if d.get('gap_groundedness'):
        d['gap_groundedness'] = asdict(result.gap_groundedness) if result.gap_groundedness else None
    return d


def load_ablation_predictions(path: str) -> Dict:
    """Load ablation predictions with metadata."""
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def load_baseline_predictions(path: str, tree_ids: Set[str]) -> List[Dict]:
    """Load baseline predictions filtered to specific tree_ids."""
    with open(path, 'r') as f:
        data = json.load(f)
    
    predictions = data.get('predictions', [])
    filtered = [p for p in predictions if p.get('tree_id', '') in tree_ids]
    return filtered


def evaluate_predictions(
    predictions: List[Dict],
    retriever,
    corpus: List[Dict],
    top_k: int = 10,
    judge_model: str = "gpt-4.1-mini",
    judge_top_n: int = 5,
    num_workers: int = 8,
    strip_reasoning: bool = True,
    subfield_scores: bool = False,
    desc: str = "Evaluating"
) -> Dict:
    """Evaluate a list of predictions and return metrics."""
    
    # Strip reasoning if needed
    if strip_reasoning:
        for pred in predictions:
            original = pred.get('prediction', '')
            pred['prediction'] = strip_reasoning_from_proposal(original)
    
    # Evaluate each prediction
    results = []
    
    def eval_single(pred):
        return evaluate_prediction(
            prediction=pred,
            retriever=retriever,
            corpus=corpus,
            top_k=top_k,
            judge_model=judge_model,
            judge_top_n=judge_top_n,
            include_subfields=subfield_scores,
        )
    
    if num_workers > 1:
        results = [None] * len(predictions)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_idx = {
                executor.submit(eval_single, pred): idx
                for idx, pred in enumerate(predictions)
            }
            for future in tqdm(as_completed(future_to_idx), total=len(predictions), desc=desc):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"Error evaluating prediction {idx}: {e}")
    else:
        for pred in tqdm(predictions, desc=desc):
            results.append(eval_single(pred))
    
    # Remove None results and pair with predictions for ID tracking
    valid_results = []
    valid_predictions = []
    for i, r in enumerate(results):
        if r is not None:
            valid_results.append(r)
            valid_predictions.append(predictions[i])
    
    # Compute metrics
    metrics = compute_aggregate_metrics(valid_results, include_subfields=subfield_scores)
    
    # Build individual scores list
    individual_scores = []
    for pred, result in zip(valid_predictions, valid_results):
        individual_scores.append({
            "tree_id": pred.get('tree_id', ''),
            "root_title": pred.get('root_title', ''),
            "max_llm_score": result.max_llm_score,
            "avg_llm_score": result.avg_llm_score,
            "true_root_rank": result.true_root_rank,
            "true_root_llm_score": result.true_root_llm_score,
            "cost_usd": result.cost_usd,
        })
    
    return {
        "num_samples": len(valid_results),
        "metrics": metrics,
        "results": valid_results,
        "individual_scores": individual_scores
    }


def run_ablation_evaluation(
    ablation_path: str,
    baseline_path: str,
    corpus_paths: List[str],
    output_path: str,
    citation_types_path: str = "analyze/citation_types_full.json",
    retriever_type: str = "embedding",
    embedding_model: str = "text-embedding-3-large",
    top_k: int = 10,
    judge_model: str = "gpt-4.1-mini",
    judge_top_n: int = 5,
    num_workers: int = 8,
    filter_has_removed_type: bool = True,
    subfield_scores: bool = False,
):
    """Run evaluation comparing ablation to baseline."""
    
    # Load ablation predictions
    logger.info(f"Loading ablation predictions from {ablation_path}...")
    ablation_data = load_ablation_predictions(ablation_path)
    ablation_config = ablation_data['config']
    ablation_predictions = ablation_data['predictions']
    
    logger.info(f"Ablation config: {ablation_config['name']}")
    remove_types = ablation_config['remove_types']
    logger.info(f"Remove types: {remove_types}")
    logger.info(f"Total samples: {len(ablation_predictions)}")
    
    # Load citation types to filter samples that have the removed type
    logger.info(f"Loading citation types from {citation_types_path}...")
    citation_types_by_tree = load_citation_types(citation_types_path)
    
    # Filter to samples that HAVE the removed citation type(s)
    # This ensures we're only evaluating samples where the ablation matters
    if filter_has_removed_type:
        original_count = len(ablation_predictions)
        ablation_predictions = [
            p for p in ablation_predictions 
            if any(rt in citation_types_by_tree.get(p['tree_id'], set()) for rt in remove_types)
        ]
        logger.info(f"Filtered to samples with removed type(s) {remove_types}: {len(ablation_predictions)} / {original_count}")
    
    # Get tree_ids for baseline matching
    tree_ids = set(p['tree_id'] for p in ablation_predictions)
    
    # Load baseline predictions for same samples
    logger.info(f"Loading baseline predictions from {baseline_path}...")
    baseline_predictions = load_baseline_predictions(baseline_path, tree_ids)
    logger.info(f"Matched baseline samples: {len(baseline_predictions)}")
    
    # Load corpus
    logger.info(f"Loading corpus from {corpus_paths}...")
    corpus = load_corpus(corpus_paths)
    logger.info(f"Corpus size: {len(corpus)}")
    
    # Initialize retriever
    if retriever_type == "bm25":
        retriever = BM25Retriever(corpus, text_field="structured")
    else:
        # Use the same cache as the main evaluation script
        cache_path = f"evaluation/cache/{embedding_model.replace('-', '_').replace('/', '_')}_corpus_embeddings.npy"
        retriever = EmbeddingRetriever(
            corpus,
            model=embedding_model,
            text_field="structured",
            cache_path=cache_path
        )
    
    # Evaluate ablation predictions
    logger.info(f"\n{'='*50}")
    logger.info(f"Evaluating ABLATION ({ablation_config['name']})...")
    logger.info(f"{'='*50}")
    
    ablation_results = evaluate_predictions(
        predictions=ablation_predictions,
        retriever=retriever,
        corpus=corpus,
        top_k=top_k,
        judge_model=judge_model,
        judge_top_n=judge_top_n,
        num_workers=num_workers,
        subfield_scores=subfield_scores,
        desc=f"Ablation ({ablation_config['name']})"
    )
    
    # Evaluate baseline predictions
    logger.info(f"\n{'='*50}")
    logger.info(f"Evaluating BASELINE (full) on same samples...")
    logger.info(f"{'='*50}")
    
    baseline_results = evaluate_predictions(
        predictions=baseline_predictions,
        retriever=retriever,
        corpus=corpus,
        top_k=top_k,
        judge_model=judge_model,
        judge_top_n=judge_top_n,
        num_workers=num_workers,
        subfield_scores=subfield_scores,
        desc="Baseline (full)"
    )
    
    # Compare metrics
    logger.info(f"\n{'='*50}")
    logger.info("COMPARISON")
    logger.info(f"{'='*50}")
    
    def get_metric(results, key):
        return results['metrics'].get(key, 0)
    
    comparison = {
        "ablation_config": ablation_config,
        "filter_has_removed_type": filter_has_removed_type,
        "num_samples": {
            "ablation": ablation_results['num_samples'],
            "baseline": baseline_results['num_samples']
        },
        "metrics": {}
    }
    
    key_metrics = ['avg_max_llm_score', 'recall_at_k', 'true_root_top1_rate']
    for metric in key_metrics:
        abl_val = get_metric(ablation_results, metric)
        base_val = get_metric(baseline_results, metric)
        diff = abl_val - base_val
        comparison['metrics'][metric] = {
            "ablation": abl_val,
            "baseline": base_val,
            "diff": diff,
            "pct_change": (diff / base_val * 100) if base_val else 0
        }
        logger.info(f"{metric}:")
        logger.info(f"  Ablation: {abl_val:.4f}")
        logger.info(f"  Baseline: {base_val:.4f}")
        logger.info(f"  Diff: {diff:+.4f} ({comparison['metrics'][metric]['pct_change']:+.1f}%)")
    
    # Build paired individual scores (ablation vs baseline for same sample)
    ablation_scores_by_tree = {s['tree_id']: s for s in ablation_results['individual_scores']}
    baseline_scores_by_tree = {s['tree_id']: s for s in baseline_results['individual_scores']}
    
    paired_scores = []
    for tree_id in ablation_scores_by_tree:
        if tree_id in baseline_scores_by_tree:
            abl_s = ablation_scores_by_tree[tree_id]
            base_s = baseline_scores_by_tree[tree_id]
            paired_scores.append({
                "tree_id": tree_id,
                "root_title": abl_s['root_title'],
                "ablation_max_llm_score": abl_s['max_llm_score'],
                "baseline_max_llm_score": base_s['max_llm_score'],
                "score_diff": abl_s['max_llm_score'] - base_s['max_llm_score'],
            })
    
    # Sort by score difference (most negative = most affected by ablation)
    paired_scores.sort(key=lambda x: x['score_diff'])
    
    logger.info(f"\nTop 5 most affected samples (score dropped):")
    for s in paired_scores[:5]:
        logger.info(f"  {s['root_title'][:50]}: {s['baseline_max_llm_score']:.2f} -> {s['ablation_max_llm_score']:.2f} ({s['score_diff']:+.2f})")
    
    # Save results
    output_data = {
        "config": {
            "ablation_path": ablation_path,
            "baseline_path": baseline_path,
            "corpus_paths": corpus_paths,
            "retriever_type": retriever_type,
            "filter_has_removed_type": filter_has_removed_type,
            "remove_types": remove_types
        },
        "comparison": comparison,
        "ablation_results": {
            "num_samples": ablation_results['num_samples'],
            "metrics": ablation_results['metrics'],
            "individual_scores": ablation_results['individual_scores']
        },
        "baseline_results": {
            "num_samples": baseline_results['num_samples'],
            "metrics": baseline_results['metrics'],
            "individual_scores": baseline_results['individual_scores']
        },
        "paired_scores": paired_scores
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"\nResults saved to {output_path}")
    
    return output_data


def main():
    parser = argparse.ArgumentParser(description="Evaluate ablation experiments")
    parser.add_argument(
        "--ablation", "-a",
        required=True,
        help="Path to ablation predictions JSON"
    )
    parser.add_argument(
        "--baseline", "-b",
        default="analyze/predictions_819/qwen-14b-stepwise-cot.json",
        help="Path to baseline (full) predictions JSON"
    )
    parser.add_argument(
        "--corpus", "-c",
        nargs='+',
        default=["data/paper_structured/neurips_2025.json", "data/paper_structured/icml_2025.json",
                 "data/paper_structured/iclr_2025.json"],
        help="Corpus JSON files"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output path (default: ablation_path with _eval suffix)"
    )
    parser.add_argument(
        "--retriever", "-r",
        choices=["bm25", "embedding"],
        default="embedding",
        help="Retriever type"
    )
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-large",
        help="Embedding model: text-embedding-3-large (OpenAI, default), bge-large-en-v1.5 (local)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of papers to retrieve"
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-4.1-mini",
        help="Model for LLM judge"
    )
    parser.add_argument(
        "--judge-top-n",
        type=int,
        default=5,
        help="Only judge top-N retrieved papers"
    )
    parser.add_argument(
        "--num-workers", "-w",
        type=int,
        default=8,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--citation-types-path",
        default="analyze/citation_types_full.json",
        help="Path to citation types JSON file"
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Don't filter to samples that have the removed citation type (evaluate all)"
    )
    parser.add_argument(
        "--subfield-scores",
        action="store_true",
        help="Compute subfield-level scores"
    )
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent
    
    # Resolve paths
    ablation_path = args.ablation if os.path.isabs(args.ablation) else str(base_dir / args.ablation)
    baseline_path = args.baseline if os.path.isabs(args.baseline) else str(base_dir / args.baseline)
    corpus_paths = [c if os.path.isabs(c) else str(base_dir / c) for c in args.corpus]
    
    if args.output:
        output_path = args.output if os.path.isabs(args.output) else str(base_dir / args.output)
    else:
        output_path = ablation_path.replace('.json', '_eval.json')
    
    citation_types_path = args.citation_types_path if os.path.isabs(args.citation_types_path) else str(base_dir / args.citation_types_path)
    
    run_ablation_evaluation(
        ablation_path=ablation_path,
        baseline_path=baseline_path,
        corpus_paths=corpus_paths,
        output_path=output_path,
        citation_types_path=citation_types_path,
        retriever_type=args.retriever,
        embedding_model=args.embedding_model,
        top_k=args.top_k,
        judge_model=args.judge_model,
        judge_top_n=args.judge_top_n,
        num_workers=args.num_workers,
        filter_has_removed_type=not args.no_filter,
        subfield_scores=args.subfield_scores,
    )


if __name__ == "__main__":
    main()
