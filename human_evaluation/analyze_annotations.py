#!/usr/bin/env python3
"""
Analyze human annotation results for proposal comparison.

Requirements:
- Each proposal pair is evaluated by 3 domain-expert graduate students
- Report win rates with 95% confidence intervals
- Treat ties as half wins when computing aggregate preference scores
- Aggregate judgments using majority vote
- Report percentage of unanimous decisions for inter-annotator consistency
"""
import json
import os
import sys
import math
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


def load_batch_data(batch_path: str) -> Dict:
    """Load batch comparison data."""
    with open(batch_path) as f:
        return json.load(f)


def load_annotations(annotations_dir: str) -> Dict[str, Dict[str, Dict]]:
    """
    Load all annotation files.
    Returns: {batch_letter: {annotator_name: {pair_idx: annotations}}}
    """
    annotations = defaultdict(dict)
    
    for filename in os.listdir(annotations_dir):
        if not filename.startswith("comparison_batch_") or not filename.endswith(".json"):
            continue
        if "_annotations_" not in filename:
            continue
        
        # Parse filename: comparison_batch_A_annotations_zhiyi_shi.json
        parts = filename.replace(".json", "").split("_annotations_")
        batch_part = parts[0]  # comparison_batch_A
        annotator_name = parts[1]  # zhiyi_shi
        batch_letter = batch_part.split("_")[-1]  # A
        
        filepath = os.path.join(annotations_dir, filename)
        with open(filepath) as f:
            data = json.load(f)
        
        annotations[batch_letter][annotator_name] = data
    
    return dict(annotations)


def get_majority_vote(votes: List[str]) -> str:
    """
    Get majority vote from list of votes ('A', 'B', or 'tie').
    Returns: 'A', 'B', or 'tie'
    """
    if not votes:
        return 'tie'
    
    count_a = sum(1 for v in votes if v == 'A')
    count_b = sum(1 for v in votes if v == 'B')
    count_tie = sum(1 for v in votes if v == 'tie')
    
    # Majority vote
    if count_a > count_b and count_a > count_tie:
        return 'A'
    elif count_b > count_a and count_b > count_tie:
        return 'B'
    elif count_tie > count_a and count_tie > count_b:
        return 'tie'
    else:
        # No clear majority - treat as tie
        return 'tie'


def is_unanimous(votes: List[str]) -> bool:
    """Check if all votes are the same."""
    if not votes:
        return False
    return len(set(votes)) == 1


def wilson_confidence_interval(wins: float, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate Wilson score confidence interval for a proportion.
    Handles ties as half-wins.
    """
    if total == 0:
        return (0.0, 0.0)
    
    p = wins / total
    z = 1.96 if confidence == 0.95 else 1.645  # z-score for 95% CI
    
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
    
    lower = max(0, center - spread)
    upper = min(1, center + spread)
    
    return (lower, upper)


def analyze_comparisons(
    batch_data_dir: str,
    annotations_dir: str,
    metrics: List[str] = ['overall', 'soundness', 'excitement']
) -> Dict:
    """
    Analyze all comparisons across batches.
    
    Returns analysis results including:
    - Win rates for each comparison type
    - 95% confidence intervals
    - Percentage of unanimous decisions
    """
    # Load all batch data
    batches = {}
    for filename in sorted(os.listdir(batch_data_dir)):
        if filename.startswith("comparison_batch_") and filename.endswith(".json"):
            if "_annotations_" in filename or "_readability_" in filename:
                continue
            batch_letter = filename.split("_")[2].replace(".json", "")
            batch_path = os.path.join(batch_data_dir, filename)
            batches[batch_letter] = load_batch_data(batch_path)
    
    # Load annotations
    annotations = load_annotations(annotations_dir)
    
    # Results structure
    results = {
        'stepwise_vs_human': {metric: {'wins': 0, 'losses': 0, 'ties': 0, 'total': 0, 'unanimous': 0} for metric in metrics},
        'stepwise_vs_prompting': {metric: {'wins': 0, 'losses': 0, 'ties': 0, 'total': 0, 'unanimous': 0} for metric in metrics},
    }
    
    # Track individual pair results for detailed analysis
    pair_results = []
    
    # Process each batch
    for batch_letter, batch_data in sorted(batches.items()):
        if batch_letter not in annotations:
            print(f"Warning: No annotations found for batch {batch_letter}")
            continue
        
        batch_annotations = annotations[batch_letter]
        annotator_names = list(batch_annotations.keys())
        
        if len(annotator_names) < 3:
            print(f"Warning: Batch {batch_letter} has only {len(annotator_names)} annotators (expected 3)")
        
        pairs = batch_data.get('pairs', [])
        batch_start = batch_data.get('config', {}).get('batch_start', 0)
        
        for local_idx, pair in enumerate(pairs):
            pair_id = pair.get('id', '')
            comparison_type = pair.get('comparison_type', '')
            
            if comparison_type not in results:
                continue
            
            # Determine which side is stepwise/our method
            # For stepwise_vs_human: generated_is_a tells us if stepwise is side A
            # For stepwise_vs_prompting: stepwise_is_a tells us if stepwise is side A
            if comparison_type == 'stepwise_vs_human':
                stepwise_is_a = not pair.get('generated_is_a', True)  # If AI-generated is A, then stepwise is B (actually human is B)
                # Actually let's re-check: generated_is_a means the AI-generated (stepwise) is side A
                # So if generated_is_a is True, stepwise is A
                stepwise_is_a = pair.get('generated_is_a', True)
            else:  # stepwise_vs_prompting
                stepwise_is_a = pair.get('stepwise_is_a', True)
            
            # Collect votes from all annotators for this pair
            for metric in metrics:
                votes = []
                for annotator_name, ann_data in batch_annotations.items():
                    pair_key = str(local_idx)
                    if pair_key in ann_data and metric in ann_data[pair_key]:
                        votes.append(ann_data[pair_key][metric])
                
                if not votes:
                    continue
                
                # Get majority vote
                majority = get_majority_vote(votes)
                unanimous = is_unanimous(votes)
                
                # Convert to stepwise win/loss/tie
                if majority == 'A':
                    if stepwise_is_a:
                        outcome = 'win'
                    else:
                        outcome = 'loss'
                elif majority == 'B':
                    if stepwise_is_a:
                        outcome = 'loss'
                    else:
                        outcome = 'win'
                else:
                    outcome = 'tie'
                
                # Update results
                results[comparison_type][metric]['total'] += 1
                if outcome == 'win':
                    results[comparison_type][metric]['wins'] += 1
                elif outcome == 'loss':
                    results[comparison_type][metric]['losses'] += 1
                else:
                    results[comparison_type][metric]['ties'] += 1
                
                if unanimous:
                    results[comparison_type][metric]['unanimous'] += 1
                
                # Store pair result
                pair_results.append({
                    'pair_id': pair_id,
                    'batch': batch_letter,
                    'comparison_type': comparison_type,
                    'metric': metric,
                    'votes': votes,
                    'majority': majority,
                    'unanimous': unanimous,
                    'stepwise_is_a': stepwise_is_a,
                    'outcome': outcome,
                })
    
    return results, pair_results


def compute_win_rates(results: Dict, metrics: List[str] = ['overall', 'soundness', 'excitement']) -> Dict:
    """
    Compute win rates with 95% CI and unanimous percentages.
    Ties count as 0.5 wins.
    """
    win_rates = {}
    
    for comparison_type, type_results in results.items():
        win_rates[comparison_type] = {}
        
        for metric in metrics:
            data = type_results[metric]
            total = data['total']
            wins = data['wins']
            ties = data['ties']
            losses = data['losses']
            unanimous = data['unanimous']
            
            if total == 0:
                win_rates[comparison_type][metric] = {
                    'win_rate': 0.0,
                    'ci_lower': 0.0,
                    'ci_upper': 0.0,
                    'unanimous_pct': 0.0,
                    'wins': 0,
                    'ties': 0,
                    'losses': 0,
                    'total': 0,
                }
                continue
            
            # Treat ties as half wins
            effective_wins = wins + ties * 0.5
            win_rate = effective_wins / total
            
            # Confidence interval
            ci_lower, ci_upper = wilson_confidence_interval(effective_wins, total)
            
            # Unanimous percentage
            unanimous_pct = unanimous / total * 100
            
            win_rates[comparison_type][metric] = {
                'win_rate': win_rate * 100,
                'ci_lower': ci_lower * 100,
                'ci_upper': ci_upper * 100,
                'unanimous_pct': unanimous_pct,
                'wins': wins,
                'ties': ties,
                'losses': losses,
                'total': total,
            }
    
    return win_rates


def print_results(win_rates: Dict, metrics: List[str] = ['overall', 'soundness', 'excitement']):
    """Print formatted results."""
    print("\n" + "=" * 80)
    print("HUMAN ANNOTATION ANALYSIS RESULTS")
    print("=" * 80)
    
    for comparison_type, type_results in win_rates.items():
        print(f"\n{'─' * 80}")
        comparison_label = comparison_type.replace('_', ' ').title()
        print(f"Comparison: {comparison_label}")
        print(f"{'─' * 80}")
        
        for metric in metrics:
            data = type_results[metric]
            print(f"\n  {metric.upper()}:")
            print(f"    Win Rate: {data['win_rate']:.1f}% (95% CI: [{data['ci_lower']:.1f}%, {data['ci_upper']:.1f}%])")
            print(f"    Wins: {data['wins']}, Ties: {data['ties']}, Losses: {data['losses']} (Total: {data['total']})")
            print(f"    Unanimous Decisions: {data['unanimous_pct']:.1f}%")
    
    print("\n" + "=" * 80)


def generate_latex_table(win_rates: Dict, metrics: List[str] = ['overall', 'soundness', 'excitement']) -> str:
    """Generate LaTeX table for paper."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Human evaluation results. Win rates (\%) with 95\% confidence intervals. Ties count as 0.5 wins.}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Comparison & Overall & Soundness & Excitement \\")
    lines.append(r"\midrule")
    
    for comparison_type, type_results in win_rates.items():
        if comparison_type == 'stepwise_vs_human':
            label = "Stepwise-CoT vs Human"
        else:
            label = "Stepwise-CoT vs Prompting"
        
        cells = [label]
        for metric in metrics:
            data = type_results[metric]
            cell = f"{data['win_rate']:.1f} [{data['ci_lower']:.1f}, {data['ci_upper']:.1f}]"
            cells.append(cell)
        
        lines.append(" & ".join(cells) + r" \\")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)


def analyze_per_annotator(
    batch_data_dir: str,
    annotations_dir: str,
    metrics: List[str] = ['overall', 'soundness', 'excitement']
) -> Dict:
    """Analyze preferences per individual annotator."""
    # Load all batch data
    batches = {}
    for filename in sorted(os.listdir(batch_data_dir)):
        if filename.startswith("comparison_batch_") and filename.endswith(".json"):
            if "_annotations_" in filename or "_readability_" in filename:
                continue
            batch_letter = filename.split("_")[2].replace(".json", "")
            batch_path = os.path.join(batch_data_dir, filename)
            batches[batch_letter] = load_batch_data(batch_path)
    
    # Load annotations
    annotations = load_annotations(annotations_dir)
    
    # Per-annotator results
    annotator_results = defaultdict(lambda: {
        'stepwise_vs_human': {m: {'A': 0, 'B': 0, 'tie': 0, 'stepwise_wins': 0, 'stepwise_losses': 0, 'ties': 0, 'total': 0} for m in metrics},
        'stepwise_vs_prompting': {m: {'A': 0, 'B': 0, 'tie': 0, 'stepwise_wins': 0, 'stepwise_losses': 0, 'ties': 0, 'total': 0} for m in metrics},
    })
    
    # Process each batch
    for batch_letter, batch_data in sorted(batches.items()):
        if batch_letter not in annotations:
            continue
        
        batch_annotations = annotations[batch_letter]
        pairs = batch_data.get('pairs', [])
        
        for local_idx, pair in enumerate(pairs):
            comparison_type = pair.get('comparison_type', '')
            if comparison_type not in ['stepwise_vs_human', 'stepwise_vs_prompting']:
                continue
            
            # Determine which side is stepwise
            if comparison_type == 'stepwise_vs_human':
                stepwise_is_a = pair.get('generated_is_a', True)
            else:
                stepwise_is_a = pair.get('stepwise_is_a', True)
            
            # Process each annotator's vote
            for annotator_name, ann_data in batch_annotations.items():
                pair_key = str(local_idx)
                if pair_key not in ann_data:
                    continue
                
                for metric in metrics:
                    if metric not in ann_data[pair_key]:
                        continue
                    
                    vote = ann_data[pair_key][metric]
                    res = annotator_results[annotator_name][comparison_type][metric]
                    res['total'] += 1
                    res[vote] += 1
                    
                    # Convert to stepwise outcome
                    if vote == 'A':
                        if stepwise_is_a:
                            res['stepwise_wins'] += 1
                        else:
                            res['stepwise_losses'] += 1
                    elif vote == 'B':
                        if stepwise_is_a:
                            res['stepwise_losses'] += 1
                        else:
                            res['stepwise_wins'] += 1
                    else:
                        res['ties'] += 1
    
    return dict(annotator_results)


def print_per_annotator_results(annotator_results: Dict, metrics: List[str] = ['overall']):
    """Print per-annotator preference statistics."""
    print("\n" + "=" * 80)
    print("PER-ANNOTATOR PREFERENCES")
    print("=" * 80)
    
    for annotator_name in sorted(annotator_results.keys()):
        results = annotator_results[annotator_name]
        print(f"\n{'─' * 80}")
        print(f"Annotator: {annotator_name}")
        print(f"{'─' * 80}")
        
        for comparison_type in ['stepwise_vs_human', 'stepwise_vs_prompting']:
            type_label = comparison_type.replace('_', ' ').title()
            print(f"\n  {type_label}:")
            
            for metric in metrics:
                data = results[comparison_type][metric]
                total = data['total']
                if total == 0:
                    continue
                
                wins = data['stepwise_wins']
                losses = data['stepwise_losses']
                ties = data['ties']
                
                # Win rate with ties as 0.5
                effective_wins = wins + ties * 0.5
                win_rate = effective_wins / total * 100 if total > 0 else 0
                
                print(f"    {metric}: Stepwise wins {wins}, ties {ties}, losses {losses} (n={total}) → Win rate: {win_rate:.1f}%")


def main():
    # Paths
    base_dir = Path(__file__).parent
    batch_data_dir = str(base_dir)
    annotations_dir = str(base_dir / "annotations")
    
    metrics = ['overall', 'soundness', 'excitement']
    
    print("Loading annotations...")
    results, pair_results = analyze_comparisons(batch_data_dir, annotations_dir, metrics)
    
    print("Computing win rates...")
    win_rates = compute_win_rates(results, metrics)
    
    # Print results
    print_results(win_rates, metrics)
    
    # Per-annotator analysis
    annotator_results = analyze_per_annotator(batch_data_dir, annotations_dir, metrics)
    print_per_annotator_results(annotator_results, metrics)
    
    # Generate LaTeX table
    print("\n" + "=" * 80)
    print("LaTeX Table:")
    print("=" * 80)
    print(generate_latex_table(win_rates, metrics))
    
    # Save detailed results
    output_path = base_dir / "annotation_analysis_results.json"
    with open(output_path, 'w') as f:
        json.dump({
            'win_rates': win_rates,
            'raw_counts': results,
            'per_annotator': {k: dict(v) for k, v in annotator_results.items()},
        }, f, indent=2, default=dict)
    print(f"\nDetailed results saved to: {output_path}")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    total_pairs = sum(results[ct]['overall']['total'] for ct in results)
    total_unanimous = sum(results[ct]['overall']['unanimous'] for ct in results)
    
    print(f"Total pairs evaluated: {total_pairs}")
    print(f"Overall unanimous agreement: {total_unanimous / total_pairs * 100:.1f}%" if total_pairs > 0 else "N/A")
    
    # Coverage check
    annotations = load_annotations(annotations_dir)
    print(f"\nAnnotation coverage:")
    for batch, annotators in sorted(annotations.items()):
        print(f"  Batch {batch}: {len(annotators)} annotators - {', '.join(sorted(annotators.keys()))}")


if __name__ == "__main__":
    main()
