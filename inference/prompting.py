#!/usr/bin/env python3
"""
Baseline: Prompt LLM with research inspiration data and generate novel research ideas.

This script loads prompts from training data, calls an LLM to generate research proposals,
and saves the predictions for later evaluation.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.api import call_chat_completion


def load_structured_papers(json_path: str) -> Dict[str, Dict]:
    """Load structured papers from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def get_research_question(tree_id: str, paper_lookup: Dict[str, Dict]) -> Optional[str]:
    """Get research question for a tree_id from structured papers."""
    # The tree_id has format "tree_<paper_id>", extract the paper_id
    paper_id = tree_id.replace('tree_', '') if tree_id.startswith('tree_') else tree_id
    
    if paper_id in paper_lookup:
        return paper_lookup[paper_id].get('research_question', '')
    return None


SYSTEM_PROMPT = """You are an expert AI research scientist. Given a set of inspiring research papers, your task is to propose a novel research idea that builds upon their key contributions.

Your response should include:
1. A reasoning process explaining how the inspiring papers connect and what gaps/opportunities they reveal
2. A proposed research idea with:
   - A clear title
   - Research question
   - Hypothesis (if applicable)
   - Proposed method
   - Novelty claims

Format your response with clear section headers (## Reasoning Process, ## Proposed Research)."""


def load_training_data(jsonl_path: str) -> List[Dict]:
    """Load training data from JSONL file."""
    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def generate_prediction(
    prompt: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> Tuple[str, float]:
    """Generate a research proposal using LLM."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]

    response, cost = call_chat_completion(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )

    return response, cost


def run_baseline(
    input_file: str,
    output_file: str,
    structured_papers_file: str = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: int = 2048,
    max_samples: int = None
):
    """
    Run the baseline prompting on all samples.
    
    Args:
        input_file: Path to training_data.jsonl
        output_file: Path to save predictions
        structured_papers_file: Path to structured_papers.json (for research questions)
        model: LLM model to use
        temperature: Generation temperature
        max_samples: Maximum number of samples to process (None = all)
    """
    print("\n" + "="*80)
    print("BASELINE (TWO LAYER): LLM Prompting for Research Idea Generation")
    print("="*80)
    print(f"Input: {input_file}")
    print(f"Structured papers: {structured_papers_file or 'Not provided'}")
    print(f"Output: {output_file}")
    print(f"Model: {model}")
    print(f"Temperature: {temperature}")
    print("="*80 + "\n")
    
    # Load data
    data = load_training_data(input_file)
    if max_samples:
        data = data[:max_samples]
    
    # Load structured papers for research questions
    paper_lookup = {}
    if structured_papers_file and os.path.exists(structured_papers_file):
        paper_lookup = load_structured_papers(structured_papers_file)
        print(f"Loaded {len(paper_lookup)} structured papers")
    
    print(f"Loaded {len(data)} samples")
    
    # Generate predictions
    predictions = []
    total_cost = 0.0
    
    for i, sample in enumerate(tqdm(data, desc="Generating (two-layer)")):
        original_prompt = sample['prompt']
        metadata = sample.get('metadata', {})
        tree_id = metadata.get('tree_id', '')
        
        # Get research question from structured papers
        research_question = get_research_question(tree_id, paper_lookup) if paper_lookup else None
        
        # Augment prompt with research question if available
        if research_question:
            prompt = f"""Target Research Question: {research_question}

{original_prompt}

Use the inspiring papers above to propose a research idea that addresses the target research question."""
        else:
            prompt = original_prompt
        
        try:
            prediction, cost = generate_prediction(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            total_cost += cost
            
            predictions.append({
                'id': i,
                'tree_id': tree_id,
                'root_title': metadata.get('root_title', ''),
                'research_question': research_question,
                'prompt': prompt,
                'original_prompt': original_prompt,
                'prediction': prediction,
                'ground_truth': sample.get('completion', ''),
                'cost': cost
            })
            
        except Exception as e:
            tqdm.write(f"  ✗ Error on sample {i}: {e}")
            predictions.append({
                'id': i,
                'tree_id': tree_id,
                'root_title': metadata.get('root_title', ''),
                'research_question': research_question,
                'prompt': prompt,
                'original_prompt': original_prompt,
                'prediction': None,
                'ground_truth': sample.get('completion', ''),
                'error': str(e)
            })
    
    # Save predictions
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            'config': {
                'model': model,
                'temperature': temperature,
                'input_file': input_file,
                'structured_papers_file': structured_papers_file,
                'variant': 'two_layer'
            },
            'predictions': predictions,
            'total_samples': len(predictions),
            'successful': sum(1 for p in predictions if p.get('prediction')),
            'total_cost': total_cost
        }, f, indent=2)
    
    print(f"\n✓ Saved {len(predictions)} predictions to: {output_file}")
    print(f"💰 Total cost: ${total_cost:.4f}")
    print(f"📊 Average cost per sample: ${total_cost/len(predictions):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two-layer baseline: LLM prompting with full inspiration tree")
    parser.add_argument(
        "--input",
        default="./data/test_set/test_set_n819.jsonl",
        help="Path to test JSONL file"
    )
    parser.add_argument(
        "--structured-papers",
        default="./data/test_set/structured_papers.json",
        help="Path to structured_papers.json (for research questions)"
    )
    parser.add_argument(
        "--output",
        default="./predictions/prompting_predictions.json",
        help="Path to save predictions"
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="LLM model to use (default: gpt-4.1-mini)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature (default: 0.7)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate (default: 2048)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to process (default: all)"
    )
    
    args = parser.parse_args()
    
    run_baseline(
        input_file=args.input,
        output_file=args.output,
        structured_papers_file=args.structured_papers,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_samples=args.max_samples
    )

