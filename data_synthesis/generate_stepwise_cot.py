#!/usr/bin/env python3
"""
Generate Step-by-Step CoT SFT data from existing SFT data.

Takes existing SFT data (with full proposals) and generates a step-by-step
Chain-of-Thought version where the proposal is built incrementally:

  Reasoning 1 (Problem Identification)
  → Research Question + Hypothesis
  → Reasoning 2 (Method Design)
  → Proposed Method + Novelty Claims
  → Reasoning 3 (Experiment Design)
  → Experiment Details

Usage:
    python data_synthesis/generate_stepwise_cot.py \
        --input data/sft/sft_n2823_with_reasoning.jsonl \
        --structured-papers data/sft/structured_papers.json \
        --output-dir data/sft
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthesize.one_layer_sft import (
    generate_stepwise_reasoning,
    extract_stepwise_sections,
    build_stepwise_completion,
)
from utils.api import call_chat_completion


def load_sft_data(input_file: str) -> List[Dict]:
    """Load existing SFT data from JSONL."""
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    print(f"Loaded {len(data)} examples from {input_file}")
    return data


def load_structured_papers(structured_file: str) -> Dict[str, Dict]:
    """Load structured papers JSON."""
    with open(structured_file, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} structured papers")
    return data


def main():
    parser = argparse.ArgumentParser(description="Generate Step-by-Step CoT SFT data")
    
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL file with existing SFT data"
    )
    parser.add_argument(
        "--structured-papers",
        required=True,
        help="Path to structured_papers.json"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--model",
        default="gpt-5",
        help="Model for reasoning generation (default: gpt-5)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples to process (for testing)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip examples that already have stepwise reasoning cached"
    )
    
    args = parser.parse_args()
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    sft_data = load_sft_data(args.input)
    structured_papers = load_structured_papers(args.structured_papers)
    
    if args.max_samples:
        sft_data = sft_data[:args.max_samples]
        print(f"Limited to {args.max_samples} samples")
    
    # Cache file for incremental progress
    cache_file = output_path / "stepwise_cot_cache.json"
    cache = {}
    if cache_file.exists() and args.skip_existing:
        with open(cache_file, 'r') as f:
            cache = json.load(f)
        print(f"Loaded {len(cache)} cached stepwise reasoning results")
    
    print("\n" + "="*80)
    print("GENERATING STEP-BY-STEP COT DATA")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Examples to process: {len(sft_data)}")
    print("="*80)
    
    results = []
    total_cost = 0.0
    
    for item in tqdm(sft_data, desc="Generating stepwise CoT"):
        metadata = item.get('metadata', {})
        root_paper_id = metadata.get('root_paper_id', '')
        tree_id = metadata.get('tree_id', '')
        
        # Check if we have structured data for this paper
        if root_paper_id not in structured_papers:
            tqdm.write(f"  ⚠ Root paper {root_paper_id[:20]} not in structured papers, skipping")
            continue
        
        root_paper = structured_papers[root_paper_id]
        
        # Check cache
        if tree_id in cache:
            reasoning_text = cache[tree_id]['reasoning']
            cost = 0.0
        else:
            # Get inspiring papers from structured data
            # We need to extract inspiring paper IDs - they're not directly in the JSONL
            # but we can get them from the prompt (or the trees file)
            # For now, use the root paper info to generate stepwise reasoning
            
            # Collect inspiring papers from the prompt
            # The prompt contains "### Paper N:" sections with paper info
            # We'll pass the root paper directly to the reasoning generator
            
            # Get inspiring paper IDs from existing data
            inspiring_papers = _extract_inspiring_from_prompt(item['prompt'], structured_papers)
            
            if not inspiring_papers:
                tqdm.write(f"  ⚠ Could not extract inspiring papers for {root_paper_id[:20]}, skipping")
                continue
            
            reasoning_text, cost = generate_stepwise_reasoning(
                inspiring_papers=inspiring_papers,
                target_paper=root_paper,
                model=args.model
            )
            total_cost += cost
            
            if not reasoning_text:
                tqdm.write(f"  ⚠ Failed to generate stepwise reasoning for {root_paper_id[:20]}")
                continue
            
            # Cache the result
            cache[tree_id] = {
                'reasoning': reasoning_text,
                'cost': cost
            }
            
            # Save cache incrementally
            if len(cache) % 50 == 0:
                with open(cache_file, 'w') as f:
                    json.dump(cache, f, indent=2)
        
        # Extract the 3 reasoning steps
        steps = extract_stepwise_sections(reasoning_text)
        
        # Build the stepwise completion
        completion = build_stepwise_completion(steps, root_paper)
        
        # Update the prompt to instruct step-by-step generation
        prompt = _rewrite_prompt_for_stepwise(item['prompt'])
        
        results.append({
            'prompt': prompt,
            'completion': completion,
            'metadata': {
                'tree_id': tree_id,
                'root_paper_id': root_paper_id,
                'root_title': metadata.get('root_title', ''),
                'research_question': metadata.get('research_question', ''),
                'num_inspiring': metadata.get('num_inspiring', 0),
                'reasoning_cost': cost,
                'format': 'stepwise_cot',
            }
        })
    
    # Save final cache
    with open(cache_file, 'w') as f:
        json.dump(cache, f, indent=2)
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    n = len(results)
    
    # JSONL
    jsonl_file = output_path / f"one_layer_sft_n{n}_stepwise_cot_{timestamp}.jsonl"
    with open(jsonl_file, 'w') as f:
        for item in results:
            f.write(json.dumps(item) + '\n')
    
    # JSON
    json_file = output_path / f"one_layer_sft_n{n}_stepwise_cot_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump({
            'metadata': {
                'num_examples': n,
                'format': 'stepwise_cot',
                'model': args.model,
                'total_cost': total_cost,
                'timestamp': timestamp,
                'source_file': args.input,
            },
            'examples': results
        }, f, indent=2)
    
    print("\n" + "="*80)
    print("STEP-BY-STEP COT GENERATION COMPLETE")
    print("="*80)
    print(f"Examples generated: {n}")
    print(f"Output JSONL: {jsonl_file}")
    print(f"Output JSON: {json_file}")
    print(f"💰 Total cost: ${total_cost:.4f}")
    print("="*80)


STEPWISE_INSTRUCTION = """Based on these inspiring papers, propose a novel research idea step by step.

First, analyze the gaps and identify what ideas to borrow from the inspiring papers. Then, formulate the research question and hypothesis. Next, reason about the method design, and propose the method with its novelty claims. Finally, reason about how to validate the method, and describe the experiment details.

Generate each section in this order:
1. reasoning
2. Research Question + Hypothesis
3. reasoning
4. Proposed Method + Novelty Claims
5. reasoning
6. Experiment Details"""


def _rewrite_prompt_for_stepwise(original_prompt: str) -> str:
    """Rewrite the prompt to instruct step-by-step generation."""
    # Replace the original instruction at the end of the prompt
    old_endings = [
        "Based on these inspiring papers, propose a novel research idea.\nYour proposal should address the target research question.",
        "Based on these inspiring papers, propose a novel research idea.",
    ]
    
    new_prompt = original_prompt
    for old_ending in old_endings:
        if old_ending in new_prompt:
            new_prompt = new_prompt.replace(old_ending, STEPWISE_INSTRUCTION)
            if "Your proposal should address the target research question." not in STEPWISE_INSTRUCTION:
                # Add back the RQ instruction if present in original
                if "target research question" in original_prompt:
                    new_prompt += "\nYour proposal should address the target research question."
            return new_prompt
    
    # Fallback: just append the instruction
    return original_prompt + "\n\n" + STEPWISE_INSTRUCTION


def _extract_inspiring_from_prompt(prompt: str, structured_papers: Dict) -> List[Dict]:
    """
    Extract inspiring paper info from the SFT prompt text by matching titles
    back to structured_papers.
    """
    import re
    
    # Extract paper titles from the prompt (they appear as **Title** (Year))
    title_pattern = r'\*\*(.+?)\*\*\s*\(\d{4}\)'
    titles_in_prompt = re.findall(title_pattern, prompt)
    
    inspiring = []
    for title in titles_in_prompt:
        # Find matching paper in structured data by title
        for paper_id, paper_data in structured_papers.items():
            if paper_data.get('title', '').strip() == title.strip():
                inspiring.append(paper_data)
                break
    
    return inspiring


if __name__ == "__main__":
    main()
