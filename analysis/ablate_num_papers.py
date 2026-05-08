#!/usr/bin/env python3
"""
Ablation study: Effect of number of inspiring papers on proposal quality.

Tests stepwise-CoT model with 1, 2, 3, 4, 5 inspiring papers.
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


STEPWISE_COT_INSTRUCTION = """Based on these inspiring papers, propose a novel research idea step by step.

First, analyze the gaps and identify what ideas to borrow from the inspiring papers. Then, formulate the research question and hypothesis. Next, reason about the method design, and propose the method with its novelty claims. Finally, reason about how to validate the method, and describe the experiment details.

Generate each section in this order:
1. reasoning
2. Research Question + Hypothesis
3. reasoning
4. Proposed Method + Novelty Claims
5. reasoning
6. Experiment Details"""


def load_test_data(path: str, limit: Optional[int] = None) -> List[Dict]:
    """Load test data from JSONL file."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
            if limit and len(data) >= limit:
                break
    return data


def extract_papers_from_prompt(prompt: str) -> List[str]:
    """Extract individual papers from a prompt."""
    papers = []
    
    # Find all paper sections using regex
    pattern = r'(### Paper \d+:.*?)(?=### Paper \d+:|Based on these|$)'
    matches = re.findall(pattern, prompt, re.DOTALL)
    
    for match in matches:
        paper_text = match.strip()
        if paper_text and len(paper_text) > 50:
            papers.append(paper_text)
    
    return papers


def create_prompt_with_n_papers(sample: Dict, n_papers: int) -> str:
    """Create a prompt with exactly n inspiring papers."""
    original_prompt = sample.get('prompt', '')
    metadata = sample.get('metadata', {})
    rq = metadata.get('research_question', '')
    
    # Extract all papers
    papers = extract_papers_from_prompt(original_prompt)
    
    if len(papers) < n_papers:
        return None  # Not enough papers
    
    # Select first n papers
    selected_papers = papers[:n_papers]
    
    # Build new prompt
    prompt = f"Target Research Question: {rq}\n\n"
    prompt += "Given the following inspiring research papers:\n\n"
    
    for i, paper in enumerate(selected_papers, 1):
        # Renumber papers
        paper = re.sub(r'### Paper \d+:', f'### Paper {i}:', paper)
        prompt += f"\n{paper}\n"
    
    prompt += f"\n{STEPWISE_COT_INSTRUCTION}"
    
    return prompt


def load_model(base_model: str, adapter_path: Optional[str] = None):
    """Load local Qwen model with optional LoRA adapter."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"Loading tokenizer from {base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading model from {base_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    if adapter_path and os.path.exists(adapter_path):
        from peft import PeftModel
        print(f"Loading LoRA adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            torch_dtype=torch.bfloat16,
        )
        print("Merging adapter weights...")
        model = model.merge_and_unload()
    
    model.eval()
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 10000,
    temperature: float = 0.7,
) -> str:
    """Generate response using local model."""
    import torch
    
    # Use chat template
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generated = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True)
    return response


def run_ablation(
    test_data: List[Dict],
    model,
    tokenizer,
    n_papers_list: List[int],
    output_dir: str,
    temperature: float = 0.7,
):
    """Run ablation for different numbers of papers."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    for n_papers in n_papers_list:
        print(f"\n{'='*50}")
        print(f"Running with {n_papers} inspiring paper(s)")
        print(f"{'='*50}")
        
        predictions = []
        
        for sample in tqdm(test_data, desc=f"n_papers={n_papers}"):
            prompt = create_prompt_with_n_papers(sample, n_papers)
            
            if prompt is None:
                print(f"  Skipping sample (not enough papers)")
                continue
            
            response = generate_response(
                model, tokenizer, prompt,
                temperature=temperature
            )
            
            metadata = sample.get('metadata', {})
            predictions.append({
                "id": len(predictions),
                "tree_id": metadata.get('tree_id', ''),
                "root_paper_id": metadata.get('root_paper_id', ''),
                "root_title": metadata.get('root_title', ''),
                "research_question": metadata.get('research_question', ''),
                "prediction": response,
                "ground_truth": sample.get('completion', ''),
            })
        
        # Save predictions for this n_papers setting
        output_path = os.path.join(output_dir, f"ablation_n{n_papers}.json")
        with open(output_path, 'w') as f:
            json.dump({
                "config": {
                    "n_papers": n_papers,
                    "num_samples": len(predictions),
                    "prompt_mode": "stepwise_cot",
                    "ablation_type": "num_inspiring_papers"
                },
                "predictions": predictions
            }, f, indent=2)
        
        print(f"Saved {len(predictions)} predictions to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Ablation study on number of inspiring papers")
    parser.add_argument(
        "--test-file", "-t",
        default="data/test_set/test_set_selected_35pct.jsonl",
        help="Path to test JSONL file"
    )
    parser.add_argument(
        "--base-model", "-b",
        default="Qwen/Qwen2.5-14B-Instruct",
        help="Base model name or path"
    )
    parser.add_argument(
        "--adapter-path", "-a",
        default="models/qwen2.5-14b-sft-stepwise-cot-v4",
        help="Path to LoRA adapter"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="analyze/ablation_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--n-papers", "-n",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Number of papers to test (default: 1 2 3 4 5)"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit number of test samples"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    base_dir = Path(__file__).parent.parent
    test_path = args.test_file if os.path.isabs(args.test_file) else str(base_dir / args.test_file)
    adapter_path = args.adapter_path if os.path.isabs(args.adapter_path) else str(base_dir / args.adapter_path)
    output_dir = args.output_dir if os.path.isabs(args.output_dir) else str(base_dir / args.output_dir)
    
    # Load test data
    print(f"Loading test data from {test_path}...")
    test_data = load_test_data(test_path, args.limit)
    print(f"Loaded {len(test_data)} samples")
    
    # Check paper counts
    paper_counts = []
    for sample in test_data:
        papers = extract_papers_from_prompt(sample.get('prompt', ''))
        paper_counts.append(len(papers))
    print(f"Papers per sample: min={min(paper_counts)}, max={max(paper_counts)}, avg={sum(paper_counts)/len(paper_counts):.1f}")
    
    # Load model
    model, tokenizer = load_model(args.base_model, adapter_path)
    
    # Run ablation
    run_ablation(
        test_data=test_data,
        model=model,
        tokenizer=tokenizer,
        n_papers_list=args.n_papers,
        output_dir=output_dir,
        temperature=args.temperature,
    )
    
    print(f"\nAblation complete! Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
