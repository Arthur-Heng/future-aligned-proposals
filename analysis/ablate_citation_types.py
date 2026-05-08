#!/usr/bin/env python3
"""
Ablation study: Effect of removing papers by citation type on proposal quality.

Citation Types:
- background: provides general framing, theoretical context
- method: provides specific techniques directly used/adapted
- benchmark: introduces datasets, metrics, evaluation protocols

Tests removing each citation type while keeping others.
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Set
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


def load_test_data(path: str) -> List[Dict]:
    """Load test data from JSONL file."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def load_citation_types(path: str) -> Dict[str, Dict]:
    """Load citation type classifications."""
    with open(path, 'r') as f:
        data = json.load(f)
    
    # Build lookup by tree_id
    lookup = {}
    for sample in data['samples']:
        tree_id = sample['tree_id']
        lookup[tree_id] = {
            p['paper_num']: p['citation_types'] 
            for p in sample['papers']
        }
    return lookup


def extract_papers_from_prompt(prompt: str) -> List[Dict]:
    """Extract individual papers from a prompt with their numbers."""
    papers = []
    pattern = r'(### Paper (\d+):.*?)(?=### Paper \d+:|Based on these|$)'
    matches = re.findall(pattern, prompt, re.DOTALL)
    
    for full_text, num in matches:
        paper_text = full_text.strip()
        if paper_text and len(paper_text) > 50:
            papers.append({
                'num': int(num),
                'text': paper_text
            })
    
    return papers


def extract_research_question(prompt: str) -> str:
    """Extract the target research question from prompt."""
    match = re.search(r'Target Research Question:\s*(.+?)(?=\n\n|Given the following)', prompt, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def create_prompt_without_types(
    sample: Dict,
    citation_lookup: Dict[int, List[str]],
    remove_types: Set[str]
) -> Optional[str]:
    """Create a prompt excluding papers of specified citation types."""
    original_prompt = sample.get('prompt', '')
    metadata = sample.get('metadata', {})
    if isinstance(metadata, str):
        metadata = json.loads(metadata.replace("'", '"'))
    rq = extract_research_question(original_prompt)
    
    papers = extract_papers_from_prompt(original_prompt)
    
    # Filter papers: keep those that don't have ALL their types in remove_types
    kept_papers = []
    for paper in papers:
        paper_types = set(citation_lookup.get(paper['num'], ['method']))
        # Keep paper if it has at least one type NOT in remove_types
        if not paper_types.issubset(remove_types):
            kept_papers.append(paper)
    
    if len(kept_papers) == 0:
        return None  # No papers left
    
    # Build new prompt with renumbered papers
    prompt = f"Target Research Question: {rq}\n\n"
    prompt += "Given the following inspiring research papers:\n\n"
    
    for i, paper in enumerate(kept_papers, 1):
        # Renumber papers
        paper_text = re.sub(r'### Paper \d+:', f'### Paper {i}:', paper['text'])
        prompt += f"\n{paper_text}\n"
    
    prompt += f"\n{STEPWISE_COT_INSTRUCTION}"
    
    return prompt, len(kept_papers)


def get_samples_with_type(
    test_data: List[Dict],
    citation_types_lookup: Dict[str, Dict],
    target_type: str
) -> Set[str]:
    """Get tree_ids of samples that have at least one paper of the target type."""
    samples_with_type = set()
    
    for sample in test_data:
        metadata = sample.get('metadata', {})
        if isinstance(metadata, str):
            metadata = json.loads(metadata.replace("'", '"'))
        tree_id = metadata.get('tree_id', '')
        
        if tree_id in citation_types_lookup:
            for paper_num, types in citation_types_lookup[tree_id].items():
                if target_type in types:
                    samples_with_type.add(tree_id)
                    break
    
    return samples_with_type


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
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
) -> str:
    """Generate response using local model (single sample)."""
    import torch
    
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


def generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
) -> List[str]:
    """Generate responses for a batch of prompts."""
    import torch
    
    # Prepare all inputs
    texts = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        texts.append(text)
    
    # Tokenize with padding (left padding for generation)
    tokenizer.padding_side = "left"
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=8192
    ).to(model.device)
    
    input_lengths = [inputs['attention_mask'][i].sum().item() for i in range(len(prompts))]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode each response
    responses = []
    for i, output in enumerate(outputs):
        # Skip the input tokens (account for padding)
        generated = output[inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(generated, skip_special_tokens=True)
        responses.append(response)
    
    return responses


def run_ablation(
    test_data: List[Dict],
    citation_types_lookup: Dict[str, Dict],
    model,
    tokenizer,
    ablation_configs: List[Dict],
    output_dir: str,
    temperature: float = 0.7,
    batch_size: int = 4,
):
    """Run ablation for different citation type removal configs."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get samples with benchmark citations for separate evaluation
    samples_with_benchmark = get_samples_with_type(
        test_data, citation_types_lookup, 'benchmark'
    )
    print(f"\nSamples with benchmark citations: {len(samples_with_benchmark)}")
    
    for config in ablation_configs:
        remove_types = set(config['remove_types'])
        config_name = config['name']
        
        print(f"\n{'='*50}")
        print(f"Running: {config_name}")
        print(f"Removing types: {remove_types}")
        print(f"{'='*50}")
        
        # Prepare all samples first
        prepared_samples = []
        skipped = 0
        
        for sample in test_data:
            metadata = sample.get('metadata', {})
            if isinstance(metadata, str):
                metadata = json.loads(metadata.replace("'", '"'))
            tree_id = metadata.get('tree_id', '')
            
            sample_citation_types = citation_types_lookup.get(tree_id, {})
            
            result = create_prompt_without_types(
                sample, sample_citation_types, remove_types
            )
            
            if result is None:
                skipped += 1
                continue
            
            prompt, n_papers_kept = result
            prepared_samples.append({
                'prompt': prompt,
                'n_papers_kept': n_papers_kept,
                'tree_id': tree_id,
                'metadata': metadata,
                'has_benchmark': tree_id in samples_with_benchmark,
                'ground_truth': sample.get('completion', ''),
            })
        
        print(f"Prepared {len(prepared_samples)} samples, skipped {skipped}")
        
        # Process in batches
        predictions = []
        papers_kept_stats = []
        
        for i in tqdm(range(0, len(prepared_samples), batch_size), desc=config_name):
            batch = prepared_samples[i:i+batch_size]
            prompts = [s['prompt'] for s in batch]
            
            responses = generate_batch(
                model, tokenizer, prompts,
                temperature=temperature
            )
            
            for sample, response in zip(batch, responses):
                papers_kept_stats.append(sample['n_papers_kept'])
                predictions.append({
                    "id": len(predictions),
                    "tree_id": sample['tree_id'],
                    "root_paper_id": sample['metadata'].get('root_paper_id', ''),
                    "root_title": sample['metadata'].get('root_title', ''),
                    "research_question": sample['metadata'].get('research_question', ''),
                    "papers_kept": sample['n_papers_kept'],
                    "has_benchmark": sample['has_benchmark'],
                    "prediction": response,
                    "ground_truth": sample['ground_truth'],
                })
        
        # Save predictions
        output_path = os.path.join(output_dir, f"ablation_{config_name}.json")
        with open(output_path, 'w') as f:
            json.dump({
                "config": {
                    "name": config_name,
                    "remove_types": list(remove_types),
                    "num_samples": len(predictions),
                    "num_skipped": skipped,
                    "avg_papers_kept": sum(papers_kept_stats) / len(papers_kept_stats) if papers_kept_stats else 0,
                    "num_with_benchmark": sum(1 for p in predictions if p['has_benchmark']),
                    "prompt_mode": "stepwise_cot",
                    "ablation_type": "citation_type"
                },
                "predictions": predictions
            }, f, indent=2)
        
        print(f"Saved {len(predictions)} predictions to {output_path}")
        print(f"Skipped: {skipped} (no papers left after filtering)")
        print(f"Avg papers kept: {sum(papers_kept_stats)/len(papers_kept_stats):.1f}" if papers_kept_stats else "N/A")


def main():
    parser = argparse.ArgumentParser(description="Ablation study on citation types")
    parser.add_argument(
        "--test-file", "-t",
        default="data/test_set/test_set_n819.jsonl",
        help="Path to test JSONL file"
    )
    parser.add_argument(
        "--citation-types", "-c",
        default="analyze/citation_types_full.json",
        help="Path to citation types JSON file"
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
        default="analyze/ablation_citation_types",
        help="Output directory for results"
    )
    parser.add_argument(
        "--ablations",
        nargs="+",
        default=["background", "method", "benchmark"],
        help="Ablation configs to run: none, background, method, benchmark"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit number of test samples"
    )
    parser.add_argument(
        "--batch-size", "-bs",
        type=int,
        default=4,
        help="Batch size for inference"
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
    citation_path = args.citation_types if os.path.isabs(args.citation_types) else str(base_dir / args.citation_types)
    adapter_path = args.adapter_path if os.path.isabs(args.adapter_path) else str(base_dir / args.adapter_path)
    output_dir = args.output_dir if os.path.isabs(args.output_dir) else str(base_dir / args.output_dir)
    
    # Load test data
    print(f"Loading test data from {test_path}...")
    test_data = load_test_data(test_path)
    if args.limit:
        test_data = test_data[:args.limit]
    print(f"Loaded {len(test_data)} samples")
    
    # Load citation types
    print(f"Loading citation types from {citation_path}...")
    citation_types_lookup = load_citation_types(citation_path)
    print(f"Loaded citation types for {len(citation_types_lookup)} samples")
    
    # Build ablation configs
    ablation_configs = []
    for abl in args.ablations:
        if abl == "none":
            ablation_configs.append({"name": "full", "remove_types": []})
        elif abl == "background":
            ablation_configs.append({"name": "no_background", "remove_types": ["background"]})
        elif abl == "method":
            ablation_configs.append({"name": "no_method", "remove_types": ["method"]})
        elif abl == "benchmark":
            ablation_configs.append({"name": "no_benchmark", "remove_types": ["benchmark"]})
        elif abl == "background+method":
            ablation_configs.append({"name": "no_background_method", "remove_types": ["background", "method"]})
    
    print(f"\nAblation configs: {[c['name'] for c in ablation_configs]}")
    
    # Load model
    model, tokenizer = load_model(args.base_model, adapter_path)
    
    # Run ablation
    run_ablation(
        test_data=test_data,
        citation_types_lookup=citation_types_lookup,
        model=model,
        tokenizer=tokenizer,
        ablation_configs=ablation_configs,
        output_dir=output_dir,
        temperature=args.temperature,
        batch_size=args.batch_size,
    )
    
    print(f"\nAblation complete! Results saved to {output_dir}/")
    print("\nTo evaluate, run:")
    print(f"  python evaluation/evaluate.py --predictions {output_dir}/ablation_<config>.json")


if __name__ == "__main__":
    main()
