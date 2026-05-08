#!/usr/bin/env python3
"""
Generate predictions using Qwen on test data.
Output format is compatible with evaluation/evaluate.py

Supports:
- Base Qwen via API (default)
- Fine-tuned Qwen with LoRA adapter (--adapter-path)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.api import call_chat_completion


SYSTEM_PROMPT = """You are an expert AI research scientist. Given inspiring research papers and a target research question, propose a novel research idea that addresses the question.

Your response should include:
- A proposed research idea with title, research question, hypothesis, proposed method, novelty claims, and experiment details

Format your response starting with "## Proposed Research"."""

SYSTEM_PROMPT_RQ_ONLY = """You are an expert AI research scientist. Given a research question, propose a novel research idea that addresses the question.

Your response should include:
- A proposed research idea with title, hypothesis, proposed method, novelty claims, and experiment details

Format your response starting with "## Proposed Research"."""

SYSTEM_PROMPT_PAPERS_ONLY = """You are an expert AI research scientist. Given a set of inspiring research papers, identify gaps and opportunities, then propose a novel research idea that builds upon their contributions.

Your response should include:
- A proposed research idea with title, research question, hypothesis, proposed method, novelty claims, and experiment details

Format your response starting with "## Proposed Research"."""

SYSTEM_PROMPT_COT = """You are an expert AI research scientist. Given inspiring research papers and a target research question, first reason about gaps and opportunities, then propose a novel research idea.

Your response should include:
1. A reasoning section analyzing gaps, borrowing inspiration, and synthesizing ideas
2. A proposed research idea with title, research question, hypothesis, proposed method, novelty claims, and experiment details

Format the reasoning with "### Gap Analysis", "### Inspiration Borrowing", "### Synthesis" sections, then the proposal starting with "## Proposed Research"."""

SYSTEM_PROMPT_STEPWISE_COT = """You are an expert AI research scientist. Given inspiring research papers and a target research question, develop a novel research idea step by step.

Your response should follow this exact structure:
1. Problem Identification reasoning (analyze gaps and inspiration)
2. Research Question + Hypothesis
3. Method Design reasoning (how to approach the problem)
4. Proposed Method + Novelty Claims
5. Experiment Design reasoning (how to validate)
6. Experiment Details

Use "### Step 1: Problem Identification", "### Step 2: Method Design Reasoning", "### Step 3: Experiment Design Reasoning" for reasoning sections, and "## Proposed Research" before the proposal sections."""


COT_INSTRUCTION = """Based on these inspiring papers, first analyze the research landscape, then propose a novel research idea.

Start by identifying gaps in the existing work, what ideas to borrow, and how to synthesize them. Then propose your research idea."""

STEPWISE_COT_INSTRUCTION = """Based on these inspiring papers, propose a novel research idea step by step.

First, analyze the gaps and identify what ideas to borrow from the inspiring papers. Then, formulate the research question and hypothesis. Next, reason about the method design, and propose the method with its novelty claims. Finally, reason about how to validate the method, and describe the experiment details.

Generate each section in this order:
1. reasoning
2. Research Question + Hypothesis
3. reasoning
4. Proposed Method + Novelty Claims
5. reasoning
6. Experiment Details"""


def rewrite_prompt_for_mode(original_prompt: str, mode: str) -> str:
    """Rewrite the user prompt to match the SFT training format.
    
    - direct / cot: keep original prompt (SFT with_reasoning uses the same prompt as direct)
    - stepwise_cot: rewrite to stepwise instruction (matches stepwise SFT training prompt)
    """
    if mode != "stepwise_cot":
        return original_prompt
    
    old_endings = [
        "Based on these inspiring papers, propose a novel research idea.\nYour proposal should address the target research question.",
        "Based on these inspiring papers, propose a novel research idea.",
    ]
    
    new_prompt = original_prompt
    for old_ending in old_endings:
        if old_ending in new_prompt:
            has_rq = "target research question" in original_prompt.split("Given the following")[0] if "Given the following" in original_prompt else False
            new_prompt = new_prompt.replace(old_ending, STEPWISE_COT_INSTRUCTION)
            if has_rq:
                new_prompt += "\nYour proposal should address the target research question."
            return new_prompt
    
    return original_prompt + "\n\n" + STEPWISE_COT_INSTRUCTION


def load_test_data(path: str, num_samples: Optional[int] = None) -> List[Dict]:
    """Load test data from JSONL file."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
            if num_samples and len(data) >= num_samples:
                break
    return data


def create_rq_only_prompt(sample: Dict) -> Optional[str]:
    """Create a prompt with only the research question (no inspiring papers)."""
    metadata = sample.get('metadata', {})
    rq = metadata.get('research_question', '')
    
    if not rq:
        # Try to extract from the original prompt if not in metadata
        original_prompt = sample.get('prompt', '')
        if 'Research Question:' in original_prompt:
            # Extract RQ from prompt
            start = original_prompt.find('Research Question:')
            if start != -1:
                end = original_prompt.find('\n\n', start)
                if end == -1:
                    end = original_prompt.find('##', start)
                if end != -1:
                    rq = original_prompt[start + len('Research Question:'):end].strip()
    
    if not rq:
        return None
    
    return f"""## Research Question

{rq}

Please propose a novel research idea to address this question."""


def create_papers_only_prompt(sample: Dict) -> Optional[str]:
    """Create a prompt with only the inspiring papers (no research question)."""
    original_prompt = sample.get('prompt', '')
    
    if not original_prompt:
        return None
    
    # Remove the "Target Research Question:" section if present
    # The prompt typically has format:
    # "Target Research Question: ...\n\nGiven the following inspiring research papers:\n\n..."
    
    # Find where the inspiring papers section starts
    papers_markers = [
        "Given the following inspiring research papers:",
        "## Inspiring Research Papers",
        "### Paper 1:",
    ]
    
    papers_start = -1
    for marker in papers_markers:
        idx = original_prompt.find(marker)
        if idx != -1:
            papers_start = idx
            break
    
    if papers_start == -1:
        # No inspiring papers section found
        return None
    
    # Extract the inspiring papers section
    papers_section = original_prompt[papers_start:].strip()
    
    # Remove any trailing instruction like "Based on these inspiring papers..."
    # We'll add our own instruction
    instruction_markers = [
        "Based on these inspiring papers",
        "Please propose",
        "Your task is to",
    ]
    
    for marker in instruction_markers:
        idx = papers_section.find(marker)
        if idx != -1:
            papers_section = papers_section[:idx].strip()
    
    if not papers_section:
        return None
    
    return f"""{papers_section}

Based on these inspiring papers, identify research gaps and opportunities, then propose a novel research idea."""


def load_local_model(base_model: str, adapter_path: Optional[str] = None):
    """
    Load local Qwen model, optionally with LoRA adapter.
    
    Returns:
        Tuple of (model, tokenizer)
    """
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
        # Merge adapter weights into base model for faster inference
        # This fuses LoRA weights into the base weights, eliminating the
        # per-forward-pass LoRA computation overhead (~3x speedup)
        print("Merging adapter weights into base model...")
        model = model.merge_and_unload()
    
    model.eval()
    return model, tokenizer


def generate_with_local_model(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 1500,
    temperature: float = 0.7,
    repetition_penalty: float = 1.08,
    system_prompt: str = None,
    use_chat_template: bool = False
) -> str:
    """Generate response using local model (single sample).
    
    Args:
        use_chat_template: If True, use chat template with system prompt (for base models).
                          If False, use plain text format matching SFT training (for fine-tuned models).
        repetition_penalty: Penalty for repeating tokens (1.0 = no penalty, >1.0 = discourage repetition).
                           Default 1.08 helps prevent text degeneration in long generations.
    """
    import torch
    
    if use_chat_template:
        # Use chat template for base models
        if system_prompt is None:
            system_prompt = SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        # Use plain text format matching SFT training
        text = f"{prompt}\n\n## Response:\n"
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the generated part
    generated = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True)
    
    # Clean up response
    response = _clean_generation_output(response)
    
    return response.strip()


def _clean_generation_output(response: str) -> str:
    """
    Clean up generation output that may contain repeated prompt content or artifacts.
    
    Some fine-tuned models may:
    - Repeat parts of the prompt
    - Include "## Response:" in output
    - Generate timestamps or other training artifacts
    """
    import re
    
    # If response contains "## Response:", take everything after the LAST occurrence
    if "## Response:" in response:
        parts = response.split("## Response:")
        response = parts[-1].strip()
    
    # If response contains "## Proposed Research", extract from there
    # (This is the actual start of the proposal in our format)
    if "## Proposed Research" in response:
        idx = response.find("## Proposed Research")
        # Keep a bit before if it's reasoning (### Step headers)
        # Check if there's step headers before it
        step_markers = ["### Step 1:", "### Step 2:", "### Step 3:"]
        earliest_step = len(response)
        for marker in step_markers:
            pos = response.find(marker)
            if pos != -1 and pos < earliest_step:
                earliest_step = pos
        
        if earliest_step < idx:
            response = response[earliest_step:]
        else:
            response = response[idx:]
    
    # Remove timestamps at the end (pattern: 2024-10-12 14:28:43 UTC)
    response = re.sub(r'\n\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC.*$', '', response, flags=re.DOTALL)
    
    # Remove repeated instruction patterns if they appear
    instruction_markers = [
        "Based on these inspiring papers, propose a novel research idea.",
        "Based on these inspiring papers, propose a novel research idea step by step.",
        "Your proposal should address the target research question.",
    ]
    for marker in instruction_markers:
        if marker in response:
            idx = response.find(marker)
            # Only remove if it's near the start (within first 500 chars)
            if idx < 500:
                end_idx = response.find("\n\n", idx)
                if end_idx != -1:
                    response = response[end_idx:].strip()
    
    return response.strip()


def generate_batch_with_local_model(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 1500,
    temperature: float = 0.7,
    repetition_penalty: float = 1.08,
    system_prompt: str = None,
    use_chat_template: bool = False
) -> List[str]:
    """Generate responses using local model (batched).
    
    Args:
        use_chat_template: If True, use chat template with system prompt (for base models).
                          If False, use plain text format matching SFT training (for fine-tuned models).
        repetition_penalty: Penalty for repeating tokens (1.0 = no penalty, >1.0 = discourage repetition).
                           Default 1.08 helps prevent text degeneration in long generations.
    """
    import torch
    
    # Prepare all texts
    all_texts = []
    for prompt in prompts:
        if use_chat_template:
            if system_prompt is None:
                system_prompt = SYSTEM_PROMPT
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Use plain text format matching SFT training
            text = f"{prompt}\n\n## Response:\n"
        all_texts.append(text)
    
    # Tokenize batch with padding
    inputs = tokenizer(
        all_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096  # Limit input length for memory
    ).to(model.device)
    
    input_lengths = [inputs['attention_mask'][i].sum().item() for i in range(len(prompts))]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode each response
    responses = []
    for i, output in enumerate(outputs):
        # Get the generated part (after input)
        generated = output[input_lengths[i]:]
        response = tokenizer.decode(generated, skip_special_tokens=True)
        response = _clean_generation_output(response)
        responses.append(response.strip())
    
    return responses


def generate_predictions_api(
    test_data: List[Dict],
    model: str = "qwen2.5-7b-instruct",
    temperature: float = 0.7,
    max_tokens: int = 1500,
    rq_only: bool = False,
    papers_only: bool = False,
    cot: bool = False,
    stepwise_cot: bool = False
) -> List[Dict]:
    """Generate predictions using API."""
    predictions = []
    total_cost = 0.0
    skipped = 0
    
    # Select system prompt based on mode
    if rq_only:
        system_prompt = SYSTEM_PROMPT_RQ_ONLY
        mode_name = "rq_only"
    elif papers_only:
        system_prompt = SYSTEM_PROMPT_PAPERS_ONLY
        mode_name = "papers_only"
    elif stepwise_cot:
        system_prompt = SYSTEM_PROMPT_STEPWISE_COT
        mode_name = "stepwise_cot"
    elif cot:
        system_prompt = SYSTEM_PROMPT_COT
        mode_name = "cot"
    else:
        system_prompt = SYSTEM_PROMPT
        mode_name = "full"
    
    for i, sample in enumerate(tqdm(test_data, desc=f"Generating with {model}")):
        metadata = sample.get('metadata', {})
        
        # Get prompt based on mode
        if rq_only:
            prompt = create_rq_only_prompt(sample)
            if prompt is None:
                skipped += 1
                continue
        elif papers_only:
            prompt = create_papers_only_prompt(sample)
            if prompt is None:
                skipped += 1
                continue
        else:
            prompt = rewrite_prompt_for_mode(sample['prompt'], mode_name)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response, cost = call_chat_completion(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            total_cost += cost
            
            predictions.append({
                'id': i,
                'prediction': response,
                'root_title': metadata.get('root_title', ''),
                'root_paper_id': metadata.get('root_paper_id', ''),
                'tree_id': metadata.get('tree_id', ''),
                'research_question': metadata.get('research_question', ''),
                'ground_truth': sample.get('completion', ''),
                'cost': cost,
                'prompt_mode': mode_name
            })
        except Exception as e:
            print(f"Error on sample {i}: {e}")
            predictions.append({
                'id': i,
                'prediction': f"ERROR: {e}",
                'root_title': metadata.get('root_title', ''),
                'root_paper_id': metadata.get('root_paper_id', ''),
                'error': str(e)
            })
    
    if skipped > 0:
        skip_reason = "research question" if rq_only else "inspiring papers"
        print(f"\nSkipped {skipped} samples without {skip_reason}")
    print(f"\nTotal cost for {model}: ${total_cost:.4f}")
    return predictions


def generate_predictions_local(
    test_data: List[Dict],
    model,
    tokenizer,
    model_name: str,
    temperature: float = 0.7,
    repetition_penalty: float = 1.08,
    max_tokens: int = 1500,
    batch_size: int = 1,
    rq_only: bool = False,
    papers_only: bool = False,
    cot: bool = False,
    stepwise_cot: bool = False,
    use_chat_template: bool = False
) -> List[Dict]:
    """Generate predictions using local model with optional batching.
    
    Args:
        use_chat_template: If True, use chat template with system prompt (for base models).
                          If False, use plain text format matching SFT training (for fine-tuned models).
        repetition_penalty: Penalty for repeating tokens (1.0 = no penalty, >1.0 = discourage repetition).
                           Default 1.08 helps prevent text degeneration in long generations.
    """
    predictions = []
    skipped = 0
    
    # Select system prompt based on mode
    if rq_only:
        system_prompt = SYSTEM_PROMPT_RQ_ONLY
        mode_name = "rq_only"
    elif papers_only:
        system_prompt = SYSTEM_PROMPT_PAPERS_ONLY
        mode_name = "papers_only"
    elif stepwise_cot:
        system_prompt = SYSTEM_PROMPT_STEPWISE_COT
        mode_name = "stepwise_cot"
    elif cot:
        system_prompt = SYSTEM_PROMPT_COT
        mode_name = "cot"
    else:
        system_prompt = SYSTEM_PROMPT
        mode_name = "full"
    
    if batch_size > 1:
        # Batched generation
        num_batches = (len(test_data) + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(num_batches), desc=f"Generating batches ({batch_size}/batch)"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(test_data))
            batch_samples = test_data[start_idx:end_idx]
            
            # Get prompts based on mode
            prompts = []
            valid_samples = []
            for s in batch_samples:
                if rq_only:
                    prompt = create_rq_only_prompt(s)
                    if prompt is None:
                        skipped += 1
                        continue
                elif papers_only:
                    prompt = create_papers_only_prompt(s)
                    if prompt is None:
                        skipped += 1
                        continue
                else:
                    prompt = rewrite_prompt_for_mode(s['prompt'], mode_name)
                prompts.append(prompt)
                valid_samples.append(s)
            
            if not prompts:
                continue
            
            try:
                responses = generate_batch_with_local_model(
                    model,
                    tokenizer,
                    prompts,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    system_prompt=system_prompt,
                    use_chat_template=use_chat_template
                )
                
                for i, (sample, response) in enumerate(zip(valid_samples, responses)):
                    metadata = sample.get('metadata', {})
                    predictions.append({
                        'id': start_idx + i,
                        'prediction': response,
                        'root_title': metadata.get('root_title', ''),
                        'root_paper_id': metadata.get('root_paper_id', ''),
                        'tree_id': metadata.get('tree_id', ''),
                        'research_question': metadata.get('research_question', ''),
                        'ground_truth': sample.get('completion', ''),
                        'cost': 0.0,
                        'prompt_mode': mode_name
                    })
            except Exception as e:
                print(f"Error on batch {batch_idx}: {e}")
                # Fall back to single generation for this batch
                for i, sample in enumerate(valid_samples):
                    metadata = sample.get('metadata', {})
                    prompt = prompts[i] if i < len(prompts) else sample['prompt']
                    try:
                        response = generate_with_local_model(
                            model, tokenizer, prompt,
                            max_new_tokens=max_tokens, temperature=temperature,
                            repetition_penalty=repetition_penalty,
                            system_prompt=system_prompt,
                            use_chat_template=use_chat_template
                        )
                        predictions.append({
                            'id': start_idx + i,
                            'prediction': response,
                            'root_title': metadata.get('root_title', ''),
                            'root_paper_id': metadata.get('root_paper_id', ''),
                            'tree_id': metadata.get('tree_id', ''),
                            'research_question': metadata.get('research_question', ''),
                            'ground_truth': sample.get('completion', ''),
                            'cost': 0.0,
                            'prompt_mode': mode_name
                        })
                    except Exception as e2:
                        predictions.append({
                            'id': start_idx + i,
                            'prediction': f"ERROR: {e2}",
                            'root_title': metadata.get('root_title', ''),
                            'root_paper_id': metadata.get('root_paper_id', ''),
                            'error': str(e2)
                        })
    else:
        # Single sample generation
        for i, sample in enumerate(tqdm(test_data, desc=f"Generating with {model_name}")):
            metadata = sample.get('metadata', {})
            
            # Get prompt based on mode
            if rq_only:
                prompt = create_rq_only_prompt(sample)
                if prompt is None:
                    skipped += 1
                    continue
            elif papers_only:
                prompt = create_papers_only_prompt(sample)
                if prompt is None:
                    skipped += 1
                    continue
            else:
                prompt = rewrite_prompt_for_mode(sample['prompt'], mode_name)

            try:
                response = generate_with_local_model(
                    model,
                    tokenizer,
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    system_prompt=system_prompt,
                    use_chat_template=use_chat_template
                )
                
                predictions.append({
                    'id': i,
                    'prediction': response,
                    'root_title': metadata.get('root_title', ''),
                    'root_paper_id': metadata.get('root_paper_id', ''),
                    'tree_id': metadata.get('tree_id', ''),
                    'research_question': metadata.get('research_question', ''),
                    'ground_truth': sample.get('completion', ''),
                    'cost': 0.0,
                    'prompt_mode': mode_name
                })
            except Exception as e:
                print(f"Error on sample {i}: {e}")
                predictions.append({
                    'id': i,
                    'prediction': f"ERROR: {e}",
                    'root_title': metadata.get('root_title', ''),
                    'root_paper_id': metadata.get('root_paper_id', ''),
                    'error': str(e)
                })
    
    if skipped > 0:
        skip_reason = "research question" if rq_only else "inspiring papers"
        print(f"\nSkipped {skipped} samples without {skip_reason}")
    
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Generate predictions with Qwen for evaluation")
    parser.add_argument("--test-file", default="data/test_set/test_set_n819.jsonl",
                        help="Path to test JSONL file")
    parser.add_argument("--output", default="predictions/qwen_predictions.json",
                        help="Output file path (compatible with evaluate.py)")
    parser.add_argument("--model", default="qwen2.5-7b-instruct",
                        help="Model name (API) or HuggingFace model ID (local)")
    parser.add_argument("--adapter-path", default=None,
                        help="Path to LoRA adapter directory (for fine-tuned model)")
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint folder name within adapter-path (e.g., 'checkpoint-200')")
    parser.add_argument("--local", action="store_true",
                        help="Force local model loading (for untuned Qwen with batching)")
    parser.add_argument("--no-chat-template", action="store_true",
                        help="Disable chat template (use plain text format). "
                             "Only use this for models trained without --use-chat-template.")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Number of samples to process (default: all)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for local model inference (default: 1)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition-penalty", type=float, default=1.0,
                        help="Repetition penalty for local model generation (default: 1.0). "
                             "Values >1.0 discourage repetition, helping prevent text degeneration.")
    parser.add_argument("--max-tokens", type=int, default=1500)
    
    # Mutually exclusive prompt mode options
    prompt_mode = parser.add_mutually_exclusive_group()
    prompt_mode.add_argument("--rq-only", action="store_true",
                        help="Prompt with only the research question (no inspiring papers)")
    prompt_mode.add_argument("--papers-only", action="store_true",
                        help="Prompt with only inspiring papers (no research question)")
    prompt_mode.add_argument("--cot", action="store_true",
                        help="Full CoT: reasoning (gap analysis + inspiration + synthesis) → full proposal")
    prompt_mode.add_argument("--stepwise-cot", action="store_true",
                        help="Stepwise CoT: interleaved reasoning and proposal sections")
    
    args = parser.parse_args()
    
    # Auto-increase max_tokens for CoT modes if not explicitly set
    if (args.cot or args.stepwise_cot) and args.max_tokens == 1500:
        args.max_tokens = 2500
        print(f"Auto-increased max_tokens to {args.max_tokens} for CoT mode")
    
    # Determine if using local model (has adapter OR --local flag)
    use_local = args.adapter_path is not None or args.local
    
    # Resolve adapter path with checkpoint if specified
    adapter_path = args.adapter_path
    if adapter_path and args.checkpoint:
        adapter_path = os.path.join(adapter_path, args.checkpoint)
        if not os.path.exists(adapter_path):
            print(f"ERROR: Checkpoint path does not exist: {adapter_path}")
            sys.exit(1)
    
    # Determine whether to use chat template
    # Default: always use chat template (recommended for alignment)
    # Can be disabled with --no-chat-template for legacy models trained without chat template
    use_chat_template = not args.no_chat_template
    
    # Determine prompt mode
    if args.rq_only:
        prompt_mode_str = "RQ-only (no inspiring papers)"
    elif args.papers_only:
        prompt_mode_str = "Papers-only (no research question)"
    elif args.cot:
        prompt_mode_str = "Full CoT (reasoning → proposal)"
    elif args.stepwise_cot:
        prompt_mode_str = "Stepwise CoT (interleaved reasoning + proposal)"
    else:
        prompt_mode_str = "Direct (RQ + inspiring papers → proposal)"
    
    print("=" * 80)
    print(f"GENERATING PREDICTIONS")
    print("=" * 80)
    print(f"Model: {args.model}")
    if adapter_path:
        print(f"Adapter: {adapter_path}")
    if use_local:
        mode_str = "Local (with adapter)" if adapter_path else "Local (base model)"
    else:
        mode_str = "API"
    print(f"Mode: {mode_str}")
    print(f"Prompt mode: {prompt_mode_str}")
    if use_local:
        template_str = "Chat template (with system prompt)" if use_chat_template else "Plain text (SFT format)"
        print(f"Input format: {template_str}")
        print(f"Batch size: {args.batch_size}")
    print(f"Test file: {args.test_file}")
    print(f"Output: {args.output}")
    print(f"Samples: {args.num_samples or 'all'}")
    print("=" * 80)
    
    # Load test data
    test_data = load_test_data(args.test_file, args.num_samples)
    print(f"Loaded {len(test_data)} test samples")
    
    # Determine prompt mode for metadata
    if args.rq_only:
        prompt_mode = "rq_only"
    elif args.papers_only:
        prompt_mode = "papers_only"
    elif args.cot:
        prompt_mode = "cot"
    elif args.stepwise_cot:
        prompt_mode = "stepwise_cot"
    else:
        prompt_mode = "full"
    
    # Generate predictions
    if use_local:
        model, tokenizer = load_local_model(args.model, adapter_path)
        model_name = f"{args.model}" + (f" + {adapter_path}" if adapter_path else " (base)")
        predictions = generate_predictions_local(
            test_data,
            model,
            tokenizer,
            model_name=model_name,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            max_tokens=args.max_tokens,
            batch_size=args.batch_size,
            rq_only=args.rq_only,
            papers_only=args.papers_only,
            use_chat_template=use_chat_template,
            cot=args.cot,
            stepwise_cot=args.stepwise_cot
        )
        model_info = {
            'base_model': args.model,
            'adapter_path': adapter_path,
            'checkpoint': args.checkpoint,
            'batch_size': args.batch_size,
            'mode': 'local',
            'prompt_mode': prompt_mode,
            'use_chat_template': use_chat_template
        }
    else:
        predictions = generate_predictions_api(
            test_data,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            rq_only=args.rq_only,
            papers_only=args.papers_only,
            cot=args.cot,
            stepwise_cot=args.stepwise_cot
        )
        model_info = {
            'model': args.model,
            'mode': 'api',
            'prompt_mode': prompt_mode
        }
    
    # Save in format expected by evaluate.py
    output_data = {
        'config': {
            **model_info,
            'test_file': args.test_file,
            'num_samples': len(predictions),
            'temperature': args.temperature,
            'repetition_penalty': args.repetition_penalty if use_local else None
        },
        'predictions': predictions
    }
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Saved {len(predictions)} predictions to {args.output}")
    print(f"\nTo evaluate, run:")
    print(f"  python evaluation/evaluate.py --predictions {args.output}")


if __name__ == "__main__":
    main()
