#!/usr/bin/env python3
"""
One-Layer SFT Data Synthesis Pipeline.

This simplified pipeline:
1. Only uses direct citations (Level 1 papers) - no deeper tree traversal
2. Uses a strong LLM to generate the reasoning process that bridges from 
   inspiring papers to the final paper
3. Outputs SFT-ready training data with:
   - Prompt: Layer-one papers + (optionally) research question
   - Completion: LLM-generated reasoning + the real paper
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.semantic_api import SemanticScholarAPI
from utils.api import call_chat_completion
from synthesize.build import LLMResearchTreeBuilder, load_api_key


# ============================================================================
# Reasoning Generation
# ============================================================================

REASONING_GENERATION_PROMPT = """You are an expert research scientist. Given a set of inspiring papers and a target paper, generate a reasoning process that explains how these papers could lead to new research.

## Inspiring Papers (Direct Citations):
{inspiring_papers}

## Target Paper (Research Outcome):
Title: {target_title}
Research Question: {target_research_question}
Hypothesis: {target_hypothesis}
Proposed Method: {target_proposed_method}
Novelty Claims: {target_novelty_claims}

## Task:
Generate a structured reasoning process with exactly 3 sections. Write as if you are a researcher developing ideas BEFORE creating the target paper (do NOT leak the target paper's specific solutions).

## Required Format (use these exact headers):

### Gap Analysis
Identify 2-4 specific gaps or limitations in the inspiring papers:
- Gap 1: [specific limitation]
- Gap 2: [what's missing]
...

### Inspiration Borrowing  
What techniques/ideas to borrow from which papers:
- From [Paper Title]: [specific technique or idea to adapt]
- From [Paper Title]: [framework or method to build on]
...

### Synthesis
How to combine the borrowed ideas to address the gaps (100-150 words):
- Integration approach
- Key modifications needed
- Why this combination addresses the gaps

IMPORTANT:
- Total: 300-500 words
- Be specific about paper names when borrowing ideas
- Do NOT mention "the target paper" or leak its solutions"""


@dataclass
class OneLayerTreeData:
    """Core data structure: just paper identifiers."""
    tree_id: str
    root_paper_id: str
    inspiring_paper_ids: List[str]  # List of paper IDs that inspire the root
    selections_per_level: int

@dataclass
class OneLayerSFTExample:
    """A training example for one-layer SFT."""
    tree_id: str
    prompt: str
    completion: str
    root_paper_id: str
    root_paper_title: str
    research_question: str
    num_inspiring_papers: int
    reasoning_cost: float
    # Extracted reasoning sections for ablation studies
    gap_analysis: str = ""
    inspiration_borrowing: str = ""
    synthesis: str = ""


def format_paper_for_prompt(paper: Dict, include_full_structure: bool = True) -> str:
    """Format a paper for the prompt."""
    parts = []
    
    title = paper.get('title', 'Unknown Title')
    year = paper.get('year')
    year_str = str(year) if year else 'Unknown Year'
    parts.append(f"**{title}** ({year_str})")
    
    if include_full_structure:
        if paper.get('research_question'):
            parts.append(f"Research Question: {paper['research_question']}")
        if paper.get('hypothesis'):
            parts.append(f"Hypothesis: {paper['hypothesis']}")
        if paper.get('proposed_method'):
            parts.append(f"Proposed Method: {paper['proposed_method']}")
        if paper.get('novelty_claims'):
            parts.append(f"Novelty Claims: {paper['novelty_claims']}")
    else:
        # Just title and abstract for papers without full structure
        if paper.get('abstract'):
            parts.append(f"Abstract: {paper['abstract']}")
    
    return "\n".join(parts)


def format_target_paper(paper: Dict) -> str:
    """Format the target (root) paper for the completion."""
    parts = []
    parts.append("## Proposed Research\n")
    
    title = paper.get('title', 'Unknown Title')
    year = paper.get('year')
    year_str = str(year) if year else 'Unknown Year'
    parts.append(f"**{title}** ({year_str})")
    
    if paper.get('research_question'):
        parts.append(f"Research Question: {paper['research_question']}")
    if paper.get('hypothesis'):
        parts.append(f"Hypothesis: {paper['hypothesis']}")
    if paper.get('proposed_method'):
        parts.append(f"Proposed Method: {paper['proposed_method']}")
    if paper.get('novelty_claims'):
        parts.append(f"Novelty Claims: {paper['novelty_claims']}")
    if paper.get('experiment_details'):
        parts.append(f"Experiment Details: {paper['experiment_details']}")
    
    return "\n".join(parts)


def extract_proposal_from_completion(completion: str) -> str:
    """Extract the proposal section from a completion (strips reasoning if present)."""
    # The proposal section starts with "## Proposed Research"
    marker = "## Proposed Research"
    if marker in completion:
        idx = completion.find(marker)
        return completion[idx:]
    # If no marker found, return the whole completion
    return completion


def strip_reasoning_from_completion(completion: str) -> str:
    """Strip reasoning and return completion with smooth transition + proposal."""
    proposal = extract_proposal_from_completion(completion)
    return "Based on these inspiring papers, here's a novel research proposal:\n\n" + proposal


def generate_reasoning(
    inspiring_papers: List[Dict],
    target_paper: Dict,
    model: str = "gpt-4o"
) -> Tuple[str, float]:
    """
    Use a strong LLM to generate reasoning that bridges inspiring papers to target paper.
    
    Args:
        inspiring_papers: List of inspiring paper dictionaries
        target_paper: The target (root) paper dictionary
        model: Model to use for generation
    
    Returns:
        Tuple of (reasoning_text, cost)
    """
    # Format inspiring papers
    inspiring_text = ""
    for i, paper in enumerate(inspiring_papers, 1):
        inspiring_text += f"\n### Paper {i}:\n"
        inspiring_text += format_paper_for_prompt(paper)
        inspiring_text += "\n"
    
    prompt = REASONING_GENERATION_PROMPT.format(
        inspiring_papers=inspiring_text,
        target_title=target_paper.get('title', 'Unknown'),
        target_research_question=target_paper.get('research_question', 'Not specified'),
        target_hypothesis=target_paper.get('hypothesis', 'Not specified'),
        target_proposed_method=target_paper.get('proposed_method', 'Not specified'),
        target_novelty_claims=target_paper.get('novelty_claims', 'Not specified')
    )
    
    messages = [{"role": "user", "content": prompt}]
    
    try:
        response, cost = call_chat_completion(
            messages=messages,
            model=model,
            temperature=0.7  # Some creativity for reasoning
        )
        return response, cost
    except Exception as e:
        print(f"  ⚠ Reasoning generation failed: {e}")
        return "", 0.0


# ============================================================================
# Step-by-Step CoT Reasoning Generation
# ============================================================================

STEPWISE_COT_PROMPT = """You are an expert research scientist. Given a set of inspiring papers and a target paper, generate a step-by-step reasoning process that shows how a researcher would develop this research idea incrementally.

## Inspiring Papers (Direct Citations):
{inspiring_papers}

## Target Paper (the final research output):
Title: {target_title}
Research Question: {target_research_question}
Hypothesis: {target_hypothesis}
Proposed Method: {target_proposed_method}
Novelty Claims: {target_novelty_claims}
Experiment Details: {target_experiment_details}

## Task:
Generate intermediate reasoning that bridges BETWEEN the proposal sections. You will produce exactly 3 short reasoning blocks. The final output will be interleaved as:

[Reasoning 1] → Research Question + Hypothesis → [Reasoning 2] → Proposed Method + Novelty Claims → [Reasoning 3] → Experiment Details

Write each reasoning block as described below.

## Required Format (use these EXACT headers):

### Step 1: Problem Identification
Analyze the inspiring papers to identify gaps and formulate the research direction (150-250 words):
- What gaps or limitations exist in the inspiring papers
- What techniques/ideas could be borrowed
- How combining them suggests a specific research question

### Step 2: Method Design Reasoning
Given the research question and hypothesis, reason about how to design the method (80-120 words):
- What approach would address the research question
- Which techniques from the inspiring papers to adapt
- What makes this combination novel

### Step 3: Experiment Design Reasoning
Given the proposed method, reason about how to validate it (60-100 words):
- What datasets/benchmarks are appropriate
- What baselines to compare against
- What metrics would demonstrate the method's effectiveness

IMPORTANT:
- Do NOT copy the target paper's sections verbatim into the reasoning
- Write as forward-looking reasoning (as if you haven't seen the answer yet)
- Do NOT mention "the target paper" - write as your own thought process
- Each step should naturally lead to the next section of the proposal"""


def generate_stepwise_reasoning(
    inspiring_papers: List[Dict],
    target_paper: Dict,
    model: str = "gpt-4o"
) -> Tuple[str, float]:
    """
    Generate step-by-step reasoning that bridges between proposal sections.
    
    Returns:
        Tuple of (reasoning_text with 3 steps, cost)
    """
    inspiring_text = ""
    for i, paper in enumerate(inspiring_papers, 1):
        inspiring_text += f"\n### Paper {i}:\n"
        inspiring_text += format_paper_for_prompt(paper)
        inspiring_text += "\n"
    
    prompt = STEPWISE_COT_PROMPT.format(
        inspiring_papers=inspiring_text,
        target_title=target_paper.get('title', 'Unknown'),
        target_research_question=target_paper.get('research_question', 'Not specified'),
        target_hypothesis=target_paper.get('hypothesis', 'Not specified'),
        target_proposed_method=target_paper.get('proposed_method', 'Not specified'),
        target_novelty_claims=target_paper.get('novelty_claims', 'Not specified'),
        target_experiment_details=target_paper.get('experiment_details', 'Not specified')
    )
    
    messages = [{"role": "user", "content": prompt}]
    
    try:
        response, cost = call_chat_completion(
            messages=messages,
            model=model,
            temperature=0.7
        )
        return response, cost
    except Exception as e:
        print(f"  ⚠ Stepwise reasoning generation failed: {e}")
        return "", 0.0


def extract_stepwise_sections(reasoning_text: str) -> Dict[str, str]:
    """Extract the 3 reasoning steps from stepwise CoT output."""
    import re
    
    sections = {
        'step1': '',  # Problem Identification
        'step2': '',  # Method Design Reasoning
        'step3': '',  # Experiment Design Reasoning
    }
    
    patterns = {
        'step1': [
            r'###?\s*Step 1[:\s]*Problem Identification\s*\n(.*?)(?=\n###?\s*Step 2|\Z)',
            r'###?\s*Step 1[:\s]*\s*\n(.*?)(?=\n###?\s*Step 2|\Z)',
        ],
        'step2': [
            r'###?\s*Step 2[:\s]*Method Design Reasoning\s*\n(.*?)(?=\n###?\s*Step 3|\Z)',
            r'###?\s*Step 2[:\s]*\s*\n(.*?)(?=\n###?\s*Step 3|\Z)',
        ],
        'step3': [
            r'###?\s*Step 3[:\s]*Experiment Design Reasoning\s*\n(.*?)(?=\n###?\s|\Z)',
            r'###?\s*Step 3[:\s]*\s*\n(.*?)(?=\n###?\s|\Z)',
        ],
    }
    
    for key, pats in patterns.items():
        for pat in pats:
            match = re.search(pat, reasoning_text, re.IGNORECASE | re.DOTALL)
            if match:
                sections[key] = match.group(1).strip()
                break
    
    return sections


def build_stepwise_completion(
    reasoning_steps: Dict[str, str],
    target_paper: Dict,
) -> str:
    """
    Build a step-by-step CoT completion:
    Reasoning1 -> RQ + Hypothesis -> Reasoning2 -> Method + Novelty -> Reasoning3 -> Experiments
    """
    parts = []
    
    title = target_paper.get('title', 'Unknown Title')
    year = target_paper.get('year')
    year_str = str(year) if year else 'Unknown Year'
    
    # Step 1: Problem Identification reasoning
    if reasoning_steps.get('step1'):
        parts.append("### Step 1: Problem Identification")
        parts.append(reasoning_steps['step1'])
    
    # Research Question + Hypothesis
    parts.append("\n## Proposed Research\n")
    parts.append(f"**{title}** ({year_str})")
    if target_paper.get('research_question'):
        parts.append(f"Research Question: {target_paper['research_question']}")
    if target_paper.get('hypothesis'):
        parts.append(f"Hypothesis: {target_paper['hypothesis']}")
    
    # Step 2: Method Design reasoning
    if reasoning_steps.get('step2'):
        parts.append(f"\n### Step 2: Method Design Reasoning")
        parts.append(reasoning_steps['step2'])
    
    # Proposed Method + Novelty Claims
    if target_paper.get('proposed_method'):
        parts.append(f"\nProposed Method: {target_paper['proposed_method']}")
    if target_paper.get('novelty_claims'):
        parts.append(f"Novelty Claims: {target_paper['novelty_claims']}")
    
    # Step 3: Experiment Design reasoning
    if reasoning_steps.get('step3'):
        parts.append(f"\n### Step 3: Experiment Design Reasoning")
        parts.append(reasoning_steps['step3'])
    
    # Experiment Details
    if target_paper.get('experiment_details'):
        parts.append(f"\nExperiment Details: {target_paper['experiment_details']}")
    
    return "\n".join(parts)


# ============================================================================
# One-Layer Tree Building (Simplified)
# ============================================================================

def build_one_layer_trees(
    paper_titles: List[str],
    api_key: Optional[str] = None,
    candidates_per_level: int = 10,
    selections_per_level: int = 4,
    model: str = "gpt-5-mini",
    rate_limit_delay: float = 1.1,
    checkpoint_dir: Optional[str] = None
) -> List[OneLayerTreeData]:
    """
    Build one-layer research trees (only direct citations).
    Returns just paper identifiers, not full tree structure.

    Args:
        paper_titles: List of paper titles to process
        api_key: Semantic Scholar API key
        candidates_per_level: Max candidates to show LLM
        selections_per_level: How many papers to select
        model: Model for selection
        rate_limit_delay: Delay between API calls
        checkpoint_dir: Directory to save checkpoints (enables resume on crash)

    Returns:
        List of OneLayerTreeData with just paper identifiers
    """
    # Use the existing builder with max_depth=1
    builder = LLMResearchTreeBuilder(
        api_key=api_key,
        max_depth=1,  # Only one layer
        candidates_per_level=candidates_per_level,
        selections_per_level=selections_per_level,
        model=model,
        rate_limit_delay=rate_limit_delay,
        min_citation_count=5,
        max_year_gap=5,
        recent_boost=100.0
    )

    trees = builder.build_trees_from_papers(
        paper_titles,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=100
    )

    # Convert to our simplified data structure
    tree_data_list = []
    for tree in trees:
        nodes = tree.nodes
        root_node = None
        inspiring_nodes = []

        for node in nodes:
            if node.depth == 0:
                root_node = node
            elif node.depth == 1:
                inspiring_nodes.append(node)

        if root_node and inspiring_nodes:
            tree_data = OneLayerTreeData(
                tree_id=tree.tree_id,
                root_paper_id=root_node.paper_id,
                inspiring_paper_ids=[node.paper_id for node in inspiring_nodes],
                selections_per_level=selections_per_level
            )
            tree_data_list.append(tree_data)

    return tree_data_list


# ============================================================================
# SFT Data Generation
# ============================================================================

def extract_reasoning_section(reasoning_text: str, section_name: str) -> str:
    """Extract a specific section from the reasoning text.
    
    Args:
        reasoning_text: Full reasoning text
        section_name: One of 'gap_analysis', 'inspiration_borrowing', 'synthesis'
    
    Returns:
        Extracted section text or empty string if not found
    """
    import re
    
    # Map section names to possible header patterns
    section_patterns = {
        'gap_analysis': [
            r'###?\s*Gap Analysis\s*\n(.*?)(?=\n###?\s|\Z)',
            r'##\s*Gap Analysis\s*\n(.*?)(?=\n##|\Z)',
        ],
        'inspiration_borrowing': [
            r'###?\s*Inspiration Borrowing\s*\n(.*?)(?=\n###?\s|\Z)',
            r'###?\s*Borrowed Ideas?\s*\n(.*?)(?=\n###?\s|\Z)',
        ],
        'synthesis': [
            r'###?\s*Synthesis\s*\n(.*?)(?=\n###?\s|\Z)',
            r'###?\s*Integration\s*\n(.*?)(?=\n###?\s|\Z)',
        ],
    }
    
    patterns = section_patterns.get(section_name, [])
    
    for pattern in patterns:
        match = re.search(pattern, reasoning_text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    
    return ""


def extract_all_reasoning_sections(reasoning_text: str) -> Dict[str, str]:
    """Extract all reasoning sections from the text.
    
    Returns:
        Dictionary with keys: gap_analysis, inspiration_borrowing, synthesis
    """
    return {
        'gap_analysis': extract_reasoning_section(reasoning_text, 'gap_analysis'),
        'inspiration_borrowing': extract_reasoning_section(reasoning_text, 'inspiration_borrowing'),
        'synthesis': extract_reasoning_section(reasoning_text, 'synthesis'),
    }


def extract_gap_analysis(reasoning_text: str) -> str:
    """Extract the gap analysis section from reasoning text (backward compatibility)."""
    return extract_reasoning_section(reasoning_text, 'gap_analysis')


def generate_sft_data(
    tree_data_list: List[OneLayerTreeData],
    structured_data: Dict[str, Dict],
    output_dir: str,
    reasoning_model: str = "gpt-4o",
    include_research_question: bool = True,
    include_reasoning: bool = False,
    max_samples: Optional[int] = None
) -> Tuple[List[OneLayerSFTExample], float]:
    """
    Generate SFT training data from one-layer research trees.

    Args:
        tree_data_list: List of OneLayerTreeData objects
        structured_data: Dictionary of structured paper summaries
        output_dir: Directory to save training data
        reasoning_model: Model to use for reasoning generation
        include_research_question: Whether to include RQ in the prompt
        include_reasoning: Whether to include synthetic reasoning in completion
        max_samples: Maximum samples to generate

    Returns:
        List of SFT examples
    """
    print("\n" + "="*80)
    print("GENERATING ONE-LAYER SFT DATA")
    print("="*80)
    print(f"Reasoning model: {reasoning_model}")
    print(f"Include research question: {include_research_question}")
    print(f"Include reasoning trace: {include_reasoning}")

    if max_samples:
        tree_data_list = tree_data_list[:max_samples]

    examples = []
    total_cost = 0.0

    for tree_data in tqdm(tree_data_list, desc="Generating SFT examples"):
        root_id = tree_data.root_paper_id
        inspiring_ids = tree_data.inspiring_paper_ids

        if not inspiring_ids:
            continue

        # Get structured data for root paper
        if root_id not in structured_data:
            tqdm.write(f"  ⚠ Root paper {root_id[:20]} not in structured data, skipping")
            continue

        root_structured = structured_data[root_id]

        # Get structured data for inspiring papers
        inspiring_papers = []
        for paper_id in inspiring_ids:
            if paper_id in structured_data:
                inspiring_papers.append(structured_data[paper_id])
            else:
                tqdm.write(f"  ⚠ Inspiring paper {paper_id[:20]} not in structured data, skipping tree")
                break
        else:
            # Only proceed if all inspiring papers are available
            pass

        if len(inspiring_papers) != len(inspiring_ids):
            continue
        
        # Build prompt
        prompt_parts = []
        
        # Optionally include research question
        if include_research_question:
            rq = root_structured.get('research_question', '')
            if rq:
                prompt_parts.append(f"Target Research Question: {rq}\n")
        
        prompt_parts.append("Given the following inspiring research papers:\n")
        
        for i, paper in enumerate(inspiring_papers, 1):
            prompt_parts.append(f"\n### Paper {i}:")
            prompt_parts.append(format_paper_for_prompt(paper))
            prompt_parts.append("")
        
        prompt_parts.append("\nBased on these inspiring papers, propose a novel research idea.")
        if include_research_question:
            prompt_parts.append("Your proposal should address the target research question.")
        
        prompt = "\n".join(prompt_parts)
        
        # Generate reasoning using strong LLM (if needed)
        reasoning = ""
        cost = 0.0
        if include_reasoning:
            reasoning, cost = generate_reasoning(
                inspiring_papers=inspiring_papers,
                target_paper=root_structured,
                model=reasoning_model
            )
            total_cost += cost

            if not reasoning:
                tqdm.write(f"  ⚠ Failed to generate reasoning for {root_id[:20]}")
                continue

        # Build completion
        completion_parts = []
        if include_reasoning:
            completion_parts.append(reasoning)
            completion_parts.append("\n")
        else:
            # Add a smooth transition without reasoning
            completion_parts.append("Based on these inspiring papers, here's a novel research proposal:")
            completion_parts.append("\n")

        completion_parts.append(format_target_paper(root_structured))

        completion = "\n".join(completion_parts)

        # Extract reasoning sections for ablation studies
        reasoning_sections = extract_all_reasoning_sections(reasoning)

        # Create example
        example = OneLayerSFTExample(
            tree_id=tree_data.tree_id,
            prompt=prompt,
            completion=completion,
            root_paper_id=root_id,
            root_paper_title=root_structured.get('title', 'Unknown'),
            research_question=root_structured.get('research_question', ''),
            num_inspiring_papers=len(inspiring_papers),
            reasoning_cost=cost,
            gap_analysis=reasoning_sections.get('gap_analysis', ''),
            inspiration_borrowing=reasoning_sections.get('inspiration_borrowing', ''),
            synthesis=reasoning_sections.get('synthesis', '')
        )
        examples.append(example)
    
    # Save training data
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    num_examples = len(examples)
    reasoning_suffix = "with_reasoning" if include_reasoning else "direct"
    training_file = output_path / f"one_layer_sft_n{num_examples}_{reasoning_suffix}_{timestamp}.json"
    with open(training_file, 'w') as f:
        json.dump([asdict(ex) for ex in examples], f, indent=2)

    # Save as JSONL (for training)
    jsonl_file = output_path / f"one_layer_sft_n{num_examples}_{reasoning_suffix}_{timestamp}.jsonl"
    with open(jsonl_file, 'w') as f:
        for ex in examples:
            f.write(json.dumps({
                'prompt': ex.prompt,
                'completion': ex.completion,
                'metadata': {
                    'tree_id': ex.tree_id,
                    'root_paper_id': ex.root_paper_id,
                    'root_title': ex.root_paper_title,
                    'research_question': ex.research_question,
                    'num_inspiring': ex.num_inspiring_papers,
                    'reasoning_cost': ex.reasoning_cost,
                    'gap_analysis': ex.gap_analysis
                }
            }) + '\n')
    
    print(f"\n✓ Generated {len(examples)} SFT examples")
    print(f"✓ Saved to: {training_file}")
    print(f"✓ Saved JSONL to: {jsonl_file}")
    print(f"💰 Total reasoning generation cost: ${total_cost:.4f}")

    # If reasoning was included, also save a non-reasoning version
    if include_reasoning:
        print(f"\n💡 Also saving non-reasoning version...")

        # Create non-reasoning examples by stripping reasoning from completions
        direct_examples = []
        for ex in examples:
            # Reuse the same prompt
            direct_prompt = ex.prompt

            # Strip reasoning from completion, keep only the proposal
            direct_completion = strip_reasoning_from_completion(ex.completion)

            direct_examples.append(OneLayerSFTExample(
                tree_id=ex.tree_id,
                prompt=direct_prompt,
                completion=direct_completion,
                root_paper_id=ex.root_paper_id,
                root_paper_title=ex.root_paper_title,
                research_question=ex.research_question,
                num_inspiring_papers=ex.num_inspiring_papers,
                reasoning_cost=0.0,  # No cost for direct version
                gap_analysis="",  # No sections for direct version
                inspiration_borrowing="",
                synthesis=""
            ))

        # Save direct version
        direct_suffix = "direct"
        direct_training_file = output_path / f"one_layer_sft_n{num_examples}_{direct_suffix}_{timestamp}.json"
        with open(direct_training_file, 'w') as f:
            json.dump({
                'metadata': {
                    'num_examples': len(direct_examples),
                    'include_reasoning': False,
                    'include_research_question': include_research_question,
                    'timestamp': timestamp,
                    'reasoning_model': reasoning_model if include_reasoning else None,
                    'total_cost': 0.0
                },
                'examples': [
                    {
                        'prompt': ex.prompt,
                        'completion': ex.completion,
                        'root_paper_id': ex.root_paper_id,
                        'root_title': ex.root_paper_title,
                        'research_question': ex.research_question,
                        'num_inspiring': ex.num_inspiring_papers,
                        'reasoning_cost': ex.reasoning_cost,
                        'gap_analysis': ex.gap_analysis
                    }
                    for ex in direct_examples
                ]
            }, f, indent=2)

        direct_jsonl_file = output_path / f"one_layer_sft_n{num_examples}_{direct_suffix}_{timestamp}.jsonl"
        with open(direct_jsonl_file, 'w') as f:
            for ex in direct_examples:
                f.write(json.dumps({
                    'prompt': ex.prompt,
                    'completion': ex.completion,
                    'root_paper_id': ex.root_paper_id,
                    'root_title': ex.root_paper_title,
                    'research_question': ex.research_question,
                    'num_inspiring': ex.num_inspiring_papers,
                    'reasoning_cost': ex.reasoning_cost,
                    'gap_analysis': ex.gap_analysis
                }) + '\n')

        print(f"✓ Saved direct version to: {direct_training_file}")
        print(f"✓ Saved direct JSONL to: {direct_jsonl_file}")

    return examples, total_cost


# ============================================================================
# Full Pipeline
# ============================================================================

def run_one_layer_pipeline(
    num_papers: int = 10,
    output_dir: str = "data/arxiv/one_layer",
    selection_model: str = "gpt-5-mini",
    reasoning_model: str = "gpt-5",
    structuring_model: str = "gpt-4.1",
    include_research_question: bool = True,
    include_reasoning: bool = False,
    use_existing_trees: Optional[str] = None,
    use_existing_structured: Optional[str] = None,
    selections_per_level: int = 5,
    max_samples: Optional[int] = None
):
    """
    Run the complete one-layer SFT data synthesis pipeline.

    Args:
        num_papers: Number of papers to process
        output_dir: Output directory
        selection_model: Model for inspiring paper selection
        reasoning_model: Model for reasoning generation (use strong model)
        structuring_model: Model for paper structuring
        include_research_question: Include RQ in prompt
        include_reasoning: Include synthetic reasoning in completion
        use_existing_trees: Path to existing trees file (skip tree building)
        use_existing_structured: Path to existing structured papers (skip structuring)
        selections_per_level: Number of inspiring papers to select
    """
    import random
    from synthesize.pipeline import download_tree_papers, structure_papers
    
    print("\n" + "#"*80)
    print("ONE-LAYER SFT DATA SYNTHESIS PIPELINE")
    print("#"*80)
    print(f"Papers to process: {num_papers}")
    print(f"Selection model: {selection_model}")
    print(f"Reasoning model: {reasoning_model}")
    print(f"Include research question: {include_research_question}")
    print(f"Selections per paper: {selections_per_level}")
    print("#"*80)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Build or load one-layer trees
    if use_existing_trees:
        print(f"\n⏭ Using existing trees: {use_existing_trees}")
        trees_file = use_existing_trees

        # If max_samples or num_papers is specified, we need to create a subset of trees
        limit_trees = max_samples if max_samples else (num_papers if num_papers < 100 else None)
        if limit_trees and limit_trees < 1000:  # Only apply if it's a reasonable test size
            print(f"⚠ Limiting to first {limit_trees} trees for testing")
            with open(trees_file, 'r') as f:
                trees_data = json.load(f)

            # Take only the first limit_trees trees
            original_count = len(trees_data.get('tree_data', []))
            trees_data['tree_data'] = trees_data['tree_data'][:limit_trees]
            trees_data['total_trees'] = len(trees_data['tree_data'])

            # Save subset to a temporary file
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
            json.dump(trees_data, temp_file, indent=2)
            temp_file.close()
            trees_file = temp_file.name

            print(f"✓ Created subset with {len(trees_data['tree_data'])} trees (from {original_count})")
    else:
        print("\n" + "="*80)
        print("STEP 1: Building One-Layer Research Trees")
        print("="*80)
        
        # Load NeurIPS 2024 papers and filter for LLM agent topics
        neurips_file = "data/accepted_papers/NeurIPS.cc_2024.json"
        iclr_file = "data/accepted_papers/ICLR.cc_2024.json"
        papers = []
        with open(neurips_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                title = data['title'].lower()
                # Focus on LLM agent and related topics
                # if any(keyword in title for keyword in [
                #     'agent', 'llm', 'language model', 'reinforcement learning',
                #     'reasoning', 'planning', 'tool use', 'multi-agent'
                # ]):
                papers.append(data['title'])
        
        with open(iclr_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                title = data['title'].lower()
                papers.append(data['title'])

        print(f"Found {len(papers)} papers from NeurIPS 2024 and ICLR 2024")

        # Sample papers
        selected_papers = random.sample(papers, min(num_papers, len(papers)))
        print(f"Selected {len(selected_papers)} papers for processing")
        
        # Load API key
        api_key = load_api_key()
        
        # Build trees
        trees = build_one_layer_trees(
            paper_titles=selected_papers,
            api_key=api_key,
            candidates_per_level=15,
            selections_per_level=selections_per_level,
            model=selection_model,
            rate_limit_delay=1.1
        )
        
        # Save trees (simplified format)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        trees_file = str(output_path / f"one_layer_trees_n{len(trees)}_{timestamp}.json")

        trees_data = {
            'config': {
                'max_depth': 1,
                'selections_per_level': selections_per_level,
                'model': selection_model
            },
            'tree_data': [asdict(tree) for tree in trees],
            'total_trees': len(trees)
        }

        with open(trees_file, 'w') as f:
            json.dump(trees_data, f, indent=2)
        print(f"✓ Saved trees to: {trees_file}")

    # Step 2: Download and structure papers (reuse existing pipeline)
    if use_existing_structured:
        print(f"\n⏭ Using existing structured papers: {use_existing_structured}")
        with open(use_existing_structured, 'r') as f:
            structured_data = json.load(f)
    else:
        print("\n" + "="*80)
        print("STEP 2: Downloading and Structuring Papers")
        print("="*80)

        download_results, paper_info = download_tree_papers(trees_file, str(output_path))
        structured_data = structure_papers(
            download_results, paper_info, str(output_path), model=structuring_model
        )

    # Load tree data for SFT generation
    with open(trees_file, 'r') as f:
        trees_data = json.load(f)
        tree_data_list = [OneLayerTreeData(**td) for td in trees_data['tree_data']]

    # Step 3: Generate SFT data with reasoning
    examples, reasoning_cost = generate_sft_data(
        tree_data_list=tree_data_list,
        structured_data=structured_data,
        output_dir=str(output_path),
        reasoning_model=reasoning_model,
        include_research_question=include_research_question,
        include_reasoning=include_reasoning
    )
    
    # Summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"SFT examples generated: {len(examples)}")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    return examples


def generate_from_existing_data(
    trees_file: str,
    structured_file: str,
    output_dir: str,
    reasoning_model: str = "gpt-5",
    include_research_question: bool = True,
    include_reasoning: bool = False,
    max_samples: Optional[int] = None
):
    """
    Generate SFT data from existing trees and structured papers.

    This is useful when you already have trees built and just want to
    generate the reasoning and SFT data.
    """
    print("\n" + "#"*80)
    print("GENERATING SFT DATA FROM EXISTING DATA")
    print("#"*80)
    print(f"Trees file: {trees_file}")
    print(f"Structured file: {structured_file}")
    print(f"Reasoning model: {reasoning_model}")
    print(f"Include reasoning trace: {include_reasoning}")
    print("#"*80)

    # Load structured data
    with open(structured_file, 'r') as f:
        structured_data = json.load(f)

    # Load tree data
    with open(trees_file, 'r') as f:
        trees_data = json.load(f)
        tree_data_list = [OneLayerTreeData(**td) for td in trees_data.get('tree_data', trees_data.get('trees', []))]

    # Generate SFT data
    examples, reasoning_cost = generate_sft_data(
        tree_data_list=tree_data_list,
        structured_data=structured_data,
        output_dir=output_dir,
        reasoning_model=reasoning_model,
        include_research_question=include_research_question,
        include_reasoning=include_reasoning,
        max_samples=max_samples
    )

    return examples


def convert_reasoning_to_direct(
    input_file: str,
    output_dir: str = None
) -> str:
    """
    Convert existing SFT data with reasoning to direct (non-reasoning) format.
    
    This is useful when you already have generated reasoning data and want to
    create a non-reasoning version without re-running the pipeline.
    
    Args:
        input_file: Path to existing SFT data JSON file (with reasoning)
        output_dir: Output directory (default: same as input file)
    
    Returns:
        Path to the converted file
    """
    print(f"\n🔄 Converting reasoning data to direct format...")
    print(f"   Input: {input_file}")
    
    # Load existing data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Handle both list format and dict format with metadata
    if isinstance(data, list):
        examples = data
    else:
        examples = data.get('examples', data.get('data', []))
    
    if not examples:
        print("❌ No examples found in input file")
        return None
    
    print(f"   Found {len(examples)} examples")
    
    # Convert each example
    converted = []
    for ex in examples:
        completion = ex.get('completion', '')
        direct_completion = strip_reasoning_from_completion(completion)
        
        converted_ex = {
            **ex,
            'completion': direct_completion,
            'reasoning_cost': 0.0,
            'gap_analysis': ''
        }
        converted.append(converted_ex)
    
    # Determine output path
    input_path = Path(input_file)
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = input_path.parent
    
    # Generate output filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    num_examples = len(converted)
    output_file = output_path / f"one_layer_sft_n{num_examples}_direct_{timestamp}.json"
    
    # Save JSON
    with open(output_file, 'w') as f:
        json.dump(converted, f, indent=2)
    print(f"✓ Saved JSON to: {output_file}")
    
    # Save JSONL
    jsonl_file = output_path / f"one_layer_sft_n{num_examples}_direct_{timestamp}.jsonl"
    with open(jsonl_file, 'w') as f:
        for ex in converted:
            f.write(json.dumps({
                'prompt': ex.get('prompt', ''),
                'completion': ex.get('completion', ''),
                'metadata': {
                    'tree_id': ex.get('tree_id', ''),
                    'root_paper_id': ex.get('root_paper_id', ''),
                    'root_title': ex.get('root_paper_title', ''),
                    'research_question': ex.get('research_question', ''),
                    'num_inspiring': ex.get('num_inspiring_papers', 0),
                    'reasoning_cost': 0.0,
                    'gap_analysis': ''
                }
            }) + '\n')
    print(f"✓ Saved JSONL to: {jsonl_file}")
    
    return str(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="One-Layer SFT Data Synthesis Pipeline")
    
    parser.add_argument(
        "--num-papers",
        type=int,
        default=10,
        help="Number of papers to process (default: 10)"
    )
    parser.add_argument(
        "--output-dir",
        default="data/arxiv/one_layer",
        help="Output directory"
    )
    parser.add_argument(
        "--selection-model",
        default="gpt-5-mini",
        help="Model for inspiring paper selection (default: gpt-5-mini)"
    )
    parser.add_argument(
        "--reasoning-model",
        default="gpt-5",
        help="Model for reasoning generation (default: gpt-5)"
    )
    parser.add_argument(
        "--structuring-model",
        default="gpt-4.1",
        help="Model for paper structuring (default: gpt-4.1)"
    )
    parser.add_argument(
        "--selections",
        type=int,
        default=5,
        help="Number of inspiring papers to select per root paper (default: 5)"
    )
    parser.add_argument(
        "--no-research-question",
        action="store_true",
        help="Do not include research question in the prompt"
    )
    parser.add_argument(
        "--include-reasoning",
        action="store_true",
        help="Include synthetic reasoning trace in the completion (default: direct proposals only)"
    )
    parser.add_argument(
        "--use-existing-trees",
        default=None,
        help="Path to existing trees file (skip tree building)"
    )
    parser.add_argument(
        "--use-existing-structured",
        default=None,
        help="Path to existing structured papers file (skip structuring)"
    )
    parser.add_argument(
        "--from-existing",
        action="store_true",
        help="Generate SFT data from existing trees and structured papers"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to generate (for testing)"
    )
    parser.add_argument(
        "--convert-to-direct",
        default=None,
        help="Convert existing reasoning SFT data to direct (non-reasoning) format. Provide path to input JSON file."
    )
    
    args = parser.parse_args()
    
    # Handle conversion mode
    if args.convert_to_direct:
        convert_reasoning_to_direct(
            input_file=args.convert_to_direct,
            output_dir=args.output_dir
        )
        sys.exit(0)
    
    if args.from_existing:
        if not args.use_existing_trees or not args.use_existing_structured:
            print("Error: --from-existing requires --use-existing-trees and --use-existing-structured")
            sys.exit(1)
        
        generate_from_existing_data(
            trees_file=args.use_existing_trees,
            structured_file=args.use_existing_structured,
            output_dir=args.output_dir,
            reasoning_model=args.reasoning_model,
            include_research_question=not args.no_research_question,
            include_reasoning=args.include_reasoning,
            max_samples=args.max_samples
        )
    else:
        run_one_layer_pipeline(
            num_papers=args.num_papers,
            output_dir=args.output_dir,
            selection_model=args.selection_model,
            reasoning_model=args.reasoning_model,
            structuring_model=args.structuring_model,
            include_research_question=not args.no_research_question,
            include_reasoning=args.include_reasoning,
            use_existing_trees=args.use_existing_trees,
            use_existing_structured=args.use_existing_structured,
            selections_per_level=args.selections,
            max_samples=args.max_samples
        )

