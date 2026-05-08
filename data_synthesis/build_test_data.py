#!/usr/bin/env python3
"""
Build Test Data from ICLR'25, NeurIPS'25, ICML'25 papers.

This script generates test data similar to SFT data format:
- Input: Inspiring papers + Research Question
- Output: Target paper (without reasoning)

Usage:
    python synthesize/build_test_data.py --num-papers 1000 --output-dir data/test_set
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthesize.one_layer_sft import (
    OneLayerTreeData,
    OneLayerSFTExample,
    build_one_layer_trees,
    format_paper_for_prompt,
    format_target_paper,
    generate_sft_data,
)
from synthesize.build import load_api_key
from synthesize.pipeline import download_tree_papers, structure_papers


# Conference paper files
CONFERENCE_FILES = {
    'ICLR_2025': 'data/accepted_papers/ICLR.cc_2025.json',
    'ICML_2025': 'data/accepted_papers/ICML.cc_2025.json',
    'NeurIPS_2025': 'data/accepted_papers/NeurIPS.cc_2025.json',
}


def load_conference_papers(conference_files: Dict[str, str]) -> Dict[str, List[Dict]]:
    """Load papers from conference files."""
    papers_by_venue = {}
    
    for venue, filepath in conference_files.items():
        papers = []
        if not os.path.exists(filepath):
            print(f"⚠ File not found: {filepath}")
            continue
            
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    paper = json.loads(line.strip())
                    papers.append(paper)
                except json.JSONDecodeError:
                    continue
        
        papers_by_venue[venue] = papers
        print(f"✓ Loaded {len(papers)} papers from {venue}")
    
    return papers_by_venue


def sample_papers(
    papers_by_venue: Dict[str, List[Dict]],
    total_papers: int = 1000,
    balanced: bool = True,
    exclude_ids: Optional[set] = None
) -> List[Dict]:
    """Sample papers from venues, optionally balanced across venues.
    
    Args:
        papers_by_venue: Papers organized by venue
        total_papers: Total number of papers to sample
        balanced: Whether to balance across venues
        exclude_ids: Set of paper IDs to exclude from sampling
    """
    exclude_ids = exclude_ids or set()
    
    # Filter out excluded papers
    if exclude_ids:
        filtered_by_venue = {}
        for venue, papers in papers_by_venue.items():
            filtered = [p for p in papers if p.get('id') not in exclude_ids]
            filtered_by_venue[venue] = filtered
            excluded_count = len(papers) - len(filtered)
            if excluded_count > 0:
                print(f"  Excluded {excluded_count} papers from {venue}")
        papers_by_venue = filtered_by_venue
    
    if balanced:
        # Equal sampling from each venue
        num_venues = len(papers_by_venue)
        papers_per_venue = total_papers // num_venues
        
        sampled = []
        for venue, papers in papers_by_venue.items():
            n_sample = min(papers_per_venue, len(papers))
            venue_sample = random.sample(papers, n_sample)
            # Add venue info to each paper
            for p in venue_sample:
                p['_venue'] = venue
            sampled.extend(venue_sample)
            print(f"  Sampled {n_sample} from {venue}")
        
        # If we need more papers to reach total, sample from largest venue
        remaining = total_papers - len(sampled)
        if remaining > 0:
            largest_venue = max(papers_by_venue.keys(), key=lambda v: len(papers_by_venue[v]))
            already_sampled_ids = {p.get('id') for p in sampled}
            available = [p for p in papers_by_venue[largest_venue] if p.get('id') not in already_sampled_ids]
            extra = random.sample(available, min(remaining, len(available)))
            for p in extra:
                p['_venue'] = largest_venue
            sampled.extend(extra)
    else:
        # Random sampling across all papers
        all_papers = []
        for venue, papers in papers_by_venue.items():
            for p in papers:
                p['_venue'] = venue
            all_papers.extend(papers)
        sampled = random.sample(all_papers, min(total_papers, len(all_papers)))
    
    random.shuffle(sampled)
    return sampled


def build_test_data(
    num_papers: int = 1000,
    output_dir: str = "data/test_set",
    selection_model: str = "gpt-4o-mini",
    structuring_model: str = "gpt-4o",
    selections_per_level: int = 5,
    use_existing_trees: Optional[str] = None,
    use_existing_structured: Optional[str] = None,
    balanced: bool = True,
    exclude_file: Optional[str] = None,
):
    """
    Build test data from 2025 conference papers.
    
    Args:
        num_papers: Total number of test papers to generate
        output_dir: Output directory for test data
        selection_model: Model for inspiring paper selection
        structuring_model: Model for paper structuring
        selections_per_level: Number of inspiring papers per target
        use_existing_trees: Path to existing trees file (skip tree building)
        use_existing_structured: Path to existing structured papers
        balanced: Whether to balance sampling across venues
        exclude_file: Path to JSON file with papers to exclude (e.g., sampled_papers_3000.json)
    """
    print("\n" + "#"*80)
    print("BUILDING TEST DATA FROM 2025 CONFERENCE PAPERS")
    print("#"*80)
    print(f"Target papers: {num_papers}")
    print(f"Selection model: {selection_model}")
    print(f"Structuring model: {structuring_model}")
    print(f"Selections per paper: {selections_per_level}")
    print(f"Balanced sampling: {balanced}")
    print(f"Exclude file: {exclude_file or 'None'}")
    print("#"*80)
    
    # Load excluded paper IDs if provided
    exclude_ids = set()
    if exclude_file and os.path.exists(exclude_file):
        print(f"\nLoading excluded papers from: {exclude_file}")
        with open(exclude_file, 'r') as f:
            exclude_data = json.load(f)
        
        # Handle both formats: list of papers or dict with 'papers' key
        if isinstance(exclude_data, dict) and 'papers' in exclude_data:
            exclude_papers = exclude_data['papers']
        elif isinstance(exclude_data, list):
            exclude_papers = exclude_data
        else:
            exclude_papers = []
        
        for p in exclude_papers:
            if isinstance(p, dict) and 'id' in p:
                exclude_ids.add(p['id'])
            elif isinstance(p, str):
                exclude_ids.add(p)
        
        print(f"✓ Loaded {len(exclude_ids)} paper IDs to exclude")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load and sample papers
    if use_existing_trees:
        print(f"\n⏭ Using existing trees: {use_existing_trees}")
        trees_file = use_existing_trees
    else:
        print("\n" + "="*80)
        print("STEP 1: Loading and Sampling Conference Papers")
        print("="*80)
        
        papers_by_venue = load_conference_papers(CONFERENCE_FILES)
        
        if not papers_by_venue:
            print("❌ No papers loaded from any venue!")
            return
        
        sampled_papers = sample_papers(papers_by_venue, num_papers, balanced, exclude_ids)
        print(f"\n✓ Total sampled: {len(sampled_papers)} papers")
        
        # Extract titles for tree building
        paper_titles = [p['title'] for p in sampled_papers]
        
        # Save sampled paper info
        sampled_file = output_path / f"sampled_papers_{len(sampled_papers)}.json"
        with open(sampled_file, 'w') as f:
            json.dump({
                'total': len(sampled_papers),
                'by_venue': {v: sum(1 for p in sampled_papers if p.get('_venue') == v) for v in CONFERENCE_FILES.keys()},
                'papers': sampled_papers
            }, f, indent=2)
        print(f"✓ Saved sampled papers to: {sampled_file}")
        
        # Step 2: Build one-layer trees
        print("\n" + "="*80)
        print("STEP 2: Building One-Layer Research Trees")
        print("="*80)
        
        api_key = load_api_key()
        
        trees = build_one_layer_trees(
            paper_titles=paper_titles,
            api_key=api_key,
            candidates_per_level=15,
            selections_per_level=selections_per_level,
            model=selection_model,
            rate_limit_delay=1.1,
            checkpoint_dir=str(output_path)  # Enable checkpointing for resume
        )
        
        # Save trees
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        trees_file = str(output_path / f"test_trees_n{len(trees)}_{timestamp}.json")
        
        trees_data = {
            'config': {
                'max_depth': 1,
                'selections_per_level': selections_per_level,
                'model': selection_model,
                'source_venues': list(CONFERENCE_FILES.keys()),
            },
            'tree_data': [asdict(tree) for tree in trees],
            'total_trees': len(trees)
        }
        
        with open(trees_file, 'w') as f:
            json.dump(trees_data, f, indent=2)
        print(f"✓ Saved {len(trees)} trees to: {trees_file}")
    
    # Step 3: Download and structure papers
    if use_existing_structured:
        print(f"\n⏭ Using existing structured papers: {use_existing_structured}")
        with open(use_existing_structured, 'r') as f:
            structured_data = json.load(f)
    else:
        print("\n" + "="*80)
        print("STEP 3: Downloading and Structuring Papers")
        print("="*80)
        
        download_results, paper_info = download_tree_papers(trees_file, str(output_path))
        structured_data, structuring_cost = structure_papers(
            download_results, paper_info, str(output_path), model=structuring_model
        )
        print(f"💰 Structuring cost: ${structuring_cost:.4f}")
    
    # Load tree data
    with open(trees_file, 'r') as f:
        trees_data = json.load(f)
        tree_data_list = [OneLayerTreeData(**td) for td in trees_data['tree_data']]
    
    # Step 4: Generate test data (without reasoning)
    print("\n" + "="*80)
    print("STEP 4: Generating Test Data")
    print("="*80)
    
    examples, _ = generate_sft_data(
        tree_data_list=tree_data_list,
        structured_data=structured_data,
        output_dir=str(output_path),
        reasoning_model="gpt-4o",  # Not used since include_reasoning=False
        include_research_question=True,
        include_reasoning=False  # Test data without reasoning
    )
    
    # Rename output files to indicate test data
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Create final test set file with clear naming
    test_file = output_path / f"test_set_n{len(examples)}_{timestamp}.jsonl"
    with open(test_file, 'w') as f:
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
                }
            }) + '\n')
    
    # Generate test_subset_100
    subset_size = min(100, len(examples))
    subset = random.sample(examples, subset_size)
    subset_file = output_path / "test_subset_100.jsonl"
    
    # Backup existing subset if present
    if subset_file.exists():
        backup = output_path / f"test_subset_100.backup_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
        import shutil
        shutil.copy2(subset_file, backup)
        print(f"✓ Backed up existing test_subset_100.jsonl -> {backup.name}")
    
    with open(subset_file, 'w') as f:
        for ex in subset:
            f.write(json.dumps({
                'prompt': ex.prompt,
                'completion': ex.completion,
                'metadata': {
                    'tree_id': ex.tree_id,
                    'root_paper_id': ex.root_paper_id,
                    'root_title': ex.root_paper_title,
                    'research_question': ex.research_question,
                    'num_inspiring': ex.num_inspiring_papers,
                }
            }) + '\n')
    print(f"✓ Saved test_subset_100.jsonl ({subset_size} samples)")
    
    # Summary
    print("\n" + "="*80)
    print("TEST DATA GENERATION COMPLETE")
    print("="*80)
    print(f"Test examples generated: {len(examples)}")
    print(f"Output file: {test_file}")
    print(f"Subset file: {subset_file} ({subset_size} samples)")
    print("="*80)
    
    return examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Test Data from 2025 Conference Papers")
    
    parser.add_argument(
        "--num-papers",
        type=int,
        default=1000,
        help="Total number of papers to sample (default: 1000)"
    )
    parser.add_argument(
        "--output-dir",
        default="data/test_set",
        help="Output directory for test data"
    )
    parser.add_argument(
        "--selection-model",
        default="gpt-5-mini",
        help="Model for inspiring paper selection (default: gpt-5-mini)"
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
        help="Number of inspiring papers per target (default: 5)"
    )
    parser.add_argument(
        "--use-existing-trees",
        default=None,
        help="Path to existing trees file (skip tree building)"
    )
    parser.add_argument(
        "--use-existing-structured",
        default=None,
        help="Path to existing structured papers file"
    )
    parser.add_argument(
        "--unbalanced",
        action="store_true",
        help="Don't balance sampling across venues"
    )
    parser.add_argument(
        "--exclude",
        default=None,
        help="Path to JSON file with papers to exclude (e.g., sampled_papers_3000.json)"
    )
    
    args = parser.parse_args()
    
    build_test_data(
        num_papers=args.num_papers,
        output_dir=args.output_dir,
        selection_model=args.selection_model,
        structuring_model=args.structuring_model,
        selections_per_level=args.selections,
        use_existing_trees=args.use_existing_trees,
        use_existing_structured=args.use_existing_structured,
        balanced=not args.unbalanced,
        exclude_file=args.exclude,
    )
