#!/usr/bin/env python3
"""
Research Tree Pipeline: Download, Extract, Structure, and Generate Training Data.

This pipeline:
1. Downloads all papers in research trees from arXiv
2. Extracts main content (before appendix/references)
3. Converts papers to structured summaries using LLM
4. Generates training prompts and answers from trees
"""

import os
import sys
import json
import re
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.arxiv_download import ArxivDownloader
from utils.semantic_api import SemanticScholarAPI
from utils.api import call_chat_completion
from data_proc.structuring import extract_paper_structure

import pdfplumber
from PyPDF2 import PdfReader, PdfWriter


# ============================================================================
# Step 1: Download Papers
# ============================================================================

def download_tree_papers(
    trees_file: str,
    output_dir: str = "data/arxiv/test",
    delay: float = 1.0
) -> Tuple[Dict[str, Optional[str]], Dict[str, Dict]]:
    """
    Download all papers from research trees.

    Args:
        trees_file: Path to the research trees JSON file (can contain either full trees or just paper IDs)
        output_dir: Directory to save downloaded PDFs
        delay: Delay between downloads

    Returns:
        Tuple of:
        - Dictionary mapping paper_id to PDF path (or None if failed)
        - Dictionary mapping paper_id to paper info (title, year, etc.)
    """
    print("\n" + "="*80)
    print("STEP 1: Downloading Papers from ArXiv")
    print("="*80)

    # Load trees
    with open(trees_file, 'r') as f:
        data = json.load(f)

    trees = data.get('trees', [])
    tree_data = data.get('tree_data', [])  # Support new format

    # Collect all unique papers
    all_papers = {}

    # Handle full tree format (legacy)
    for tree in trees:
        for node in tree.get('nodes', []):
            paper_id = node.get('paper_id')
            if paper_id and paper_id not in all_papers:
                all_papers[paper_id] = node

    # Handle simplified tree_data format
    for tree_item in tree_data:
        # Add root paper
        root_id = tree_item.get('root_paper_id')
        if root_id and root_id not in all_papers:
            # Create minimal node info - will be enriched later
            all_papers[root_id] = {'paper_id': root_id, 'depth': 0}

        # Add inspiring papers
        for paper_id in tree_item.get('inspiring_paper_ids', []):
            if paper_id and paper_id not in all_papers:
                # Create minimal node info - will be enriched later
                all_papers[paper_id] = {'paper_id': paper_id, 'depth': 1}

    print(f"Found {len(all_papers)} unique papers across {len(trees)} trees")

    # Check for existing paper info to avoid re-fetching metadata
    paper_info_file = Path(output_dir) / "paper_info.json"
    existing_paper_info = {}
    if paper_info_file.exists():
        try:
            with open(paper_info_file, 'r') as f:
                existing_paper_info = json.load(f)
            print(f"Loaded {len(existing_paper_info)} existing paper metadata")
        except Exception as e:
            print(f"Could not load existing paper info: {e}")

    # Update all_papers with existing info
    for paper_id, existing_info in existing_paper_info.items():
        if paper_id in all_papers:
            all_papers[paper_id].update(existing_info)
    
    # Check for existing download results to skip already processed papers
    results_file = Path(output_dir) / "download_results.json"
    existing_results = {}
    if results_file.exists():
        try:
            with open(results_file, 'r') as f:
                existing_results = json.load(f)
            print(f"Loaded {len(existing_results)} existing download results")
        except Exception as e:
            print(f"Could not load existing download results: {e}")
    
    # Initialize downloader and API - store PDFs in a 'pdfs' subdirectory
    pdfs_dir = Path(output_dir) / "pdfs"
    pdfs_dir.mkdir(parents=True, exist_ok=True)
    downloader = ArxivDownloader(cache_dir=str(pdfs_dir))
    api = SemanticScholarAPI()
    
    # Download each paper
    results = dict(existing_results)  # Start with existing results
    downloaded = 0
    cached = 0
    skipped = 0
    failed = 0
    
    # Determine which papers actually need processing
    papers_to_process = []
    for paper_id, node in all_papers.items():
        # Skip if already in results with a valid, existing path
        if paper_id in existing_results:
            existing_path = existing_results[paper_id]
            if existing_path and Path(existing_path).exists():
                skipped += 1
                continue
        papers_to_process.append((paper_id, node))
    
    print(f"Skipping {skipped} papers (already downloaded)")
    print(f"Processing {len(papers_to_process)} papers...")
    
    for i, (paper_id, node) in enumerate(tqdm(papers_to_process, desc="Downloading")):
        title = node.get('title', 'Unknown')

        # First check if we have arXiv info in the node
        # If not, fetch from Semantic Scholar
        external_ids = node.get('externalIds')

        if not external_ids:
            # Fetch paper details - try by ID first (for simplified format), then by title
            time.sleep(delay)
            if title != 'Unknown':
                paper_details = api.get_paper_by_title(title)
            else:
                # For simplified format, we only have paper_id, so fetch by ID
                paper_details = api.get_paper_by_id(paper_id)
            if paper_details:
                external_ids = paper_details.get('externalIds', {})
                # Update the paper info with fetched details
                all_papers[paper_id].update({
                    'title': paper_details.get('title', node.get('title', 'Unknown')),
                    'year': paper_details.get('year', node.get('year')),
                    'abstract': paper_details.get('abstract', node.get('abstract')),
                    'venue': paper_details.get('venue', node.get('venue')),
                    'authors': paper_details.get('authors', node.get('authors', [])),
                    'citationCount': paper_details.get('citationCount', node.get('citationCount')),
                    'externalIds': external_ids
                })
                # Update title variable for downloading
                title = paper_details.get('title', title)
        
        if external_ids and external_ids.get('ArXiv'):
            arxiv_id = external_ids['ArXiv']
            
            # Check if already cached
            if downloader.is_cached(arxiv_id):
                results[paper_id] = str(downloader.get_cached_path(arxiv_id))
                cached += 1
            else:
                time.sleep(delay)
                path = downloader.download(arxiv_id, title=title)
                if path:
                    results[paper_id] = str(path)
                    downloaded += 1
                else:
                    results[paper_id] = None
                    failed += 1
        else:
            tqdm.write(f"  ⚠ No arXiv ID: {title[:50]}...")
            results[paper_id] = None
            failed += 1
    
    print(f"\n✓ Download complete: {downloaded} new, {cached} cached, {skipped} skipped, {failed} failed")
    
    # Save download results
    results_file = Path(output_dir) / "download_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to: {results_file}")
    
    # Save paper info for later use
    paper_info_file = Path(output_dir) / "paper_info.json"
    with open(paper_info_file, 'w') as f:
        json.dump(all_papers, f, indent=2)
    
    return results, all_papers


def enrich_paper_info(
    paper_info: Dict[str, Dict],
    structured_data_file: Optional[str] = None,
    api_key: Optional[str] = None
) -> Dict[str, Dict]:
    """
    Enrich paper information with full details from structured data or Semantic Scholar.

    Args:
        paper_info: Dictionary mapping paper_id to minimal paper info
        structured_data_file: Path to existing structured_papers.json file
        api_key: Semantic Scholar API key for fetching missing info

    Returns:
        Enriched paper_info dictionary
    """
    enriched_info = paper_info.copy()
    api = SemanticScholarAPI(api_key) if api_key else None

    # Load existing structured data if available
    existing_structured = {}
    if structured_data_file and os.path.exists(structured_data_file):
        with open(structured_data_file, 'r') as f:
            existing_structured = json.load(f)
        print(f"Loaded {len(existing_structured)} existing structured papers")

    # Enrich each paper
    for paper_id, info in tqdm(paper_info.items(), desc="Enriching paper info"):
        if paper_id in existing_structured:
            # Use existing structured data
            structured = existing_structured[paper_id]
            enriched_info[paper_id] = {
                'paper_id': paper_id,
                'title': structured.get('title', info.get('title', 'Unknown')),
                'year': structured.get('year', info.get('year')),
                'abstract': structured.get('abstract'),
                'venue': structured.get('venue'),
                'citation_count': structured.get('citation_count'),
                'depth': info.get('depth', 0)
            }
        elif api:
            # Fetch from Semantic Scholar
            try:
                time.sleep(0.5)  # Rate limiting
                paper_details = api.get_paper_by_title(info.get('title', '') or paper_id)
                if paper_details:
                    enriched_info[paper_id] = {
                        'paper_id': paper_id,
                        'title': paper_details.get('title', info.get('title', 'Unknown')),
                        'year': paper_details.get('year', info.get('year')),
                        'abstract': paper_details.get('abstract'),
                        'venue': paper_details.get('venue'),
                        'citation_count': paper_details.get('citationCount', 0),
                        'depth': info.get('depth', 0)
                    }
                else:
                    print(f"  ⚠ Could not find paper: {paper_id}")
            except Exception as e:
                print(f"  ⚠ Error fetching {paper_id}: {e}")

    return enriched_info


def load_structured_data_by_id(structured_file: str) -> Dict[str, Dict]:
    """
    Load structured paper data indexed by paper_id.

    Args:
        structured_file: Path to structured_papers.json file

    Returns:
        Dictionary mapping paper_id to structured paper data
    """
    if not os.path.exists(structured_file):
        return {}

    with open(structured_file, 'r') as f:
        structured_data = json.load(f)

    # Index by paper_id
    by_id = {}
    for paper_id, data in structured_data.items():
        by_id[paper_id] = data

    return by_id


# ============================================================================
# Step 2: Extract Main Content
# ============================================================================

def find_section_end(text: str, section_names: List[str]) -> int:
    """Find the start position of a section by its name."""
    patterns = [
        rf'\n\s*{re.escape(name)}\s*\n',  # Section as header
        rf'\n\s*\d+\.?\s+{re.escape(name)}\s*\n',  # Numbered section
        rf'\n\s*[A-Z]\.?\s+{re.escape(name)}\s*\n',  # Lettered section
    ]
    
    min_pos = len(text)
    for name in section_names:
        for pattern in patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            for match in matches:
                if match.start() < min_pos:
                    min_pos = match.start()
    
    return min_pos if min_pos < len(text) else -1


def extract_main_content_from_pdf(
    pdf_path: str,
    max_pages: int = 15
) -> Tuple[str, int]:
    """
    Extract main content from PDF, excluding appendix and references.
    
    Args:
        pdf_path: Path to the PDF file
        max_pages: Maximum pages to read
    
    Returns:
        Tuple of (extracted_text, pages_read)
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            pages_to_read = min(max_pages, total_pages)
            
            text = ""
            for i in range(pages_to_read):
                page_text = pdf.pages[i].extract_text()
                if page_text:
                    text += page_text + "\n"
            
            # Try to find and remove appendix/references
            end_sections = [
                'References', 'REFERENCES', 'Bibliography',
                'Appendix', 'APPENDIX', 'Appendices',
                'Supplementary Material', 'Supplementary'
            ]
            
            end_pos = find_section_end(text, end_sections)
            if end_pos > 0:
                text = text[:end_pos]
            
            return text.strip(), pages_to_read
            
    except Exception as e:
        print(f"  ✗ Error extracting: {e}")
        return "", 0


def extract_main_pages_to_pdf(
    input_pdf: str,
    output_pdf: str,
    max_pages: int = 10
) -> bool:
    """
    Extract main pages from PDF to a new file.
    
    Args:
        input_pdf: Path to input PDF
        output_pdf: Path to output PDF
        max_pages: Maximum pages to extract
    
    Returns:
        True if successful
    """
    try:
        reader = PdfReader(input_pdf)
        writer = PdfWriter()
        
        pages_to_extract = min(max_pages, len(reader.pages))
        
        for i in range(pages_to_extract):
            writer.add_page(reader.pages[i])
        
        os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
        with open(output_pdf, 'wb') as f:
            writer.write(f)
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


# ============================================================================
# Step 3: Structure Papers
# ============================================================================

def structure_papers(
    download_results: Dict[str, Optional[str]],
    paper_info: Dict[str, Dict],
    output_dir: str,
    model: str = "gpt-4o-mini",
    max_pages: int = 10
) -> Dict[str, Dict]:
    """
    Convert downloaded papers to structured summaries.
    
    Args:
        download_results: Dictionary mapping paper_id to PDF path
        paper_info: Dictionary mapping paper_id to paper info (title, year, etc.)
        output_dir: Directory to save structured data
        model: Model to use for structuring
        max_pages: Max pages to extract from each PDF
    
    Returns:
        Dictionary mapping paper_id to structured summary
    """
    print("\n" + "="*80)
    print("STEP 2-3: Extracting Content and Structuring Papers")
    print("="*80)
    
    output_path = Path(output_dir)
    
    # Store extracted PDFs in pdfs_main/ with same filename as original
    pdfs_main_dir = output_path / "pdfs_main"
    pdfs_main_dir.mkdir(parents=True, exist_ok=True)
    
    structured_data = {}
    total_cost = 0.0
    
    # Load existing structured data if available
    structured_file = output_path / "structured_papers.json"
    if structured_file.exists():
        with open(structured_file, 'r') as f:
            structured_data = json.load(f)
        print(f"Loaded {len(structured_data)} existing structured papers")
    # add title and year to structured_data
    # for paper_id, paper_info in paper_info.items():
    #     if paper_id in structured_data:
    #         structured_data[paper_id]['title'] = paper_info['title']
    #         structured_data[paper_id]['year'] = paper_info['year']
    #     else:
    #         print(f"Paper {paper_id} not found in structured_data")
    
    # Process each paper
    papers_to_process = [
        (pid, path) for pid, path in download_results.items()
        if path and pid not in structured_data
    ]
    
    print(f"Processing {len(papers_to_process)} new papers...")
    
    for paper_id, pdf_path in tqdm(papers_to_process, desc="Structuring"):
        try:
            # Extract main pages to pdfs_main/ with same filename
            original_filename = Path(pdf_path).name
            extracted_pdf = pdfs_main_dir / original_filename
            
            if not extracted_pdf.exists():
                success = extract_main_pages_to_pdf(pdf_path, str(extracted_pdf), max_pages)
                if not success:
                    continue
            
            # Structure using LLM
            structure, cost = extract_paper_structure(
                str(extracted_pdf),
                model=model,
                use_upload=False
            )
            
            # Get title and year from paper_info
            info = paper_info.get(paper_id, {})
            
            structured_data[paper_id] = {
                'paper_id': paper_id,
                'title': info.get('title', 'Unknown Title'),
                'year': info.get('year'),
                'pdf_path': pdf_path,
                'extracted_pdf_path': str(extracted_pdf),
                **structure
            }
            total_cost += cost
            
            # Save incrementally
            with open(structured_file, 'w') as f:
                json.dump(structured_data, f, indent=2)
            
        except Exception as e:
            tqdm.write(f"  ✗ Error structuring {paper_id[:20]}: {e}")
    
    print(f"\n✓ Structured {len(structured_data)} papers")
    print(f"💰 Total structuring cost: ${total_cost:.4f}")
    
    return structured_data, total_cost


# ============================================================================
# Step 4: Generate Training Data
# ============================================================================

@dataclass
class TrainingExample:
    """A training example with prompt and answer."""
    tree_id: str
    prompt: str
    answer: str
    root_paper_title: str
    num_inspiring_papers: int


def format_paper_summary(paper: Dict, include_data: bool = False) -> str:
    """Format a structured paper as text.
    
    Works with both structured_data (has research_question, etc.) and 
    raw node data (has title, year from tree).
    """
    parts = []
    
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
    
    if include_data and paper.get('data_used'):
        parts.append(f"Data Used: {paper['data_used']}")
    
    if paper.get('novelty_claims'):
        parts.append(f"Novelty Claims: {paper['novelty_claims']}")
    
    return "\n".join(parts)


def format_inspiration_chain(node: Dict, structured_data: Dict) -> str:
    """Format a node's inspiration reasoning."""
    reasoning = node.get('inspiration_reasoning', {})
    if not reasoning:
        return ""
    
    parts = []
    parts.append(f"- **{node.get('title', 'Unknown')}** ({node.get('year', '?')})")
    parts.append(f"  Inspiration Type: {reasoning.get('inspiration_type', 'unknown')}")
    
    key_ideas = reasoning.get('key_ideas_borrowed', [])
    if key_ideas:
        parts.append(f"  Key Ideas: {', '.join(key_ideas)}")
    
    if reasoning.get('reasoning'):
        parts.append(f"  Reasoning: {reasoning['reasoning']}")
    
    return "\n".join(parts)


def _process_single_tree(tree_id: str, root_node: Dict, inspiring_nodes: List[Dict], structured_data: Dict[str, Dict], examples: List):
    """Helper function to process a single tree and add examples to the list."""
    # Build prompt (inspiring papers)
    prompt_parts = ["Given the following research papers that might inspire new research:\n"]

    # Group by depth
    by_depth = {}
    for node in inspiring_nodes:
        depth = node.get('depth', 1)
        if depth not in by_depth:
            by_depth[depth] = []
        by_depth[depth].append(node)

    for depth in sorted(by_depth.keys()):
        if depth == 1:
            prompt_parts.append(f"\n## Level {depth} Papers (direct inspirations):\n")
        else:
            prompt_parts.append(f"\n## Level {depth} Papers (inspired Level {depth-1} papers):\n")
        for node in by_depth[depth]:
            paper_id = node.get('paper_id')
            if paper_id in structured_data:
                prompt_parts.append(format_paper_summary(structured_data[paper_id]))
            else:
                # Fallback to node info (which has title, year from tree)
                prompt_parts.append(format_paper_summary(node))
            prompt_parts.append("")

    prompt_parts.append("\nBased on these inspiring papers, propose a novel research idea that builds upon their key contributions.")
    prompt = "\n".join(prompt_parts)

    # Build answer (reasoning + root paper)
    answer_parts = ["## Reasoning Process\n"]
    answer_parts.append("Analyzing the inspiring papers, I identified the following key connections:\n")

    for node in inspiring_nodes:
        chain = format_inspiration_chain(node, structured_data)
        if chain:
            answer_parts.append(chain)
            answer_parts.append("")

    answer_parts.append("\n## Proposed Research\n")

    root_id = root_node.get('paper_id')
    if root_id in structured_data:
        answer_parts.append(format_paper_summary(structured_data[root_id], include_data=False))
    else:
        # Fallback to node info
        answer_parts.append(format_paper_summary(root_node))

    answer = "\n".join(answer_parts)

    # Create example
    example = TrainingExample(
        tree_id=tree_id,
        prompt=prompt,
        answer=answer,
        root_paper_title=root_node.get('title', 'Unknown'),
        num_inspiring_papers=len(inspiring_nodes)
    )
    examples.append(example)


def generate_training_examples(
    trees_file: str,
    structured_data: Dict[str, Dict],
    output_dir: str
) -> List[TrainingExample]:
    """
    Generate training examples from research trees.
    
    Each example has:
    - Prompt: The inspiring papers (tree without root)
    - Answer: Reasoning process + the inspired (root) paper
    
    Args:
        trees_file: Path to the research trees JSON
        structured_data: Dictionary of structured paper summaries
        output_dir: Directory to save training data
    
    Returns:
        List of training examples
    """
    print("\n" + "="*80)
    print("STEP 4: Generating Training Data")
    print("="*80)
    
    # Load trees
    with open(trees_file, 'r') as f:
        data = json.load(f)

    trees = data.get('trees', [])  # Old format
    tree_data = data.get('tree_data', [])  # New format

    examples = []

    # Process old format (full trees)
    for tree in trees:
        nodes = tree.get('nodes', [])
        if len(nodes) < 2:
            continue  # Need at least root + 1 inspiring paper

        # Get root node
        root_node = None
        inspiring_nodes = []

        for node in nodes:
            if node.get('depth', 0) == 0:
                root_node = node
            else:
                inspiring_nodes.append(node)

        if not root_node or not inspiring_nodes:
            continue

        # Process this tree
        _process_single_tree(tree.get('tree_id', ''), root_node, inspiring_nodes, structured_data, examples)

    # Process new format (paper IDs only)
    for tree_item in tree_data:
        tree_id = tree_item.get('tree_id', '')
        root_paper_id = tree_item.get('root_paper_id', '')
        inspiring_paper_ids = tree_item.get('inspiring_paper_ids', [])

        if not root_paper_id or not inspiring_paper_ids:
            continue

        # Convert to node-like format for compatibility
        root_node = structured_data.get(root_paper_id, {'paper_id': root_paper_id, 'depth': 0})
        inspiring_nodes = []
        for pid in inspiring_paper_ids:
            node_data = structured_data.get(pid, {'paper_id': pid, 'depth': 1})
            inspiring_nodes.append(node_data)

        # Process this tree
        _process_single_tree(tree_id, root_node, inspiring_nodes, structured_data, examples)
    
    # Save training data
    output_path = Path(output_dir)
    
    # Save as JSON
    training_file = output_path / "training_data.json"
    with open(training_file, 'w') as f:
        json.dump([asdict(ex) for ex in examples], f, indent=2)
    
    # Save as JSONL (for training)
    jsonl_file = output_path / "training_data.jsonl"
    with open(jsonl_file, 'w') as f:
        for ex in examples:
            f.write(json.dumps({
                'prompt': ex.prompt,
                'completion': ex.answer,
                'metadata': {
                    'tree_id': ex.tree_id,
                    'root_title': ex.root_paper_title,
                    'num_inspiring': ex.num_inspiring_papers
                }
            }) + '\n')
    
    print(f"\n✓ Generated {len(examples)} training examples")
    print(f"✓ Saved to: {training_file}")
    print(f"✓ Saved JSONL to: {jsonl_file}")
    
    return examples


# ============================================================================
# Main Pipeline
# ============================================================================

def run_pipeline(
    trees_file: str,
    output_dir: str = "data/arxiv/test",
    model: str = "gpt-4o-mini",
    skip_download: bool = False,
    skip_structure: bool = False,
    structured_data_file: Optional[str] = None
):
    """
    Run the complete pipeline.

    Args:
        trees_file: Path to research trees JSON file (supports both full trees and paper ID format)
        output_dir: Base output directory
        model: Model to use for structuring
        skip_download: Skip download step if already done
        skip_structure: Skip structuring step if already done
        structured_data_file: Path to existing structured_papers.json (optional)
    """
    print("\n" + "#"*80)
    print("RESEARCH TREE PIPELINE")
    print("#"*80)
    print(f"Trees file: {trees_file}")
    print(f"Output directory: {output_dir}")
    print(f"Model: {model}")
    print("#"*80)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Download papers
    download_results_file = output_path / "download_results.json"
    paper_info_file = output_path / "paper_info.json"

    if skip_download and download_results_file.exists() and paper_info_file.exists():
        print("\n⏭ Skipping download (using cached results)")
        with open(download_results_file, 'r') as f:
            download_results = json.load(f)
        with open(paper_info_file, 'r') as f:
            paper_info = json.load(f)
    else:
        download_results, paper_info = download_tree_papers(trees_file, str(output_path))

    # Enrich paper info with structured data if available
    if structured_data_file:
        paper_info = enrich_paper_info(paper_info, structured_data_file)

    # Step 2-3: Extract and structure papers
    structured_file = output_path / "structured_papers.json"
    if skip_structure and structured_file.exists():
        print("\n⏭ Skipping structuring (using cached results)")
        with open(structured_file, 'r') as f:
            structured_data = json.load(f)
    else:
        structured_data = structure_papers(download_results, paper_info, str(output_path), model=model)

    # Step 4: Generate training data
    examples = generate_training_examples(trees_file, structured_data, str(output_path))
    
    # Summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"Papers downloaded: {sum(1 for v in download_results.values() if v)}/{len(download_results)}")
    print(f"Papers structured: {len(structured_data)}")
    print(f"Training examples: {len(examples)}")
    print(f"Output directory: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Research Tree Pipeline")
    parser.add_argument(
        "--trees",
        required=True,
        help="Path to research trees JSON file"
    )
    parser.add_argument(
        "--output",
        default="data/arxiv/test",
        help="Output directory"
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Model for structuring (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download step if already done"
    )
    parser.add_argument(
        "--skip-structure",
        action="store_true",
        help="Skip structuring step if already done"
    )
    parser.add_argument(
        "--structured-data",
        default=None,
        help="Path to existing structured_papers.json file for enriching paper info"
    )

    args = parser.parse_args()

    run_pipeline(
        trees_file=args.trees,
        output_dir=args.output,
        model=args.model,
        skip_download=args.skip_download,
        skip_structure=args.skip_structure,
        structured_data_file=args.structured_data
    )

