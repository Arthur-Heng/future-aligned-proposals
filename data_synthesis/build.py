#!/usr/bin/env python3
"""
Build research trees using LLMs to select inspiring candidates.

This script traces references from papers and uses LLMs to identify which
references most inspired the current paper's methodology/approach.
Unlike rule-based filtering, the LLM analyzes citation contexts and paper
abstracts to determine intellectual influence.
"""

import os
import sys
import json
import time
import re
import math
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, asdict, field
from collections import defaultdict
import random   

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.semantic_api import SemanticScholarAPI
from utils.api import call_chat_completion


@dataclass
class InspirationReasoning:
    """Stores the LLM's reasoning for why a paper was selected."""
    selected_paper_id: str
    selected_paper_title: str
    reasoning: str  # LLM's explanation of why this paper was inspiring
    inspiration_type: str  # e.g., "methodology", "problem_formulation", "theoretical_foundation"
    key_ideas_borrowed: List[str]  # Specific ideas inherited from the paper
    confidence: float  # LLM's confidence in the selection (0-1)


@dataclass
class PaperNode:
    """Represents a paper in the research tree."""
    paper_id: str
    title: str
    year: Optional[int]
    abstract: Optional[str]
    citation_count: int
    venue: Optional[str]
    depth: int = 0
    # Citation info from parent paper
    citation_contexts: List[str] = field(default_factory=list)
    citation_intents: List[str] = field(default_factory=list)
    is_influential: bool = False
    # LLM selection info
    inspiration_reasoning: Optional[InspirationReasoning] = None
    
    def to_dict(self):
        d = asdict(self)
        if self.inspiration_reasoning:
            d['inspiration_reasoning'] = asdict(self.inspiration_reasoning)
        return d


@dataclass
class ResearchTree:
    """Represents a research tree traced from a starting paper."""
    tree_id: str
    starting_paper: PaperNode
    nodes: List[PaperNode]  # All papers in the tree
    edges: List[Tuple[str, str]]  # (parent_id, child_id) pairs
    total_depth: int
    llm_cost: float  # Total LLM API cost
    
    def to_dict(self):
        return {
            'tree_id': self.tree_id,
            'starting_paper': self.starting_paper.to_dict(),
            'nodes': [n.to_dict() for n in self.nodes],
            'edges': self.edges,
            'total_depth': self.total_depth,
            'llm_cost': self.llm_cost
        }


class LLMResearchTreeBuilder:
    """Builds research trees using LLMs to select inspiring candidates."""
    
    SELECTION_PROMPT = """You are an expert at analyzing academic paper citations to identify SPECIFIC and DIRECT intellectual influence.

Given a paper and its references, identify which reference(s) provided the most SPECIFIC and DIRECT inspiration for the paper's unique contributions.

## Current Paper
Title: {current_title}
Abstract: {current_abstract}

## Candidate References (papers cited by the current paper)
{candidates_text}

## Task
Select the {num_to_select} reference(s) that DIRECTLY and SPECIFICALLY inspired the current paper's UNIQUE contributions.

IMPORTANT GUIDELINES:
- **AVOID** selecting well-known foundational papers (e.g., Transformers, BERT, GPT, ResNet, U-Net, Chain-of-Thought) unless the current paper makes a VERY SPECIFIC extension of that exact work
- PREFER selecting papers that introduced the SPECIFIC technique, formulation, or approach that the current paper directly builds upon
- Look for papers that share NICHE methodological choices, specific problem formulations, or particular technical innovations
- The best inspirations are papers where you can point to a CONCRETE technique or idea that was directly adopted/extended

For each selection, provide:
1. The reference number you selected
2. The type of inspiration (specific_technique, direct_extension, niche_methodology, problem_variant, algorithmic_basis)
3. The SPECIFIC ideas/techniques the current paper borrowed (be concrete, not generic)
4. Your confidence (0.0-1.0) - higher confidence for more specific/direct connections
5. Detailed reasoning explaining the SPECIFIC intellectual connection

## Output Format (JSON)
{{
    "selections": [
        {{
            "reference_number": 1,
            "inspiration_type": "specific_technique",
            "key_ideas_borrowed": ["specific technique X", "particular formulation Y"],
            "confidence": 0.85,
            "reasoning": "The current paper's [specific contribution] directly extends reference 1's [specific technique]..."
        }}
    ]
}}

Respond with ONLY valid JSON, no additional text."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_depth: int = 3,
        candidates_per_level: int = 10,  # Max candidates to show LLM
        selections_per_level: Union[int, List[int]] = 2,   # How many to select per level (int or list)
        model: str = "gpt-5-mini",
        rate_limit_delay: float = 1.1,
        min_citation_count: int = 5,
        max_year_gap: int = 5,           # Max years older than current paper
        recent_boost: float = 100.0      # Score for papers within 2 years (dominant factor)
    ):
        """
        Initialize the LLM-based research tree builder.
        
        Args:
            api_key: Semantic Scholar API key
            max_depth: Maximum depth to trace references
            candidates_per_level: Max references to show LLM at each level
            selections_per_level: Number of papers LLM should select per level.
                Can be int (same for all levels) or List[int] (per level, e.g., [4, 2] for 4 in L1, 2 in L2)
            model: OpenAI model to use for selection
            rate_limit_delay: Delay between Semantic Scholar API calls
            min_citation_count: Minimum citations for a paper to be considered
            max_year_gap: Maximum years older than current paper (papers older are deprioritized)
            recent_boost: Score boost for papers within 2 years of current paper
        """
        self.semantic_api = SemanticScholarAPI(api_key)
        self.max_depth = max_depth
        self.candidates_per_level = candidates_per_level
        self.selections_per_level = selections_per_level  # int or List[int]
        self.model = model
        self.rate_limit_delay = rate_limit_delay
        self.min_citation_count = min_citation_count
        self.max_year_gap = max_year_gap
        self.recent_boost = recent_boost
        
        # Tracking
        self.processed_papers: Set[str] = set()
        self.paper_cache: Dict[str, PaperNode] = {}
        self.total_llm_cost: float = 0.0
    
    def _get_selections_for_level(self, target_level: int) -> int:
        """Get the number of selections for a given target level (1-indexed).
        
        Args:
            target_level: The level we're selecting papers FOR (1 = Level 1, 2 = Level 2, etc.)
        
        Returns:
            Number of papers to select
        """
        if isinstance(self.selections_per_level, int):
            return self.selections_per_level
        else:
            # List: [selections_for_L1, selections_for_L2, ...]
            # target_level is 1-indexed, list is 0-indexed
            idx = target_level - 1
            if 0 <= idx < len(self.selections_per_level):
                return self.selections_per_level[idx] 
            elif len(self.selections_per_level) > 0:
                # Default to last value if level exceeds list
                return self.selections_per_level[-1]
            else:
                return 2  # Default fallback
    
    def _get_references_simple(self, paper_id: str, limit: int = 100) -> List[Dict]:
        """Get references using simpler endpoint to avoid 500 errors."""
        refs = self.semantic_api.get_paper_references(paper_id, limit=limit)
        # Wrap in the format expected by other methods
        return [{'citedPaper': ref, 'contexts': [], 'intents': [], 'isInfluential': False} for ref in refs]
    
    def _fetch_paper_abstract(self, paper_id: str) -> Optional[str]:
        """Fetch abstract for a single paper."""
        try:
            paper = self.semantic_api.get_paper_by_title(paper_id)  # Will search by ID if needed
            return paper.get('abstract') if paper else None
        except Exception:
            return None
    
    def _create_paper_node(
        self,
        paper_dict: Dict,
        depth: int = 0,
        contexts: List[str] = None,
        intents: List[str] = None,
        is_influential: bool = False
    ) -> PaperNode:
        """Create a PaperNode from API response."""
        return PaperNode(
            paper_id=paper_dict.get('paperId', ''),
            title=paper_dict.get('title', 'Unknown'),
            year=paper_dict.get('year'),
            abstract=paper_dict.get('abstract'),
            citation_count=paper_dict.get('citationCount', 0),
            venue=paper_dict.get('venue'),
            depth=depth,
            citation_contexts=contexts or [],
            citation_intents=intents or [],
            is_influential=is_influential
        )
    
    def _prefilter_candidates(self, references: List[Dict], current_year: Optional[int] = None) -> List[Dict]:
        """Pre-filter references before sending to LLM, prioritizing recent papers.
        
        Scoring formula:
        - Recency score (dominant): 100 * (1 - year_gap/max_year_gap) for papers within max_year_gap
          Papers within 2 years get full 100 points, older papers get proportionally less
        - Citation score (minor): log(1+citations) * 2, capped at 20 points
          This is just a tiebreaker, not a major factor
        """
        filtered = []
        
        for ref in references:
            cited_paper = ref.get('citedPaper', {})
            if not cited_paper or not cited_paper.get('paperId'):
                continue
            
            # Filter by minimum citations (handle None values)
            citation_count = cited_paper.get('citationCount') or 0
            if citation_count < self.min_citation_count:
                continue
            
            ref_year = cited_paper.get('year')
            
            # Calculate score: recency is dominant, citations are minor
            score = 0.0
            
            # Recency scoring (dominant factor - up to 100 points)
            if current_year and ref_year:
                year_gap = current_year - ref_year
                
                if year_gap <= 0:
                    # Same year or future (preprints)
                    score += self.recent_boost
                elif year_gap <= 2:
                    # Papers within 2 years get full boost
                    score += self.recent_boost
                elif year_gap <= self.max_year_gap:
                    # Linear decay from 2 years to max_year_gap
                    decay = (year_gap - 2) / (self.max_year_gap - 2)
                    score += self.recent_boost * (1 - decay)
                # Papers older than max_year_gap get 0 recency score
            
            # Citation score (minor tiebreaker - capped at 20 points)
            citation_count = cited_paper.get('citationCount', 0)
            citation_score = min(math.log1p(citation_count) * 2, 20)
            score += citation_score
            
            filtered.append((ref, score, ref_year, citation_count))
        
        # Sort by score and return top candidates
        filtered.sort(key=lambda x: x[1], reverse=True)
        return [ref for ref, _, _, _ in filtered[:self.candidates_per_level]]
    
    def _format_candidates_for_llm(self, references: List[Dict]) -> str:
        """Format reference candidates for the LLM prompt."""
        lines = []
        for i, ref in enumerate(references, 1):
            cited_paper = ref.get('citedPaper', {})
            
            title = cited_paper.get('title', 'Unknown')
            year = cited_paper.get('year', 'N/A')
            venue = cited_paper.get('venue', '')
            venue_str = f" | {venue}" if venue else ""
            
            lines.append(
                f"{i}. {title} ({year}){venue_str}"
            )
        
        return '\n'.join(lines)
    
    def _parse_llm_response(self, response: str, references: List[Dict]) -> List[Tuple[Dict, InspirationReasoning]]:
        """Parse LLM response and extract selections with reasoning."""
        try:
            # Clean response - remove markdown code blocks if present
            response = response.strip()
            if response.startswith('```'):
                response = re.sub(r'^```(?:json)?\n?', '', response)
                response = re.sub(r'\n?```$', '', response)
            
            data = json.loads(response)
            selections = data.get('selections', [])
            
            results = []
            for sel in selections:
                ref_num = sel.get('reference_number', 0) - 1  # Convert to 0-indexed
                if 0 <= ref_num < len(references):
                    ref = references[ref_num]
                    cited_paper = ref.get('citedPaper', {})
                    
                    reasoning = InspirationReasoning(
                        selected_paper_id=cited_paper.get('paperId', ''),
                        selected_paper_title=cited_paper.get('title', 'Unknown'),
                        reasoning=sel.get('reasoning', ''),
                        inspiration_type=sel.get('inspiration_type', 'unknown'),
                        key_ideas_borrowed=sel.get('key_ideas_borrowed', []),
                        confidence=sel.get('confidence', 0.5)
                    )
                    results.append((ref, reasoning))
            
            return results
            
        except json.JSONDecodeError as e:
            print(f"  ⚠ Failed to parse LLM response: {e}")
            print(f"  Response was: {response[:500]}...")
            return []
    
    def select_inspiring_references(
        self,
        current_paper: PaperNode,
        references: List[Dict],
        target_level: int = 1
    ) -> List[Tuple[PaperNode, InspirationReasoning]]:
        """
        Use LLM to select the most inspiring references.
        
        Args:
            current_paper: The paper we're finding inspirations for
            references: List of reference dictionaries
            target_level: The level we're selecting papers FOR (1 = Level 1, 2 = Level 2, etc.)
        
        Returns:
            List of (PaperNode, InspirationReasoning) tuples
        """
        if not references:
            return []
        
        # Pre-filter candidates (pass current paper's year for recency scoring)
        candidates = self._prefilter_candidates(references, current_year=current_paper.year)
        if not candidates:
            print(f"  No candidates passed pre-filtering")
            return []
        
        # Get number of selections for this level
        num_selections = self._get_selections_for_level(target_level)
        print(f"  Asking LLM to select {num_selections} from {len(candidates)} candidates (for Level {target_level})...")
        
        # Format prompt
        candidates_text = self._format_candidates_for_llm(candidates)
        prompt = self.SELECTION_PROMPT.format(
            current_title=current_paper.title,
            current_abstract=current_paper.abstract or "No abstract available.",
            candidates_text=candidates_text,
            num_to_select=min(num_selections, len(candidates))
        )
        
        # Call LLM
        messages = [{"role": "user", "content": prompt}]
        try:
            response, cost = call_chat_completion(
                messages=messages,
                model=self.model,
                temperature=0.3  # Lower temperature for more consistent selection
            )
            self.total_llm_cost += cost
            print(f"  LLM cost: ${cost:.4f}")
        except Exception as e:
            print(f"  ⚠ LLM call failed: {e}")
            return []
        
        # Parse response
        selections = self._parse_llm_response(response, candidates)
        
        # Convert to PaperNodes
        results = []
        for ref, reasoning in selections:
            cited_paper = ref.get('citedPaper', {})
            node = self._create_paper_node(
                cited_paper,
                depth=current_paper.depth + 1,
                contexts=ref.get('contexts', []),
                intents=ref.get('intents', []),
                is_influential=ref.get('isInfluential', False)
            )
            node.inspiration_reasoning = reasoning
            results.append((node, reasoning))
        
        return results
    
    def build_tree_from_paper(
        self,
        paper_title: str
    ) -> Optional[ResearchTree]:
        """
        Build a research tree starting from a given paper title.
        
        Args:
            paper_title: Title of the starting paper
        
        Returns:
            ResearchTree object or None if failed
        """
        print(f"\n{'='*80}")
        print(f"Building research tree from: {paper_title}")
        print(f"{'='*80}")
        
        # Get starting paper
        paper_data = self.semantic_api.get_paper_by_title(paper_title)
        if not paper_data:
            print(f"✗ Could not find paper: {paper_title}")
            return None
        
        starting_paper = self._create_paper_node(paper_data, depth=0)
        self.paper_cache[starting_paper.paper_id] = starting_paper
        
        print(f"✓ Found: {starting_paper.title} ({starting_paper.year})")
        print(f"  Citations: {starting_paper.citation_count}")
        
        # Tree structure
        all_nodes = [starting_paper]
        edges = []
        
        # BFS through reference tree
        current_level = [starting_paper]
        
        for depth in range(self.max_depth):
            if not current_level:
                break
            
            print(f"\n--- Depth {depth + 1}/{self.max_depth} ---")
            next_level = []
            
            for current_paper in current_level:
                if current_paper.paper_id in self.processed_papers:
                    continue
                self.processed_papers.add(current_paper.paper_id)
                
                print(f"\nProcessing: {current_paper.title[:60]}...")
                
                # Get references using simple endpoint (avoids 500 errors)
                time.sleep(self.rate_limit_delay)
                references = self._get_references_simple(current_paper.paper_id, limit=50)
                
                if not references:
                    print(f"  No references found")
                    continue
                
                print(f"  Found {len(references)} references")
                
                # Use LLM to select inspiring references
                # depth+1 because we're selecting papers for the next level
                selections = self.select_inspiring_references(current_paper, references, target_level=depth + 1)
                
                if not selections:
                    print(f"  No inspiring references selected")
                    continue
                
                print(f"  Selected {len(selections)} inspiring references:")
                
                for node, reasoning in selections:
                    # Add to tree
                    all_nodes.append(node)
                    edges.append((current_paper.paper_id, node.paper_id))
                    self.paper_cache[node.paper_id] = node
                    
                    # Add to next level for further exploration
                    next_level.append(node)
                    
                    # Print selection info
                    print(f"    ★ [{node.year}] {node.title[:50]}...")
                    print(f"      Type: {reasoning.inspiration_type} | Confidence: {reasoning.confidence:.2f}")
                    print(f"      Key ideas: {', '.join(reasoning.key_ideas_borrowed[:2])}")
            
            current_level = next_level
        
        # Create tree
        tree = ResearchTree(
            tree_id=f"tree_{starting_paper.paper_id}",
            starting_paper=starting_paper,
            nodes=all_nodes,
            edges=edges,
            total_depth=max(n.depth for n in all_nodes),
            llm_cost=self.total_llm_cost
        )
        
        print(f"\n✓ Research tree built: {len(all_nodes)} nodes, {len(edges)} edges")
        print(f"  Total LLM cost: ${self.total_llm_cost:.4f}")
        
        return tree
    
    def build_trees_from_papers(
        self,
        paper_titles: List[str],
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval: int = 100
    ) -> List[ResearchTree]:
        """Build research trees from multiple starting papers.
        
        Args:
            paper_titles: List of paper titles to process
            checkpoint_dir: Directory to save checkpoints (enables resume)
            checkpoint_interval: Save checkpoint every N papers
        
        Returns:
            List of ResearchTree objects
        """
        trees = []
        processed_titles = set()
        start_idx = 0
        
        # Load existing checkpoint if available
        if checkpoint_dir:
            checkpoint_file = os.path.join(checkpoint_dir, "trees_checkpoint.json")
            if os.path.exists(checkpoint_file):
                print(f"\n📂 Found checkpoint: {checkpoint_file}")
                with open(checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)
                
                # Load existing trees
                for tree_dict in checkpoint_data.get('trees', []):
                    tree = ResearchTree.from_dict(tree_dict)
                    trees.append(tree)
                    # Track processed titles
                    for node in tree.nodes:
                        if node.depth == 0:
                            processed_titles.add(node.title.upper())
                
                print(f"✓ Loaded {len(trees)} trees from checkpoint")
                print(f"✓ Resuming from paper {len(processed_titles) + 1}")
        
        print(f"\n{'#'*80}")
        print(f"Building research trees from {len(paper_titles)} papers")
        print(f"Configuration:")
        print(f"  - Max depth: {self.max_depth}")
        print(f"  - Candidates per level: {self.candidates_per_level}")
        if isinstance(self.selections_per_level, list):
            sel_str = ', '.join(f"L{i+1}:{n}" for i, n in enumerate(self.selections_per_level))
            print(f"  - Selections per level: [{sel_str}]")
        else:
            print(f"  - Selections per level: {self.selections_per_level}")
        print(f"  - Model: {self.model}")
        if checkpoint_dir:
            print(f"  - Checkpoint dir: {checkpoint_dir}")
            print(f"  - Checkpoint interval: every {checkpoint_interval} papers")
        print(f"{'#'*80}")
        
        new_trees_count = 0
        for i, title in enumerate(paper_titles, 1):
            # Skip already processed titles
            if title.upper() in processed_titles:
                continue
            
            print(f"\n[{i}/{len(paper_titles)}] Processing: {title}")
            
            try:
                tree = self.build_tree_from_paper(title)
                if tree:
                    trees.append(tree)
                    new_trees_count += 1
            except Exception as e:
                print(f"⚠ Error processing '{title}': {e}")
                continue
            
            # Save checkpoint periodically
            if checkpoint_dir and new_trees_count > 0 and new_trees_count % checkpoint_interval == 0:
                self._save_checkpoint(trees, checkpoint_dir)
            
            time.sleep(self.rate_limit_delay)
        
        # Save final checkpoint
        if checkpoint_dir and new_trees_count > 0:
            self._save_checkpoint(trees, checkpoint_dir)
        
        return trees
    
    def _save_checkpoint(self, trees: List['ResearchTree'], checkpoint_dir: str):
        """Save checkpoint of current progress."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_file = os.path.join(checkpoint_dir, "trees_checkpoint.json")
        
        data = {
            'config': {
                'max_depth': self.max_depth,
                'candidates_per_level': self.candidates_per_level,
                'selections_per_level': self.selections_per_level,
                'model': self.model,
            },
            'trees': [tree.to_dict() for tree in trees],
            'total_trees': len(trees),
        }
        
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Checkpoint saved: {len(trees)} trees → {checkpoint_file}")
    
    def save_trees(self, trees: List[ResearchTree], output_file: str):
        """Save research trees to JSON file."""
        data = {
            'config': {
                'max_depth': self.max_depth,
                'candidates_per_level': self.candidates_per_level,
                'selections_per_level': self.selections_per_level,
                'model': self.model,
                'min_citation_count': self.min_citation_count
            },
            'trees': [tree.to_dict() for tree in trees],
            'total_trees': len(trees),
            'total_nodes': sum(len(tree.nodes) for tree in trees),
            'total_llm_cost': sum(tree.llm_cost for tree in trees)
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Saved {len(trees)} trees to: {output_file}")
    
    def print_tree_summary(self, tree: ResearchTree):
        """Print a visual summary of a research tree."""
        print(f"\n{'='*80}")
        print(f"RESEARCH TREE: {tree.starting_paper.title[:60]}...")
        print(f"{'='*80}")
        
        # Group nodes by depth
        by_depth = defaultdict(list)
        for node in tree.nodes:
            by_depth[node.depth].append(node)
        
        for depth in sorted(by_depth.keys()):
            nodes = by_depth[depth]
            print(f"\nDepth {depth}:")
            for node in nodes:
                prefix = "  " * depth + ("└─ " if depth > 0 else "● ")
                print(f"{prefix}[{node.year}] {node.title[:50]}...")
                
                if node.inspiration_reasoning:
                    r = node.inspiration_reasoning
                    print(f"{'  ' * depth}   └─ {r.inspiration_type}: {r.reasoning[:80]}...")
        
        print(f"\nTotal: {len(tree.nodes)} nodes | Depth: {tree.total_depth} | Cost: ${tree.llm_cost:.4f}")


def load_api_key():
    """Load Semantic Scholar API key from ~/.zshrc."""
    import subprocess
    result = subprocess.run(
        ['zsh', '-c', 'source ~/.zshrc && echo $SEMANTIC_SCHOLAR_API_KEY'],
        capture_output=True, text=True
    )
    api_key = result.stdout.strip()
    
    if api_key:
        print(f"✓ Semantic Scholar API key loaded")
    else:
        print("⚠ No Semantic Scholar API key found")
    
    return api_key


def load_neurips_2025_papers(filepath: str, num_papers: int = 5) -> List[str]:
    """Load paper titles from NeurIPS 2025 JSONL file. (randomly sample num_papers papers)"""
    papers = []
    with open(filepath, 'r') as f:
        for line in f:
            data = json.loads(line)
            papers.append(data['title'])
    return random.sample(papers, min(num_papers, len(papers)))


def test_with_examples(num_papers: int = 3):
    """Test the builder with NeurIPS 2025 papers and report costs."""
    
    print("\n" + "="*80)
    print("TESTING LLM-BASED RESEARCH TREE BUILDER")
    print("="*80)
    
    # Load API key
    api_key = load_api_key()
    
    # Load NeurIPS 2025 papers
    neurips_file = "data/accepted_papers/NeurIPS.cc_2025.json"
    test_papers = load_neurips_2025_papers(neurips_file, num_papers=num_papers)
    
    print(f"\nTest papers (NeurIPS 2025):")
    for i, title in enumerate(test_papers, 1):
        print(f"  {i}. {title}")
    
    # Initialize builder with settings prioritizing recent papers
    builder = LLMResearchTreeBuilder(
        api_key=api_key,
        max_depth=2,              # Shallow for testing
        candidates_per_level=10,   # Fewer candidates
        selections_per_level=[4, 2],  # 4 for Level 1, 2 for Level 2
        model="gpt-5-mini",       # Better model for quality selection
        rate_limit_delay=1.1,
        min_citation_count=5,     # Lower threshold to include more recent papers
        max_year_gap=5,           # Deprioritize papers older than 5 years
        recent_boost=100.0        # Recency is dominant factor (100 pts vs max 20 for citations)
    )
    
    # Build trees
    start_time = time.time()
    trees = builder.build_trees_from_papers(test_papers)
    elapsed = time.time() - start_time
    
    # Print summaries
    for tree in trees:
        builder.print_tree_summary(tree)
    
    # Save results to test_data directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = "synthesize/test_data"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/neurips2025_trees_{num_papers}_{timestamp}.json"
    builder.save_trees(trees, output_file)
    
    # Cost report
    print("\n" + "="*80)
    print("COST REPORT")
    print("="*80)
    total_cost = sum(tree.llm_cost for tree in trees)
    total_nodes = sum(len(tree.nodes) for tree in trees)
    
    print(f"Papers processed: {len(test_papers)}")
    print(f"Trees built: {len(trees)}")
    print(f"Total nodes: {total_nodes}")
    print(f"Total LLM cost: ${total_cost:.4f}")
    print(f"Average cost per tree: ${total_cost/len(trees):.4f}" if trees else "N/A")
    print(f"Average cost per node: ${total_cost/total_nodes:.4f}" if total_nodes else "N/A")
    print(f"Elapsed time: {elapsed:.1f}s")
    print("="*80)
    
    return trees, total_cost, output_file


def run_full_pipeline(num_papers: int = 20):
    """Run the complete pipeline: build trees, download, structure, generate training data."""
    from synthesize.pipeline import run_pipeline
    
    # Step 1: Build research trees
    trees, cost, trees_file = test_with_examples(num_papers=num_papers)
    
    if not trees:
        print("No trees built, aborting pipeline")
        return
    
    # Step 2-4: Run the rest of the pipeline
    output_dir = "data/arxiv/test"
    run_pipeline(
        trees_file=trees_file,
        output_dir=output_dir,
        model="gpt-4o-mini",
        skip_download=False,
        skip_structure=False
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build research trees with LLM selection")
    parser.add_argument(
        "--num-papers",
        type=int,
        default=3,
        help="Number of papers to process (default: 3)"
    )
    parser.add_argument(
        "--full-pipeline",
        action="store_true",
        help="Run full pipeline including download, structure, and training data generation"
    )
    
    args = parser.parse_args()
    
    if args.full_pipeline:
        run_full_pipeline(num_papers=args.num_papers)
    else:
        test_with_examples(num_papers=args.num_papers)

