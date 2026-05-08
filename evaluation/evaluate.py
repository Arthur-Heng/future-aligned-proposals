#!/usr/bin/env python3
"""
Evaluation: Measure how well generated proposals match real papers.

Pipeline:
1. Load generated proposals from baseline predictions
2. Use retriever (BM25 or embeddings) to find top-k similar papers from corpus
3. Use LLM-as-judge to score semantic similarity
4. Compute metrics: max score, true root rank, recall@k
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
from loguru import logger

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.api import call_chat_completion, call_batch_embedding


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class SubfieldScores:
    """Subfield similarity scores."""
    research_question: float = 0.0
    hypothesis: float = 0.0
    proposed_method: float = 0.0
    novelty_claims: float = 0.0
    experiment_details: float = 0.0
    overall: float = 0.0


@dataclass
class GapGroundednessResult:
    """Gap groundedness evaluation result."""
    overall_groundedness: float = 0.0
    num_gaps_identified: int = 0
    reasoning: str = ""

@dataclass
class EvaluationResult:
    """Result for a single prediction."""
    prediction_id: int
    root_title: str
    # Retrieval results
    top_k_titles: List[str]
    top_k_scores: List[float]
    true_root_rank: int  # -1 if not in top-k
    true_root_in_top_k: bool
    # LLM judge scores
    llm_similarity_scores: List[float]
    max_llm_score: float
    true_root_llm_score: float
    avg_llm_score: float = 0.0  # Average across all judged papers
    # Subfield scores (optional)
    subfield_scores: Optional[List[SubfieldScores]] = None
    max_subfield_scores: Optional[SubfieldScores] = None
    avg_subfield_scores: Optional[SubfieldScores] = None  # Average across retrieved papers
    true_root_subfield_scores: Optional[SubfieldScores] = None
    # Gap groundedness (optional)
    gap_groundedness: Optional[GapGroundednessResult] = None
    # Cost tracking
    cost_usd: float = 0.0


# ============================================================================
# Corpus Loading
# ============================================================================

def load_corpus(corpus_paths: Union[str, List[str]], titles_path: Optional[Union[str, List[str]]] = None) -> List[Dict]:
    """
    Load paper corpus from JSON or JSONL files.

    Supports:
    - JSON array: [{"id": ..., "research_question": ..., ...}, ...]
    - JSON object: {"<paper_id>": {...}, "<paper_id2>": {...}, ...}
    - JSONL: one JSON object per line

    Args:
        corpus_paths: Path(s) to corpus file(s) - can be single string or list of strings
        titles_path: Optional path(s) to JSONL with titles (id -> title mapping)
    """
    # Handle single path (backward compatibility)
    if isinstance(corpus_paths, str):
        corpus_paths = [corpus_paths]

    all_papers = []
    for corpus_path in corpus_paths:
        logger.info(f"Loading corpus from {corpus_path}...")
        with open(corpus_path, 'r') as f:
            content = f.read().strip()

        if content.startswith('['):
            # JSON array
            papers = json.loads(content)
        elif content.startswith('{'):
            # JSON object: {paper_id -> record}. Use values as the paper list.
            obj = json.loads(content)
            papers = list(obj.values())
        else:
            # JSONL format
            papers = []
            for line in content.split('\n'):
                line = line.strip()
                if line:
                    papers.append(json.loads(line))

        all_papers.extend(papers)
        logger.info(f"  Loaded {len(papers)} papers from {corpus_path}")

    logger.info(f"Total corpus size: {len(all_papers)} papers from {len(corpus_paths)} files")

    # If titles_path provided, merge titles into corpus
    if titles_path:
        # Handle single path (backward compatibility)
        if isinstance(titles_path, str):
            titles_paths = [titles_path]
        else:
            titles_paths = titles_path
        
        id_to_title = {}
        for tp in titles_paths:
            if not os.path.exists(tp):
                logger.warning(f"Titles file not found: {tp}")
                continue
            logger.info(f"Loading titles from {tp}...")
            with open(tp, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        paper = json.loads(line)
                        if paper.get('id') and paper.get('title'):
                            id_to_title[paper['id']] = paper['title']
        
        logger.info(f"Loaded {len(id_to_title)} titles from {len(titles_paths)} files")

        # Merge titles into all papers
        titles_added = 0
        for paper in all_papers:
            paper_id = paper.get('id')
            if paper_id and paper_id in id_to_title:
                paper['title'] = id_to_title[paper_id]
                titles_added += 1

        logger.info(f"Added titles to {titles_added}/{len(all_papers)} papers")

    return all_papers


def load_predictions(predictions_path: str) -> List[Dict]:
    """Load baseline predictions."""
    with open(predictions_path, 'r') as f:
        data = json.load(f)
    return data.get('predictions', [])


def strip_reasoning_from_proposal(text: str, remove_repetition: bool = False) -> str:
    """
    Extract only the proposal content, removing reasoning process.

    Handles three formats:
    1. Full CoT: reasoning block then proposal block (## Proposed Research)
    2. Step-wise CoT: interleaved reasoning (### Step N) and proposal sections
    3. Direct: no reasoning to strip
    4. Corrupted outputs that echo the chat template (strips content before 'assistant' marker)
    
    Args:
        text: The raw prediction text
        remove_repetition: If True, also remove duplicate sentences (helps with
                          verbose model outputs that repeat the same phrases)
    """
    import re

    if not text:
        return text

    # ------------------------------------------------------------------
    # Handle corrupted outputs that echo the chat template
    # These start with format instructions and include "user\n...assistant\n"
    # ------------------------------------------------------------------
    if 'for reasoning sections' in text[:500] or '\nuser\n' in text[:600]:
        # Find the 'assistant' marker and take content after it
        assistant_match = re.search(r'\nassistant\s*\n', text)
        if assistant_match:
            text = text[assistant_match.end():]

    # Helper: remove step-wise CoT reasoning blocks (### Step N: ...)
    # Removes each block from its header through all following lines
    # until the next ## header, non-Step ### header, or proposal field label.
    # Note: Section headers can be with or without colons
    def _strip_step_blocks(t):
        return re.sub(
            r'###\s*Step\s*\d+[^\n]*\n'
            r'(?:(?!##\s|###\s*(?!Step)|Research Question[:\s]|Hypothesis[:\s]|Proposed Method[:\s\+]|Novelty Claims?[:\s]|Experiment Details?[:\s]).*\n)*',
            '', t
        )
    
    # Helper: remove repetitive sentences (common in verbose model outputs)
    def _remove_repetition(t):
        """Remove duplicate sentences that appear multiple times."""
        # Split into sentences (simple split on period followed by space or newline)
        sentences = re.split(r'(?<=[.!?])\s+', t)
        seen = set()
        unique_sentences = []
        for s in sentences:
            # Normalize for comparison (strip, lowercase)
            normalized = s.strip().lower()
            # Skip very short "sentences" (likely fragments)
            if len(normalized) < 20:
                unique_sentences.append(s)
                continue
            if normalized not in seen:
                seen.add(normalized)
                unique_sentences.append(s)
        return ' '.join(unique_sentences)
    
    # Helper to optionally apply repetition removal
    def _finalize(t):
        return _remove_repetition(t) if remove_repetition else t

    # ------------------------------------------------------------------
    # Step-wise CoT detection: look for "### Step 1/2/3" markers
    # ------------------------------------------------------------------
    step_markers = re.findall(r'###\s*Step\s*\d', text)
    if len(step_markers) >= 2:
        cleaned = _strip_step_blocks(text)
        # Strip leading text before first proposal header
        proposal_start = re.search(
            r'(##\s*Proposed Research|##\s*Research Question|'
            r'##\s*Proposed Method|##\s*Novel Research|'
            r'##\s*Proposal|\*\*Research Question)',
            cleaned, re.IGNORECASE
        )
        if proposal_start:
            cleaned = cleaned[proposal_start.start():]
        cleaned = cleaned.strip()
        if cleaned:
            return _finalize(cleaned)

    # ------------------------------------------------------------------
    # Full CoT: find proposal section header, take everything after
    # ------------------------------------------------------------------
    proposal_patterns = [
        r'##\s*Proposed Research\s*(?:Idea)?',
        r'##\s*Research Proposal',
        r'##\s*Novel Research\s*(?:Idea)?',
        r'##\s*Proposal',
        r'\*\*Proposed Research\s*(?:Idea)?\*\*',
        r'\*\*Research Proposal\*\*',
        r'\*\*Novel Research\s*(?:Idea)?\*\*',
    ]

    for pattern in proposal_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            proposal_text = text[match.end():].strip()
            proposal_text = re.sub(r'^[\s\n#]*', '', proposal_text)
            # Also remove any remaining step reasoning blocks (mixed format)
            proposal_text = _strip_step_blocks(proposal_text)
            return _finalize(proposal_text.strip())

    # Fallback: try removing reasoning section
    reasoning_patterns = [
        r'^##\s*Reasoning\s*(?:Process)?\s*\n.*?(?=##|\*\*[A-Z])',
        r'^\*\*Reasoning\s*(?:Process)?\*\*\s*\n.*?(?=##|\*\*[A-Z])',
    ]

    for pattern in reasoning_patterns:
        cleaned = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
        if cleaned != text:
            return _finalize(cleaned.strip())

    # If no patterns matched, return original text (with optional dedup)
    return _finalize(text)


REQUIRED_SECTIONS = ['research_question', 'hypothesis', 'proposed_method', 'novelty_claims', 'experiment_details']

# Section patterns - flexible to handle with/without colons, and combined headers
SECTION_PATTERNS = {
    'research_question': r'(?:Research Question|RQ)(?:\s*[:+\s]|\s*\n)',
    'hypothesis': r'Hypothesis(?:\s*[:+\s]|\s*\n)',
    'proposed_method': r'Proposed Method(?:\s*[:+\s]|\s*\n)',
    'novelty_claims': r'Novelty(?: Claims?)?(?:\s*[:+\s]|\s*\n)',
    'experiment_details': r'Experiment(?: Details?)?(?:\s*[:+\s]|\s*\n)',
}


def check_proposal_sections(text: str) -> Dict[str, bool]:
    """
    Check which sections are present in a proposal.
    
    Returns:
        Dict mapping section name to boolean (True if present)
    """
    import re
    result = {}
    for section, pattern in SECTION_PATTERNS.items():
        result[section] = bool(re.search(pattern, text, re.IGNORECASE))
    return result


def has_all_sections(text: str) -> bool:
    """
    Check if a proposal has all required sections.
    
    Required sections: Research Question, Hypothesis, Proposed Method, 
                      Novelty Claims, Experiment Details
    """
    sections = check_proposal_sections(text)
    return all(sections.values())


def extract_sections_for_retrieval(text: str, sections: List[str]) -> str:
    """
    Extract specific sections from a proposal for retrieval.
    
    This helps with long proposals where verbose sections (like Experiment Details)
    can dilute the BM25 signal.
    
    Args:
        text: The proposal text
        sections: List of section names to extract. Valid values:
            - "research_question"
            - "hypothesis" 
            - "proposed_method"
            - "novelty_claims"
            - "experiment_details"
    
    Returns:
        Concatenated text from the specified sections
    """
    import re
    
    if not text or not sections:
        return text
    
    # Patterns for each section (case-insensitive)
    section_patterns = {
        'research_question': r'(?:Research Question|RQ)[:\s]*\n?(.*?)(?=\n(?:Hypothesis|Proposed Method|Novelty|Experiment|$)|\Z)',
        'hypothesis': r'Hypothesis[:\s]*\n?(.*?)(?=\n(?:Proposed Method|Novelty|Experiment|$)|\Z)',
        'proposed_method': r'Proposed Method[:\s]*\n?(.*?)(?=\n(?:Novelty|Experiment|$)|\Z)',
        'novelty_claims': r'Novelty(?: Claims?)?[:\s]*\n?(.*?)(?=\n(?:Experiment|$)|\Z)',
        'experiment_details': r'Experiment(?: Details?)?[:\s]*\n?(.*?)(?=\n(?:$)|\Z)',
    }
    
    extracted_parts = []
    
    for section in sections:
        section_key = section.lower().replace(' ', '_').replace('-', '_')
        if section_key in section_patterns:
            pattern = section_patterns[section_key]
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                content = match.group(1).strip()
                if content:
                    extracted_parts.append(content)
    
    # If no sections found, return original text truncated
    if not extracted_parts:
        return text
    
    return '\n\n'.join(extracted_parts)


# ============================================================================
# Retriever: BM25
# ============================================================================

class BM25Retriever:
    """BM25-based retriever for finding similar papers."""
    
    def __init__(self, corpus: List[Dict], text_field: str = "structured"):
        """
        Initialize BM25 retriever.
        
        Args:
            corpus: List of paper dictionaries
            text_field: Field to use for retrieval:
                - "title": title only
                - "structured": all structured fields (research_question, proposed_method, etc.)
                - "both": title + abstract
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("Please install rank_bm25: pip install rank-bm25")
        
        self.corpus = corpus
        self.text_field = text_field
        
        # Build document texts
        self.doc_texts = []
        for paper in corpus:
            if text_field == "structured":
                # Use all structured fields for richer matching
                parts = []
                for field in ['research_question', 'hypothesis', 'proposed_method', 'novelty_claims']:
                    if paper.get(field):
                        parts.append(paper[field])
                text = ' '.join(parts) if parts else paper.get('title', '')
            elif text_field == "both":
                text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
            else:
                text = paper.get(text_field, paper.get('title', ''))
            self.doc_texts.append(text)
        
        # Tokenize and build BM25 index
        tokenized_docs = [doc.lower().split() for doc in self.doc_texts]
        self.bm25 = BM25Okapi(tokenized_docs)
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[Dict, float]]:
        """
        Retrieve top-k most similar papers.
        
        Returns:
            List of (paper, score) tuples
        """
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((self.corpus[idx], scores[idx]))
        
        return results


# ============================================================================
# Retriever: Embedding-based
# ============================================================================

# Supported embedding models
EMBEDDING_MODELS = {
    # OpenAI API models
    "text-embedding-3-large": {"type": "openai", "dim": 3072},
    "text-embedding-3-small": {"type": "openai", "dim": 1536},
    "text-embedding-ada-002": {"type": "openai", "dim": 1536},
    # Local SOTA models (sentence-transformers)
    "bge-large-en-v1.5": {"type": "local", "hf_name": "BAAI/bge-large-en-v1.5", "dim": 1024},
    "bge-base-en-v1.5": {"type": "local", "hf_name": "BAAI/bge-base-en-v1.5", "dim": 768},
    "gte-large-en-v1.5": {"type": "local", "hf_name": "Alibaba-NLP/gte-large-en-v1.5", "dim": 1024},
    "e5-large-v2": {"type": "local", "hf_name": "intfloat/e5-large-v2", "dim": 1024},
    "specter2": {"type": "local", "hf_name": "allenai/specter2", "dim": 768},  # Best for scientific papers
}

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"


class EmbeddingRetriever:
    """Embedding-based retriever using cosine similarity.
    
    Supports both OpenAI API models and local sentence-transformers models.
    Local models are free and can be SOTA quality (e.g., BGE, GTE, SPECTER2).
    """
    
    def __init__(
        self,
        corpus: List[Dict],
        text_field: str = "structured",
        model: str = DEFAULT_EMBEDDING_MODEL,
        cache_path: Optional[str] = None
    ):
        """
        Initialize embedding retriever.
        
        Args:
            corpus: List of paper dictionaries
            text_field: Field to use for embedding (title, structured, both)
            model: Embedding model name. Supported models:
                   - OpenAI: "text-embedding-3-large" (default), "text-embedding-3-small"
                   - Local SOTA: "bge-large-en-v1.5", "gte-large-en-v1.5", "specter2"
            cache_path: Path to cache embeddings
        """
        self.corpus = corpus
        self.text_field = text_field
        self.model = model
        self.cache_path = cache_path
        
        # Determine model type
        if model in EMBEDDING_MODELS:
            self.model_info = EMBEDDING_MODELS[model]
        elif model.startswith("text-embedding"):
            self.model_info = {"type": "openai"}
        else:
            # Assume it's a HuggingFace model path
            self.model_info = {"type": "local", "hf_name": model}
        
        # Load local model if needed
        self.local_model = None
        if self.model_info["type"] == "local":
            self._load_local_model()
        
        # Build document texts
        self.doc_texts = []
        for paper in corpus:
            if text_field == "structured":
                parts = []
                for field in ['research_question', 'hypothesis', 'proposed_method', 'novelty_claims']:
                    if paper.get(field):
                        parts.append(paper[field])
                text = ' '.join(parts) if parts else paper.get('title', '')
            elif text_field == "both":
                text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
            else:
                text = paper.get(text_field, paper.get('title', ''))
            # Ensure non-empty text (OpenAI API rejects empty strings)
            if not text or not text.strip():
                text = paper.get('title') or paper.get('id', 'untitled')
            self.doc_texts.append(text)
        
        # Load or compute embeddings
        self.embeddings = self._load_or_compute_embeddings()
    
    def _load_local_model(self):
        """Load a local sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for local embedding models. "
                "Install with: pip install sentence-transformers"
            )
        
        hf_name = self.model_info.get("hf_name", self.model)
        logger.info(f"Loading local embedding model: {hf_name}")
        self.local_model = SentenceTransformer(hf_name, trust_remote_code=True)
        logger.info(f"Loaded {hf_name} (dim={self.local_model.get_sentence_embedding_dimension()})")
    
    def _embed_texts_local(self, texts: List[str]) -> np.ndarray:
        """Embed texts using local model."""
        # Some models like E5 need special prefixes
        if "e5-" in self.model.lower():
            texts = [f"query: {t}" for t in texts]
        elif "bge-" in self.model.lower():
            # BGE models work better with instruction prefix for queries
            pass  # No prefix needed for documents
        
        embeddings = self.local_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )
        return embeddings
    
    def _embed_texts_openai(self, texts: List[str]) -> np.ndarray:
        """Embed texts using OpenAI API."""
        batch_size = 100
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding (OpenAI)"):
            batch = texts[i:i+batch_size]
            embeddings, _ = call_batch_embedding(batch, model=self.model)
            all_embeddings.extend(embeddings)
        
        return np.array(all_embeddings)
    
    def _load_or_compute_embeddings(self) -> np.ndarray:
        """Load cached embeddings or compute new ones."""
        if self.cache_path and os.path.exists(self.cache_path):
            logger.info(f"Loading cached embeddings from {self.cache_path}")
            return np.load(self.cache_path)
        
        logger.info(f"Computing embeddings for {len(self.doc_texts)} documents using {self.model}...")
        
        if self.model_info["type"] == "local":
            embeddings_array = self._embed_texts_local(self.doc_texts)
        else:
            embeddings_array = self._embed_texts_openai(self.doc_texts)
        
        # Cache if path provided
        if self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            np.save(self.cache_path, embeddings_array)
            logger.info(f"Cached embeddings to {self.cache_path}")
        
        return embeddings_array
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[Dict, float]]:
        """
        Retrieve top-k most similar papers using cosine similarity.
        """
        # Embed query
        if self.model_info["type"] == "local":
            # Add query prefix for certain models
            query_text = query
            if "e5-" in self.model.lower():
                query_text = f"query: {query}"
            query_emb = self.local_model.encode(
                [query_text], 
                convert_to_numpy=True,
                normalize_embeddings=True
            )[0]
        else:
            query_emb, _ = call_batch_embedding([query], model=self.model)
            query_emb = np.array(query_emb[0])
        
        # Compute cosine similarities
        # If embeddings are normalized, dot product = cosine similarity
        if self.model_info["type"] == "local":
            similarities = np.dot(self.embeddings, query_emb)
        else:
            similarities = np.dot(self.embeddings, query_emb) / (
                np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_emb)
            )
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((self.corpus[idx], similarities[idx]))
        
        return results


# ============================================================================
# LLM-as-Judge
# ============================================================================

LLM_JUDGE_PROMPT = """You are an expert research paper evaluator. Your task is to score the semantic similarity between a generated research proposal and an existing paper.

## Generated Proposal:
{proposal}

## Candidate Paper:
{candidate_summary}

## Task:
Rate the semantic similarity on a scale of 0-10:
- 0: Completely unrelated topics/methods
- 3: Same broad area but different specific focus
- 5: Related topic with some overlapping ideas
- 7: Very similar research direction and methods
- 10: Nearly identical research idea

Consider:
1. Research question similarity
2. Methodology overlap
3. Technical approach similarity
4. Problem formulation alignment

Respond with ONLY a JSON object: {{"score": <number>, "reasoning": "<brief explanation>"}}"""


LLM_JUDGE_SUBFIELD_PROMPT = """You are an expert research paper evaluator. Your task is to score the semantic similarity between a generated research proposal and an existing paper, both overall and for specific subfields.

## Generated Proposal:
{proposal}

## Candidate Paper:
{candidate_summary}

## Task:
Rate the similarity on a scale of 0-10 for each subfield and overall:
- 0: Completely unrelated
- 3: Same broad area but different specific focus
- 5: Related with some overlapping ideas
- 7: Very similar direction
- 10: Nearly identical

Score each subfield:
1. **research_question**: How similar are the core research questions/problems being addressed?
2. **hypothesis**: How similar are the hypotheses or expected outcomes?
3. **proposed_method**: How similar are the proposed methodologies/techniques?
4. **novelty_claims**: How similar are the claimed contributions and novelty?
5. **experiment_details**: How similar are the experimental setups, datasets, or evaluation approaches?
6. **overall**: The holistic similarity considering all aspects

Respond with ONLY a JSON object:
{{
    "research_question": <0-10>,
    "hypothesis": <0-10>,
    "proposed_method": <0-10>,
    "novelty_claims": <0-10>,
    "experiment_details": <0-10>,
    "overall": <0-10>,
    "reasoning": "<brief explanation>"
}}"""

GAP_GROUNDEDNESS_PROMPT = """You are an expert research evaluator. Your task is to assess whether the "gaps" identified in a research reasoning are actually grounded in the inspiring papers, or if they are hallucinated/unsupported claims.

## Inspiring Papers:
{inspiring_papers}

## Gap Analysis from Reasoning:
{gap_analysis}

## Task:
Evaluate whether each identified gap is actually supported by the inspiring papers:

For each gap identified in the analysis, rate whether it is:
- **GROUNDED (1)**: The gap is clearly supported by limitations mentioned or implied in the inspiring papers
- **PARTIALLY GROUNDED (0.5)**: The gap is somewhat supported but may be overstated or not directly mentioned
- **UNGROUNDED (0)**: The gap appears to be hallucinated or not supported by the inspiring papers

Consider:
1. **Direct Evidence**: Does the gap directly correspond to limitations explicitly stated in the papers?
2. **Logical Inference**: Can the gap be reasonably inferred from the papers' content and scope?
3. **Overreach**: Does the gap analysis go beyond what the papers actually discuss?
4. **Accuracy**: Are the described limitations accurate representations of the papers?

Calculate an overall groundedness score (0-1) as the average across all identified gaps.

Respond with ONLY a JSON object:
{{
    "overall_groundedness": <0.0-1.0>,
    "num_gaps_identified": <number>,
    "gap_ratings": [
        {{
            "gap_description": "<brief description>",
            "rating": <0|0.5|1>,
            "reasoning": "<why this rating?>"
        }}
    ],
    "reasoning": "<brief overall assessment>"
}}"""


def format_candidate_for_judge(paper: Dict) -> str:
    """Format a candidate paper for LLM judge."""
    parts = []
    
    # Try structured fields first
    if paper.get('research_question'):
        parts.append(f"Research Question: {paper['research_question']}")
    if paper.get('hypothesis'):
        parts.append(f"Hypothesis: {paper['hypothesis']}")
    if paper.get('proposed_method'):
        parts.append(f"Proposed Method: {paper['proposed_method']}")
    if paper.get('novelty_claims'):
        parts.append(f"Novelty Claims: {paper['novelty_claims']}")
    
    # Fallback to title/abstract if no structured fields
    if not parts:
        if paper.get('title'):
            parts.append(f"Title: {paper['title']}")
        if paper.get('abstract'):
            parts.append(f"Abstract: {paper['abstract'][:1000]}")
    
    return '\n'.join(parts) if parts else "No information available"


def llm_judge_similarity(
    proposal: str,
    candidate: Dict,
    model: str = "gpt-4.1-mini",
    include_subfields: bool = False,
    max_proposal_chars: int = 11000,
    max_candidate_chars: int = 11000,
) -> Tuple[float, str, Optional[SubfieldScores], float]:
    """
    Use LLM to judge semantic similarity between proposal and candidate.
    
    Args:
        proposal: Generated research proposal text
        candidate: Candidate paper dictionary
        model: LLM model to use
        include_subfields: If True, also score individual subfields
        max_proposal_chars: Max characters for proposal (default 6000)
        max_candidate_chars: Max characters for candidate (default 4000)
    
    Returns:
        Tuple of (overall_score, reasoning, subfield_scores or None, cost_usd)
    """
    candidate_summary = format_candidate_for_judge(candidate)
    
    # Truncate if needed, with warning
    proposal_truncated = proposal[:max_proposal_chars]
    candidate_truncated = candidate_summary[:max_candidate_chars]
    
    if len(proposal) > max_proposal_chars:
        logger.debug(f"Proposal truncated: {len(proposal)} -> {max_proposal_chars} chars")
    if len(candidate_summary) > max_candidate_chars:
        logger.debug(f"Candidate truncated: {len(candidate_summary)} -> {max_candidate_chars} chars")
    
    if include_subfields:
        prompt = LLM_JUDGE_SUBFIELD_PROMPT.format(
            proposal=proposal_truncated,
            candidate_summary=candidate_truncated
        )
    else:
        prompt = LLM_JUDGE_PROMPT.format(
            proposal=proposal_truncated,
            candidate_summary=candidate_truncated
        )
    
    messages = [{"role": "user", "content": prompt}]
    
    try:
        response, cost = call_chat_completion(
            messages=messages,
            model=model,
            temperature=0.1  # Low temperature for consistent scoring
        )
        
        # Parse JSON response
        import re
        response = response.strip()
        if response.startswith('```'):
            response = re.sub(r'^```(?:json)?\n?', '', response)
            response = re.sub(r'\n?```$', '', response)
        
        result = json.loads(response)
        
        if include_subfields:
            subfield_scores = SubfieldScores(
                research_question=float(result.get('research_question', 0)),
                hypothesis=float(result.get('hypothesis', 0)),
                proposed_method=float(result.get('proposed_method', 0)),
                novelty_claims=float(result.get('novelty_claims', 0)),
                experiment_details=float(result.get('experiment_details', 0)),
                overall=float(result.get('overall', 0))
            )
            return subfield_scores.overall, result.get('reasoning', ''), subfield_scores, cost
        else:
            return result.get('score', 0), result.get('reasoning', ''), None, cost
        
    except Exception as e:
        logger.warning(f"LLM judge error: {e}")
        if include_subfields:
            return 0.0, str(e), SubfieldScores(), 0.0
        return 0.0, str(e), None, 0.0


def evaluate_gap_groundedness(
    gap_analysis: str,
    inspiring_papers: List[Dict],
    model: str = "gpt-4o-mini"
) -> Tuple[GapGroundednessResult, float]:
    """
    Evaluate whether gaps identified in reasoning are grounded in inspiring papers.

    Args:
        gap_analysis: The gap analysis text extracted from reasoning
        inspiring_papers: List of inspiring paper dictionaries
        model: LLM model to use for evaluation

    Returns:
        Tuple of (GapGroundednessResult, cost_usd)
    """
    if not gap_analysis.strip():
        return GapGroundednessResult(
            overall_groundedness=0.0,
            num_gaps_identified=0,
            reasoning="No gap analysis provided"
        ), 0.0

    # Format inspiring papers for the prompt
    inspiring_text = ""
    for i, paper in enumerate(inspiring_papers, 1):
        inspiring_text += f"\n### Paper {i}:\n"
        inspiring_text += format_candidate_for_judge(paper)
        inspiring_text += "\n"

    prompt = GAP_GROUNDEDNESS_PROMPT.format(
        inspiring_papers=inspiring_text,
        gap_analysis=gap_analysis[:2000]  # Limit length
    )

    messages = [{"role": "user", "content": prompt}]

    try:
        response, cost = call_chat_completion(
            messages=messages,
            model=model,
            temperature=0.1  # Low temperature for consistent scoring
        )

        # Parse JSON response
        import re
        response = response.strip()
        if response.startswith('```'):
            response = re.sub(r'^```(?:json)?\n?', '', response)
            response = re.sub(r'\n?```$', '', response)

        result = json.loads(response)

        return GapGroundednessResult(
            overall_groundedness=float(result.get('overall_groundedness', 0.0)),
            num_gaps_identified=int(result.get('num_gaps_identified', 0)),
            reasoning=result.get('reasoning', '')
        ), cost

    except Exception as e:
        logger.warning(f"Gap groundedness evaluation error: {e}")
        return GapGroundednessResult(
            overall_groundedness=0.0,
            num_gaps_identified=0,
            reasoning=f"Error: {str(e)}"
        ), 0.0


# ============================================================================
# Evaluation Pipeline
# ============================================================================

def evaluate_prediction(
    prediction: Dict,
    retriever,
    corpus: List[Dict],
    top_k: int = 10,
    judge_model: str = "gpt-4.1-mini",
    judge_top_n: int = 5,
    include_subfields: bool = False,
    evaluate_gap_groundedness_flag: bool = False,
    corpus_by_id: Optional[Dict[str, Dict]] = None,
    retrieval_max_chars: Optional[int] = None,
    retrieval_sections: Optional[List[str]] = None
) -> EvaluationResult:
    """
    Evaluate a single prediction.

    Args:
        prediction: Prediction dictionary with 'prediction' and 'root_title'
        retriever: BM25Retriever or EmbeddingRetriever
        corpus: Full corpus for lookup
        top_k: Number of papers to retrieve
        judge_model: Model for LLM judge
        judge_top_n: Only judge top-N retrieved papers (for cost savings)
        include_subfields: If True, also score individual subfields
        evaluate_gap_groundedness_flag: If True, evaluate gap groundedness
        corpus_by_id: Optional dict mapping paper_id to paper data (for gap evaluation)
        retrieval_max_chars: If set, truncate proposal to this many chars for retrieval only
        retrieval_sections: If set, extract only these sections for retrieval query
    """
    proposal = prediction.get('prediction', '')
    root_title = prediction.get('root_title', '')
    
    if not proposal:
        return EvaluationResult(
            prediction_id=prediction.get('id', -1),
            root_title=root_title,
            top_k_titles=[],
            top_k_scores=[],
            true_root_rank=-1,
            true_root_in_top_k=False,
            llm_similarity_scores=[],
            max_llm_score=0.0,
            true_root_llm_score=0.0,
            subfield_scores=None,
            max_subfield_scores=None,
            true_root_subfield_scores=None,
            cost_usd=0.0
        )
    
    # Build retrieval query (may be different from full proposal for LLM judging)
    retrieval_query = proposal
    
    # Extract specific sections if requested
    if retrieval_sections:
        retrieval_query = extract_sections_for_retrieval(retrieval_query, retrieval_sections)
    
    # Truncate for retrieval if requested
    if retrieval_max_chars and len(retrieval_query) > retrieval_max_chars:
        retrieval_query = retrieval_query[:retrieval_max_chars]
    
    # Retrieve top-k papers using the (possibly modified) query
    retrieved = retriever.retrieve(retrieval_query, top_k=top_k)
    top_k_titles = [p.get('title', '') for p, _ in retrieved]
    top_k_scores = [score for _, score in retrieved]
    
    # Find true root rank
    true_root_rank = -1
    for i, title in enumerate(top_k_titles):
        if title and root_title:
            if title.lower().strip() == root_title.lower().strip():
                true_root_rank = i + 1  # 1-indexed
                break
    
    true_root_in_top_k = true_root_rank > 0
    
    # Debug: Check if root title could be found in any of top-k
    if not true_root_in_top_k and root_title:
        # Check if there are any titles at all
        has_titles = any(t for t in top_k_titles)
        if not has_titles:
            pass  # Silent - titles not loaded
    
    # LLM judge on top-N papers
    llm_scores = []
    subfield_scores_list = [] if include_subfields else None
    true_root_llm_score = 0.0
    true_root_subfield = None
    total_cost = 0.0
    
    for i, (paper, _) in enumerate(retrieved[:judge_top_n]):
        score, _, subfield, cost = llm_judge_similarity(
            proposal=proposal,
            candidate=paper,
            model=judge_model,
            include_subfields=include_subfields
        )
        llm_scores.append(score)
        total_cost += cost
        
        if include_subfields and subfield:
            subfield_scores_list.append(subfield)
        
        if paper.get('title', '').lower().strip() == root_title.lower().strip():
            true_root_llm_score = score
            if include_subfields:
                true_root_subfield = subfield
    
    # Compute max and avg subfield scores (across all judged papers)
    # max_subfield: from the paper with highest overall score (tiebreak by sum of subfields)
    max_subfield = None
    avg_subfield = None
    if include_subfields and subfield_scores_list:
        # Find the best paper by overall score, with tiebreak by sum of subfield scores
        def subfield_sum(s):
            return s.research_question + s.hypothesis + s.proposed_method + s.novelty_claims + s.experiment_details
        
        # Sort by (overall score, subfield sum) descending
        best_subfield = max(
            subfield_scores_list,
            key=lambda s: (s.overall, subfield_sum(s))
        )
        max_subfield = best_subfield
        
        avg_subfield = SubfieldScores(
            research_question=np.mean([s.research_question for s in subfield_scores_list]),
            hypothesis=np.mean([s.hypothesis for s in subfield_scores_list]),
            proposed_method=np.mean([s.proposed_method for s in subfield_scores_list]),
            novelty_claims=np.mean([s.novelty_claims for s in subfield_scores_list]),
            experiment_details=np.mean([s.experiment_details for s in subfield_scores_list]),
            overall=np.mean([s.overall for s in subfield_scores_list])
        )

    # Evaluate gap groundedness if requested
    gap_groundedness_result = None
    if evaluate_gap_groundedness_flag and corpus_by_id:
        gap_analysis = prediction.get('gap_analysis', '')
        inspiring_paper_ids = prediction.get('inspiring_paper_ids', [])

        if gap_analysis and inspiring_paper_ids:
            inspiring_papers = []
            for pid in inspiring_paper_ids:
                if pid in corpus_by_id:
                    inspiring_papers.append(corpus_by_id[pid])

            if inspiring_papers:
                gap_groundedness_result, gap_cost = evaluate_gap_groundedness(
                    gap_analysis=gap_analysis,
                    inspiring_papers=inspiring_papers,
                    model=judge_model
                )
                total_cost += gap_cost

    return EvaluationResult(
        prediction_id=prediction.get('id', -1),
        root_title=root_title,
        top_k_titles=top_k_titles,
        top_k_scores=top_k_scores,
        true_root_rank=true_root_rank,
        true_root_in_top_k=true_root_in_top_k,
        llm_similarity_scores=llm_scores,
        max_llm_score=max(llm_scores) if llm_scores else 0.0,
        avg_llm_score=np.mean(llm_scores) if llm_scores else 0.0,
        true_root_llm_score=true_root_llm_score,
        subfield_scores=subfield_scores_list,
        max_subfield_scores=max_subfield,
        avg_subfield_scores=avg_subfield,
        true_root_subfield_scores=true_root_subfield,
        gap_groundedness=gap_groundedness_result,
        cost_usd=total_cost
    )


def compute_aggregate_metrics(results: List[EvaluationResult], include_subfields: bool = False, include_gap_groundedness: bool = False) -> Dict:
    """Compute aggregate metrics from evaluation results."""
    n = len(results)
    if n == 0:
        return {}
    
    # Retrieval metrics (top-k, typically 10)
    recall_at_k = sum(1 for r in results if r.true_root_in_top_k) / n
    
    ranks = [r.true_root_rank for r in results if r.true_root_rank > 0]
    mrr = np.mean([1/r for r in ranks]) if ranks else 0.0  # Mean Reciprocal Rank
    avg_rank = np.mean(ranks) if ranks else float('inf')
    
    # Top-5 retrieval metrics
    recall_at_5 = sum(1 for r in results if 0 < r.true_root_rank <= 5) / n
    ranks_top5 = [r.true_root_rank for r in results if 0 < r.true_root_rank <= 5]
    mrr_at_5 = np.mean([1/r for r in ranks_top5]) if ranks_top5 else 0.0
    
    # LLM judge metrics (all judged papers)
    max_scores = [r.max_llm_score for r in results if r.llm_similarity_scores]
    avg_max_score = np.mean(max_scores) if max_scores else 0.0
    
    true_root_scores = [r.true_root_llm_score for r in results if r.true_root_llm_score > 0]
    avg_true_root_score = np.mean(true_root_scores) if true_root_scores else 0.0
    
    # Top-5 LLM judge metrics (only first 5 judged papers)
    max_scores_top5 = [max(r.llm_similarity_scores[:5]) for r in results if len(r.llm_similarity_scores) >= 5]
    avg_scores_top5 = [np.mean(r.llm_similarity_scores[:5]) for r in results if len(r.llm_similarity_scores) >= 5]
    avg_max_score_top5 = np.mean(max_scores_top5) if max_scores_top5 else 0.0
    avg_avg_score_top5 = np.mean(avg_scores_top5) if avg_scores_top5 else 0.0
    
    # Cost tracking
    total_cost = sum(r.cost_usd for r in results)
    
    metrics = {
        'n_samples': n,
        # Top-K metrics (all retrieved)
        'recall_at_k': recall_at_k,
        'mrr': mrr,
        'avg_rank': avg_rank if avg_rank != float('inf') else None,
        'avg_max_llm_score': avg_max_score,
        'avg_true_root_llm_score': avg_true_root_score,
        'n_with_true_root_in_top_k': sum(1 for r in results if r.true_root_in_top_k),
        # Top-5 metrics
        'recall_at_5': recall_at_5,
        'mrr_at_5': mrr_at_5,
        'n_with_true_root_in_top_5': sum(1 for r in results if 0 < r.true_root_rank <= 5),
        'avg_max_llm_score_top5': avg_max_score_top5,
        'avg_avg_llm_score_top5': avg_avg_score_top5,
        # Cost
        'total_cost_usd': total_cost,
        'avg_cost_per_sample_usd': total_cost / n if n > 0 else 0.0,
    }
    
    # Average LLM score across all retrieved papers
    avg_scores = [r.avg_llm_score for r in results if r.llm_similarity_scores]
    metrics['avg_avg_llm_score'] = np.mean(avg_scores) if avg_scores else 0.0
    
    # Subfield metrics (if available)
    if include_subfields:
        subfield_names = ['research_question', 'hypothesis', 'proposed_method', 'novelty_claims', 'experiment_details', 'overall']
        
        # Average max subfield scores (best match per prediction)
        max_subfields = [r.max_subfield_scores for r in results if r.max_subfield_scores]
        if max_subfields:
            metrics['avg_max_subfield_scores'] = {
                field: np.mean([getattr(s, field) for s in max_subfields])
                for field in subfield_names
            }
        
        # Average of average subfield scores (avg across retrieved papers per prediction)
        avg_subfields = [r.avg_subfield_scores for r in results if r.avg_subfield_scores]
        if avg_subfields:
            metrics['avg_avg_subfield_scores'] = {
                field: np.mean([getattr(s, field) for s in avg_subfields])
                for field in subfield_names
            }
        
        # Average true root subfield scores
        true_root_subfields = [r.true_root_subfield_scores for r in results if r.true_root_subfield_scores]
        if true_root_subfields:
            metrics['avg_true_root_subfield_scores'] = {
                field: np.mean([getattr(s, field) for s in true_root_subfields])
                for field in subfield_names
            }

    # Gap groundedness metrics (if available)
    if include_gap_groundedness:
        gap_results = [r.gap_groundedness for r in results if r.gap_groundedness]
        if gap_results:
            metrics['avg_gap_groundedness'] = np.mean([g.overall_groundedness for g in gap_results])
            metrics['avg_gaps_identified'] = np.mean([g.num_gaps_identified for g in gap_results])
            metrics['n_with_gap_evaluation'] = len(gap_results)

    return metrics


def run_evaluation(
    predictions_path: str,
    corpus_path: str,
    output_path: str,
    retriever_type: str = "bm25",
    text_field: str = "structured",
    top_k: int = 10,
    judge_model: str = "gpt-4.1-mini",
    judge_top_n: int = 5,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_cache: Optional[str] = None,
    titles_path: Optional[str] = None,
    max_samples: Optional[int] = None,
    strip_reasoning: bool = False,
    remove_repetition: bool = False,
    include_subfields: bool = False,
    evaluate_gap_groundedness: bool = False,
    retrieval_max_chars: Optional[int] = None,
    retrieval_sections: Optional[List[str]] = None,
    num_workers: int = 1,
    drop_incomplete: bool = False
):
    """
    Run full evaluation pipeline.
    
    Args:
        predictions_path: Path to baseline predictions JSON
        corpus_path: Path to corpus JSON (structured papers)
        output_path: Path to save evaluation results
        retriever_type: "bm25" or "embedding"
        text_field: Field(s) to use for retrieval: "title", "structured", "both"
        top_k: Number of papers to retrieve
        judge_model: Model for LLM judge
        judge_top_n: Only judge top-N retrieved papers
        embedding_model: Embedding model for retrieval (OpenAI or local SOTA)
        embedding_cache: Path to cache embeddings
        titles_path: Path to JSONL file with paper titles (id -> title)
        max_samples: Max samples to evaluate
        strip_reasoning: If True, remove reasoning process from proposals before evaluation
        remove_repetition: If True, also remove duplicate sentences from proposals
        include_subfields: If True, also evaluate subfield similarity scores
        retrieval_max_chars: If set, truncate proposals to this many chars for retrieval
        retrieval_sections: If set, extract only these sections for retrieval query
    """
    logger.info("=" * 80)
    logger.info("EVALUATION: Research Proposal Quality")
    logger.info("=" * 80)
    logger.info(f"Predictions: {predictions_path}")
    logger.info(f"Corpus: {corpus_path}")
    logger.info(f"Titles: {titles_path or 'Not provided'}")
    logger.info(f"Retriever: {retriever_type}")
    if retriever_type == "embedding":
        model_type = EMBEDDING_MODELS.get(embedding_model, {}).get("type", "unknown")
        logger.info(f"Embedding model: {embedding_model} ({model_type})")
    logger.info(f"Text field: {text_field}")
    logger.info(f"Top-K: {top_k}")
    logger.info(f"Judge model: {judge_model}")
    logger.info(f"Parallel workers: {num_workers}")
    logger.info(f"Strip reasoning: {strip_reasoning}")
    logger.info(f"Remove repetition: {remove_repetition}")
    logger.info(f"Drop incomplete: {drop_incomplete}")
    logger.info(f"Include subfields: {include_subfields}")
    logger.info(f"Evaluate gap groundedness: {evaluate_gap_groundedness}")
    logger.info(f"Retrieval max chars: {retrieval_max_chars or 'No limit'}")
    logger.info(f"Retrieval sections: {retrieval_sections or 'All'}")
    logger.info("=" * 80)

    # Load data
    predictions = load_predictions(predictions_path)
    corpus = load_corpus(corpus_path, titles_path=titles_path)

    # Create corpus_by_id mapping for gap groundedness evaluation
    corpus_by_id = {}
    for paper in corpus:
        paper_id = paper.get('paper_id')
        if paper_id:
            corpus_by_id[paper_id] = paper

    if max_samples:
        predictions = predictions[:max_samples]
    
    # Optionally strip reasoning from proposals
    if strip_reasoning or remove_repetition:
        action = "Stripping reasoning" + (" and removing repetition" if remove_repetition else "")
        logger.info(f"{action} from predictions...")
        stripped_count = 0
        for pred in predictions:
            original = pred.get('prediction', '')
            stripped = strip_reasoning_from_proposal(original, remove_repetition=remove_repetition)
            if stripped != original:
                pred['prediction'] = stripped
                stripped_count += 1
        logger.info(f"Modified {stripped_count}/{len(predictions)} predictions")
    
    # Optionally drop incomplete proposals (missing sections)
    if drop_incomplete:
        logger.info("Checking for incomplete proposals...")
        original_count = len(predictions)
        
        # Check each prediction and collect stats
        section_stats = {s: 0 for s in REQUIRED_SECTIONS}
        complete_predictions = []
        
        for pred in predictions:
            proposal = pred.get('prediction', '')
            sections = check_proposal_sections(proposal)
            
            # Update stats
            for section, present in sections.items():
                if present:
                    section_stats[section] += 1
            
            # Keep only complete proposals
            if all(sections.values()):
                complete_predictions.append(pred)
        
        dropped_count = original_count - len(complete_predictions)
        predictions = complete_predictions
        
        logger.info(f"Section presence in {original_count} proposals:")
        for section, count in section_stats.items():
            pct = count / original_count * 100 if original_count > 0 else 0
            logger.info(f"  {section}: {count}/{original_count} ({pct:.1f}%)")
        logger.info(f"Dropped {dropped_count}/{original_count} incomplete proposals ({dropped_count/original_count*100:.1f}%)")
        logger.info(f"Remaining: {len(predictions)} complete proposals")
    
    logger.info(f"Loaded {len(predictions)} predictions, {len(corpus)} corpus papers")
    
    # Initialize retriever
    if retriever_type == "bm25":
        retriever = BM25Retriever(corpus, text_field=text_field)
    else:
        # Auto-include model name in cache path to avoid mixing different model embeddings
        if embedding_cache:
            cache_dir = os.path.dirname(embedding_cache)
            cache_name = os.path.basename(embedding_cache)
            model_slug = embedding_model.replace("/", "_").replace("-", "_")
            if not cache_name.startswith(model_slug):
                cache_name = f"{model_slug}_{cache_name}"
            embedding_cache = os.path.join(cache_dir, cache_name)
        
        retriever = EmbeddingRetriever(
            corpus, 
            text_field=text_field,
            model=embedding_model,
            cache_path=embedding_cache
        )
    
    # Evaluate each prediction
    results = []
    
    def eval_single(pred):
        """Evaluate a single prediction (for parallel processing)."""
        return evaluate_prediction(
            prediction=pred,
            retriever=retriever,
            corpus=corpus,
            top_k=top_k,
            judge_model=judge_model,
            judge_top_n=judge_top_n,
            include_subfields=include_subfields,
            evaluate_gap_groundedness_flag=evaluate_gap_groundedness,
            corpus_by_id=corpus_by_id if evaluate_gap_groundedness else None,
            retrieval_max_chars=retrieval_max_chars,
            retrieval_sections=retrieval_sections
        )
    
    if num_workers > 1:
        # Parallel evaluation using ThreadPoolExecutor
        logger.info(f"Running parallel evaluation with {num_workers} workers")
        
        # Create list to hold results in order
        results = [None] * len(predictions)
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks with their indices
            future_to_idx = {
                executor.submit(eval_single, pred): idx 
                for idx, pred in enumerate(predictions)
            }
            
            # Collect results as they complete
            for future in tqdm(as_completed(future_to_idx), total=len(predictions), desc="Evaluating"):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Error evaluating prediction {idx}: {e}")
                    # Create a dummy result for failed evaluations
                    results[idx] = EvaluationResult(
                        prediction_id=predictions[idx].get('id', idx),
                        proposal=predictions[idx].get('prediction', ''),
                        true_root_paper_id=predictions[idx].get('root_paper_id', ''),
                        true_root_title=predictions[idx].get('root_title', ''),
                        retrieved_paper_ids=[],
                        retrieved_titles=[],
                        retrieved_scores=[],
                        llm_scores=[],
                        true_root_rank=None,
                        true_root_llm_score=0.0,
                        max_llm_score=0.0,
                        avg_llm_score=0.0,
                        max_llm_score_top5=0.0,
                        avg_llm_score_top5=0.0,
                        cost_usd=0.0
                    )
    else:
        # Sequential evaluation (original behavior)
        for pred in tqdm(predictions, desc="Evaluating"):
            result = eval_single(pred)
            results.append(result)

    # Compute aggregate metrics
    metrics = compute_aggregate_metrics(
        results,
        include_subfields=include_subfields,
        include_gap_groundedness=evaluate_gap_groundedness
    )
    
    # Log summary
    logger.info("=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)
    logger.info(f"Samples: {metrics['n_samples']}")
    logger.info("--- Retrieval Metrics ---")
    logger.info(f"Recall@5: {metrics['recall_at_5']:.2%} ({metrics['n_with_true_root_in_top_5']}/{metrics['n_samples']})")
    logger.info(f"Recall@{top_k}: {metrics['recall_at_k']:.2%} ({metrics['n_with_true_root_in_top_k']}/{metrics['n_samples']})")
    logger.info(f"MRR@5: {metrics['mrr_at_5']:.4f}")
    logger.info(f"MRR@{top_k}: {metrics['mrr']:.4f}")
    if metrics['avg_rank']:
        logger.info(f"Avg Rank (when found): {metrics['avg_rank']:.2f}")
    logger.info("--- LLM Judge Scores ---")
    logger.info(f"Avg Max Score (top-5): {metrics['avg_max_llm_score_top5']:.2f}/10")
    logger.info(f"Avg Max Score (all): {metrics['avg_max_llm_score']:.2f}/10")
    logger.info(f"Avg Avg Score (top-5): {metrics['avg_avg_llm_score_top5']:.2f}/10")
    logger.info(f"Avg Avg Score (all): {metrics['avg_avg_llm_score']:.2f}/10")
    logger.info(f"Avg True Root Score: {metrics['avg_true_root_llm_score']:.2f}/10")
    
    # Log subfield scores if available
    if include_subfields and 'avg_max_subfield_scores' in metrics:
        logger.info("--- Subfield Scores (Best paper by overall score) ---")
        for field, score in metrics['avg_max_subfield_scores'].items():
            logger.info(f"  {field}: {score:.2f}/10")
    
    if include_subfields and 'avg_avg_subfield_scores' in metrics:
        logger.info("--- Subfield Scores (Avg across retrieved) ---")
        for field, score in metrics['avg_avg_subfield_scores'].items():
            logger.info(f"  {field}: {score:.2f}/10")
    
    if include_subfields and 'avg_true_root_subfield_scores' in metrics:
        logger.info("--- Subfield Scores (True Root) ---")
        for field, score in metrics['avg_true_root_subfield_scores'].items():
            logger.info(f"  {field}: {score:.2f}/10")

    # Log gap groundedness if available
    if evaluate_gap_groundedness and 'avg_gap_groundedness' in metrics:
        logger.info("--- Gap Groundedness ---")
        logger.info(f"  Avg Gap Groundedness: {metrics['avg_gap_groundedness']:.2f}/1")
        logger.info(f"  Avg Gaps Identified: {metrics['avg_gaps_identified']:.1f}")
        logger.info(f"  Samples with Gap Eval: {metrics['n_with_gap_evaluation']}")
    
    # Log cost
    logger.info("--- Cost ---")
    logger.info(f"Total Cost: ${metrics['total_cost_usd']:.4f}")
    logger.info(f"Avg Cost per Sample: ${metrics['avg_cost_per_sample_usd']:.6f}")

    logger.info("=" * 80)
    
    # Save results
    output_data = {
        'config': {
            'predictions_path': predictions_path,
            'corpus_path': corpus_path,
            'retriever_type': retriever_type,
            'embedding_model': embedding_model if retriever_type == "embedding" else None,
            'top_k': top_k,
            'judge_model': judge_model,
            'judge_top_n': judge_top_n,
            'num_workers': num_workers,
            'strip_reasoning': strip_reasoning,
            'remove_repetition': remove_repetition,
            'drop_incomplete': drop_incomplete,
            'include_subfields': include_subfields,
            'retrieval_max_chars': retrieval_max_chars,
            'retrieval_sections': retrieval_sections
        },
        'metrics': metrics,
        'results': [asdict(r) for r in results]
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.success(f"Saved evaluation to: {output_path}")
    
    return metrics, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate research proposal generation")
    parser.add_argument(
        "--predictions",
        default="./predictions/baseline_predictions.json",
        help="Path to predictions JSON (output of inference/generate_qwen_predictions.py)"
    )
    parser.add_argument(
        "--corpus",
        nargs='+',
        default=["./data/test_set/structured_papers.json"],
        help="Path(s) to corpus JSON files (structured papers). Default ships with this repo: "
             "3,495 structured papers covering the 819 eval instances and their inspiring papers "
             "(see data/README.md)."
    )
    parser.add_argument(
        "--text-field",
        choices=["title", "structured", "both"],
        default="structured",
        help="Text field for retrieval: title, structured (all fields), both (title+abstract)"
    )
    parser.add_argument(
        "--output",
        default="./evaluation/results/eval_results.json",
        help="Path to save evaluation results"
    )
    parser.add_argument(
        "--retriever",
        choices=["bm25", "embedding"],
        default="bm25",
        help="Retriever type (default: bm25)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of papers to retrieve (default: 10)"
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-4.1-mini",
        help="Model for LLM judge (default: gpt-4.1-mini)"
    )
    parser.add_argument(
        "--judge-top-n",
        type=int,
        default=5,
        help="Only judge top-N retrieved papers (default: 5)"
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"Embedding model. OpenAI: text-embedding-3-large (default), text-embedding-3-small. "
             f"Local SOTA (free): bge-large-en-v1.5, gte-large-en-v1.5, specter2 (best for papers). "
             f"Supported: {', '.join(EMBEDDING_MODELS.keys())}"
    )
    parser.add_argument(
        "--embedding-cache",
        default="./evaluation/cache/corpus_embeddings.npy",
        help="Path to cache corpus embeddings (auto-includes model name)"
    )
    parser.add_argument(
        "--titles",
        nargs='+',
        default=None,
        help="Optional path(s) to JSON files with paper titles, used as a fallback if the corpus "
             "structured_papers.json is missing title fields. The shipped corpus already has titles, "
             "so this is rarely needed."
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples to evaluate"
    )
    parser.add_argument(
        "--strip-reasoning",
        action="store_true",
        default=True,
        help="Strip reasoning process from proposals, keeping only the proposal content (default: True)"
    )
    parser.add_argument(
        "--no-strip-reasoning",
        action="store_false",
        dest="strip_reasoning",
        help="Keep full proposals including reasoning process"
    )
    parser.add_argument(
        "--remove-repetition",
        action="store_true",
        default=False,
        help="Remove duplicate sentences from proposals (helps with verbose model outputs)"
    )
    parser.add_argument(
        "--drop-incomplete",
        action="store_true",
        default=False,
        help="Drop proposals missing any required section (Research Question, Hypothesis, "
             "Proposed Method, Novelty Claims, Experiment Details) from evaluation"
    )
    parser.add_argument(
        "--subfield-scores",
        action="store_true",
        default=False,
        help="Evaluate subfield similarity (research_question, hypothesis, proposed_method, experiment_details)"
    )
    parser.add_argument(
        "--gap-groundedness",
        action="store_true",
        default=False,
        help="Evaluate gap groundedness to detect hallucinated gaps in reasoning"
    )
    parser.add_argument(
        "--retrieval-max-chars",
        type=int,
        default=None,
        help="Truncate proposals to this many chars for retrieval only (helps with long outputs)"
    )
    parser.add_argument(
        "--retrieval-sections",
        type=str,
        default=None,
        help="Comma-separated list of sections to use for retrieval (e.g., 'research_question,hypothesis,proposed_method'). "
             "Valid sections: research_question, hypothesis, proposed_method, novelty_claims, experiment_details"
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Path to save log file (optional, logs to evaluation/logs/ by default)"
    )
    parser.add_argument(
        "--mode",
        default=None,
        help="Mode name for output file naming (e.g., 'one-layer', 'two-layer', 'no-inspiration')"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel workers for evaluation (default: 1, sequential). "
             "Recommended: 4-8 for API-based evaluation."
    )
    
    args = parser.parse_args()
    
    # Configure loguru
    import time
    log_dir = Path("./evaluation/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # Extract base names from prediction and corpus paths
    pred_name = Path(args.predictions).stem  # e.g., "baseline_predictions"
    # Handle corpus as list (multiple files)
    if isinstance(args.corpus, list):
        corpus_name = "_".join(Path(c).stem for c in args.corpus[:2])  # Use first 2 for naming
        if len(args.corpus) > 2:
            corpus_name += f"_plus{len(args.corpus)-2}"
    else:
        corpus_name = Path(args.corpus).stem
    mode_suffix = f"_{args.mode}" if args.mode else ""
    
    if args.log_file:
        log_path = args.log_file
    else:
        log_path = log_dir / f"eval_{pred_name}_{corpus_name}{mode_suffix}_{timestamp}.log"
    
    # Generate output path with mode if not explicitly provided or using default
    output_path = args.output
    if args.output == "./evaluation/results/eval_results.json":
        # Using default output, generate dynamic name with mode
        output_dir = Path("./evaluation/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"eval_{pred_name}_{corpus_name}{mode_suffix}_{timestamp}.json")
    
    # Add file handler (keep console handler as default)
    logger.add(
        log_path,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="DEBUG",
        rotation="10 MB"
    )
    logger.info(f"Logging to: {log_path}")
    
    # Parse retrieval sections if provided
    retrieval_sections = None
    if args.retrieval_sections:
        retrieval_sections = [s.strip() for s in args.retrieval_sections.split(',')]
    
    run_evaluation(
        predictions_path=args.predictions,
        corpus_path=args.corpus,
        output_path=output_path,
        retriever_type=args.retriever,
        text_field=args.text_field,
        top_k=args.top_k,
        judge_model=args.judge_model,
        judge_top_n=args.judge_top_n,
        embedding_model=args.embedding_model,
        embedding_cache=args.embedding_cache,
        titles_path=args.titles,
        max_samples=args.max_samples,
        strip_reasoning=args.strip_reasoning,
        remove_repetition=args.remove_repetition,
        include_subfields=args.subfield_scores,
        evaluate_gap_groundedness=args.gap_groundedness,
        retrieval_max_chars=args.retrieval_max_chars,
        retrieval_sections=retrieval_sections,
        num_workers=args.num_workers,
        drop_incomplete=args.drop_incomplete
    )

