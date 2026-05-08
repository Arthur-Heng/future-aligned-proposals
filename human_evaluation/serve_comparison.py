#!/usr/bin/env python3
"""
Web-based comparison tool for evaluating generated research proposals vs real papers.

Serves a modern UI that shows pairs side-by-side (generated proposal vs real paper)
and allows human annotators to rate: soundness, excitement, overall.

Usage:
    python human_evaluation/serve_comparison.py --predictions predictions/qwen-14b-stepwise-cot.json --port 8899
"""

import os
import sys
import json
import re
import argparse
import time
import random
import hashlib
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
import html as html_lib

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.api import call_chat_completion

NEWLY_ADDED_TOP_TITLES = [
    "Reasoning-Enhanced Healthcare Predictions with Knowledge Graph Community Retrieval",
    "ToolRL: Reward is All Tool Learning Needs",
    "DynamicRAG: Leveraging Outputs of Large Language Model as Feedback for Dynamic Reranking in Retrieval-Augmented Generation",
    "RAST: Reasoning Activation in LLMs via Small-model Transfer",
    "Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG",
    "PARTONOMY: Large Multimodal Models with Part-Level Visual Understanding",
    "DyMU: Dynamic Merging and Virtual Unmerging for Efficient Variable-Length VLMs",
    "DYMU: Dynamic Merging and Virtual Unmerging for Efficient VLMs",
    "Law of the Weakest Link: Cross Capabilities of Large Language Models",
    "RepoGraph: Enhancing AI Software Engineering with Repository-level Code Graph",
]


def _normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", (title or "").strip().lower())


def prioritize_newly_added_pairs(pairs):
    """
    Move known manually-added papers to the top while preserving
    relative order for all other pairs.
    """
    priority_map = {
        _normalize_title(t): i for i, t in enumerate(NEWLY_ADDED_TOP_TITLES)
    }
    decorated = list(enumerate(pairs))
    decorated.sort(
        key=lambda item: (
            _normalize_title(item[1].get("root_title", "")) not in priority_map,
            priority_map.get(_normalize_title(item[1].get("root_title", "")), 10**9),
            item[0],
        )
    )
    return [item[1] for item in decorated]


def _text_fingerprint(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()


def _extract_json_block(text: str) -> str:
    """Extract the first JSON object from a model response."""
    text = (text or "").strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start:end + 1]
    return "{}"


def generate_readability_enhancement(text: str, model: str = "gpt-4.1-nano") -> dict:
    """
    Generate a short summary and key phrases for highlighting.
    Returns {'summary': str, 'keywords': list[str], 'cost': float}.
    """
    system = (
        "You improve readability of ML research proposals. "
        "Return compact JSON only."
    )
    user = (
        "Given the proposal text, produce JSON with:\n"
        "- summary: one very short sentence (<= 16 words)\n"
        "- keywords: 4-6 short keyword phrases to highlight\n\n"
        "Rules:\n"
        "- Be faithful to the content\n"
        "- Prefer technical terms and entities\n"
        "- No markdown, no extra keys\n\n"
        "JSON schema:\n"
        "{\"summary\":\"...\",\"keywords\":[\"...\",\"...\"]}\n\n"
        f"Proposal:\n{text}"
    )
    resp, cost = call_chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        model=model,
        temperature=0.1,
        max_tokens=180,
        response_format={"type": "json_object"},
    )
    parsed = json.loads(_extract_json_block(resp))
    summary = (parsed.get("summary") or "").strip()
    keywords = parsed.get("keywords") or []
    keywords = [k.strip() for k in keywords if isinstance(k, str) and k.strip()]
    keywords = keywords[:6]
    return {"summary": summary, "keywords": keywords, "cost": cost}


def highlight_keywords_in_html(html_text: str, keywords: list[str]) -> str:
    """Highlight keywords in rendered HTML using <mark> tags."""
    out = html_text
    # Longer phrases first to avoid partial overlaps.
    for kw in sorted(set(k for k in keywords if k), key=len, reverse=True):
        pattern = re.compile(rf"(?i)\b({re.escape(kw)})\b")
        out = pattern.sub(r"<mark>\1</mark>", out)
    return out


def strip_reasoning_from_text(text: str) -> str:
    """Remove reasoning blocks from step-wise CoT text, keeping only proposal content."""
    if not text:
        return text

    def _strip_step_blocks(t):
        # Remove Step reasoning blocks but stop before:
        #   - ## headers
        #   - non-Step ### headers
        #   - known proposal field labels (Research Question:, Proposed Method:, etc.)
        return re.sub(
            r'###\s*Step\s*\d+[^\n]*\n'
            r'(?:(?!##\s|###\s*(?!Step)|Research Question:|Hypothesis:|Proposed Method:|Novelty Claims?:|Experiment Details?:).*\n)*',
            '', t
        )

    # Step-wise CoT detection
    step_markers = re.findall(r'###\s*Step\s*\d', text)
    if len(step_markers) >= 2:
        cleaned = _strip_step_blocks(text)
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
            return cleaned

    # Full CoT: find proposal section header
    proposal_patterns = [
        r'##\s*Proposed Research\s*(?:Idea)?',
        r'##\s*Research Proposal',
        r'##\s*Novel Research\s*(?:Idea)?',
        r'##\s*Proposal',
    ]
    for pattern in proposal_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            proposal_text = text[match.end():].strip()
            proposal_text = re.sub(r'^[\s\n#]*', '', proposal_text)
            proposal_text = _strip_step_blocks(proposal_text)
            return proposal_text.strip()

    return text


def markdown_to_html(text: str) -> str:
    """Simple markdown to HTML conversion with section dividers."""
    if not text:
        return ""
    text = html_lib.escape(text)
    
    # Known proposal section names (for consistent blue styling)
    section_names = ['Research Question', 'Hypothesis', 'Proposed Method', 'Novelty Claims', 'Novelty', 'Experiment Details', 'Experimental Design']
    
    # Convert ### headers for known sections to field-label style (blue)
    for section in section_names:
        # Match "### Section Name" or "### Section Name:" at start of line
        pattern = rf'^###\s*{re.escape(section)}:?\s*$'
        replacement = f'<hr class="section-divider"><span class="field-label">{section}:</span>'
        text = re.sub(pattern, replacement, text, flags=re.MULTILINE | re.IGNORECASE)
    
    # Other ### headers become regular h4
    text = re.sub(r'^### (.+)$', r'<hr class="section-divider"><h4>\1</h4>', text, flags=re.MULTILINE)
    # ## headers become h3
    text = re.sub(r'^## (.+)$', r'<hr class="section-divider"><h3>\1</h3>', text, flags=re.MULTILINE)
    
    # Bold text
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    
    # Add dividers before key proposal fields with colon format (e.g., "Research Question:")
    for section in section_names:
        field = f'{section}:'
        escaped_field = html_lib.escape(field)
        # Only replace if not already wrapped in field-label
        if escaped_field in text and f'class="field-label">{escaped_field}' not in text:
            text = text.replace(escaped_field, f'<hr class="section-divider"><span class="field-label">{escaped_field}</span>')
    
    # Lists
    text = re.sub(r'^- (.+)$', r'<li>\1</li>', text, flags=re.MULTILINE)
    # Wrap consecutive <li> in <ul>
    text = re.sub(r'((?:<li>.*?</li>\n?)+)', r'<ul>\1</ul>', text)
    # Paragraphs
    text = re.sub(r'\n\n+', '</p><p>', text)
    text = f'<p>{text}</p>'
    # Clean up empty paragraphs
    text = re.sub(r'<p>\s*</p>', '', text)
    # Remove the very first divider (no line before the first section)
    text = text.replace('<hr class="section-divider">', '', 1) if text.startswith('<p><hr') else text
    # Also handle case where first hr is inside first <p>
    text = re.sub(r'^(<p>)\s*<hr class="section-divider">', r'\1', text, count=1)
    return text


def _clean_proposal_headers_and_title(text: str) -> str:
    """
    Normalize proposal formatting for display:
    - Remove intro text like "Based on these inspiring papers..."
    - Remove "## Proposed Research" headers
    - Remove standalone "Title" lines
    - Remove plain title lines before first section header
    - Strip '(YYYY)' from bold title lines
    """
    if not text:
        return text
    
    # Remove common intro phrases
    intro_patterns = [
        r"Based on these inspiring papers,?\s*here'?s?\s*a?\s*novel research proposal:?\s*",
        r"Here'?s?\s*a?\s*novel research proposal:?\s*",
        r"Proposed Proposal:?\s*",
    ]
    for pattern in intro_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    lines = text.splitlines()
    out = []
    
    # Section header patterns
    section_header_pattern = re.compile(
        r'^\s*(###?\s*)?(Research Question|Hypothesis|Proposed Method|Novelty|Experiment)',
        re.IGNORECASE
    )
    
    # Track if we've seen a section header yet
    seen_section_header = False

    for line in lines:
        # Remove '## Proposed Research' headers entirely
        if re.match(r'^\s*##\s*Proposed Research\s*$', line, re.IGNORECASE):
            continue
        
        # Remove standalone "Title" line (without colon, often followed by actual title)
        if re.match(r'^\s*Title\s*$', line, re.IGNORECASE):
            continue
        
        # Remove "Title | **..." format - extract and keep just the bold title
        m = re.match(r'^\s*Title\s*\|\s*(\*\*.+?\*\*.*)\s*$', line, re.IGNORECASE)
        if m:
            out.append(m.group(1))
            continue
        
        # Remove "Title | text" format without bold (non-bold title line)
        if re.match(r'^\s*Title\s*\|', line, re.IGNORECASE):
            continue

        # Remove trailing year from bold title lines: **Title** (2024) -> **Title**
        m = re.match(r'^\s*(\*\*.+?\*\*)\s*\(\d{4}\)\s*$', line)
        if m:
            out.append(m.group(1))
            continue
        
        # Check if this is a section header
        if section_header_pattern.match(line):
            seen_section_header = True
        
        # Skip non-empty, non-bold lines before the first section header (likely stray titles)
        if not seen_section_header and line.strip() and not line.strip().startswith('**'):
            # This is likely a plain title line before the first section - skip it
            continue

        out.append(line)

    # Remove leading empty lines
    while out and not out[0].strip():
        out.pop(0)

    return "\n".join(out).strip()


def _has_bold_title_line(text: str) -> bool:
    """Return True if proposal contains a likely bold title line (standalone bold text)."""
    for line in (text or "").splitlines():
        # Match standalone bold line: **Title text here**
        if re.match(r'^\s*\*\*.+?\*\*\s*$', line):
            return True
    return False


def _remove_bold_title_line(text: str) -> str:
    """Remove all bold-only title lines from a proposal."""
    kept = []
    for line in (text or "").splitlines():
        # Skip standalone bold lines (likely titles)
        if re.match(r'^\s*\*\*.+?\*\*\s*$', line):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def normalize_pair_for_display(side_a: str, side_b: str) -> tuple[str, str]:
    """
    Pairwise formatting normalization:
    1) clean repeated 'Proposed Research' headers
    2) remove year from title
    3) if either side has no title line, hide title line on both sides
    """
    clean_a = _clean_proposal_headers_and_title(side_a)
    clean_b = _clean_proposal_headers_and_title(side_b)

    has_title_a = _has_bold_title_line(clean_a)
    has_title_b = _has_bold_title_line(clean_b)
    if not (has_title_a and has_title_b):
        clean_a = _remove_bold_title_line(clean_a)
        clean_b = _remove_bold_title_line(clean_b)

    return clean_a, clean_b


def extract_sections_from_proposal(text: str) -> dict:
    """Extract structured sections from a proposal text."""
    sections = {}

    # Try to extract labeled sections
    patterns = {
        'research_question': r'(?:Research Question|RQ)\s*:\s*(.+?)(?=\n(?:Hypothesis|Proposed Method|Novelty|Experiment|\*\*)|$)',
        'hypothesis': r'Hypothesis\s*:\s*(.+?)(?=\n(?:Proposed Method|Novelty|Experiment|\*\*)|$)',
        'proposed_method': r'Proposed Method\s*:\s*(.+?)(?=\n(?:Novelty|Experiment|\*\*)|$)',
        'novelty_claims': r'Novelty\s*(?:Claims?)?\s*:\s*(.+?)(?=\n(?:Experiment|\*\*)|$)',
        'experiment_details': r'Experiment\s*(?:Details?)?\s*:\s*(.+?)(?=$)',
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            sections[key] = match.group(1).strip()

    return sections


def load_data(
    predictions_path: str,
    strip_reasoning: bool = True,
    seed: int = 42,
    use_llm_readability: bool = True,
    readability_model: str = "gpt-4.1-nano",
    generate_missing_readability: bool = False,
):
    """Load prediction data and pair with ground truth.

    Randomly assigns generated vs real paper to side A or B per pair
    so the annotator cannot tell which is which from position alone.
    """
    with open(predictions_path, 'r') as f:
        data = json.load(f)

    rng = random.Random(seed)

    required_fields = ['Research Question:', 'Hypothesis:', 'Proposed Method:', 'Novelty Claims:', 'Experiment Details:']

    # Cache for LLM readability enhancements to avoid repeated API calls.
    cache_path = str(Path(predictions_path).with_suffix("")) + "_readability_cache.json"
    readability_cache = {}
    if use_llm_readability and os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                readability_cache = json.load(f)
            print(f"Loaded readability cache: {cache_path} ({len(readability_cache)} entries)")
        except Exception:
            readability_cache = {}
    cache_updated = False

    pairs = []
    skipped = 0
    for pred in data.get('predictions', []):
        prediction_text = pred.get('prediction', '')
        ground_truth = pred.get('ground_truth', '')

        # Strip reasoning from prediction if requested
        if strip_reasoning:
            stripped_prediction = strip_reasoning_from_text(prediction_text)
        else:
            stripped_prediction = prediction_text

        # Only keep pairs where BOTH sides have all required fields
        pred_has_all = all(f in stripped_prediction for f in required_fields)
        gt_has_all = all(f in ground_truth for f in required_fields)
        if not (pred_has_all and gt_has_all):
            skipped += 1
            continue

        # Randomly swap sides: generated could be A or B
        generated_is_a = rng.random() < 0.5

        if generated_is_a:
            side_a = stripped_prediction
            side_b = ground_truth
        else:
            side_a = ground_truth
            side_b = stripped_prediction

        side_a, side_b = normalize_pair_for_display(side_a, side_b)

        pair = {
            'id': pred.get('id', len(pairs)),
            'root_title': pred.get('root_title', 'Unknown'),
            'research_question': pred.get('research_question', ''),
            'side_a': side_a,
            'side_b': side_b,
            'generated_is_a': generated_is_a,
            'prompt_mode': pred.get('prompt_mode', 'unknown'),
            'side_a_summary': '',
            'side_a_keywords': [],
            'side_b_summary': '',
            'side_b_keywords': [],
        }

        if use_llm_readability:
            for side_key, sum_key, kw_key in [
                ('side_a', 'side_a_summary', 'side_a_keywords'),
                ('side_b', 'side_b_summary', 'side_b_keywords'),
            ]:
                text = pair[side_key]
                fp = _text_fingerprint(text)
                cached = readability_cache.get(fp)
                if cached:
                    pair[sum_key] = cached.get("summary", "")
                    pair[kw_key] = cached.get("keywords", [])
                    continue
                if generate_missing_readability:
                    try:
                        enh = generate_readability_enhancement(text, model=readability_model)
                        pair[sum_key] = enh.get("summary", "")
                        pair[kw_key] = enh.get("keywords", [])
                        readability_cache[fp] = {
                            "summary": pair[sum_key],
                            "keywords": pair[kw_key],
                        }
                        cache_updated = True
                    except Exception as e:
                        # Graceful fallback if API/key/rate-limit issues occur.
                        print(f"Readability enhancement skipped for one proposal: {e}")

        pairs.append(pair)

    if skipped:
        print(f"Filtered out {skipped} pairs missing sections, kept {len(pairs)}")

    pairs = prioritize_newly_added_pairs(pairs)
    if use_llm_readability and cache_updated:
        try:
            with open(cache_path, "w") as f:
                json.dump(readability_cache, f, indent=2)
            print(f"Saved readability cache: {cache_path} ({len(readability_cache)} entries)")
        except Exception as e:
            print(f"Could not save readability cache: {e}")
    return pairs, data.get('config', {})


def build_html(pairs, config, annotations, predictions_path, annotator_id=None):
    """Build the full HTML page."""
    pred_name = Path(predictions_path).stem
    num_annotated = sum(1 for a in annotations.values() if any(a.get(k) for k in ['soundness', 'excitement', 'overall']))
    annotator_display = annotator_id or 'anonymous'

    # Build pair cards
    pair_cards = []
    for i, pair in enumerate(pairs):
        ann = annotations.get(str(i), {})
        is_annotated = any(ann.get(k) for k in ['soundness', 'excitement', 'overall'])

        side_a_html = markdown_to_html(pair['side_a'])
        side_b_html = markdown_to_html(pair['side_b'])
        side_a_html = highlight_keywords_in_html(side_a_html, pair.get('side_a_keywords', []))
        side_b_html = highlight_keywords_in_html(side_b_html, pair.get('side_b_keywords', []))
        side_a_summary = html_lib.escape(pair.get('side_a_summary', '') or '')
        side_b_summary = html_lib.escape(pair.get('side_b_summary', '') or '')

        def radio_group(pair_idx, dimension):
            val = ann.get(dimension, '')
            options = [
                ('A', 'A is Better'),
                ('tie', 'Tie'),
                ('B', 'B is Better'),
            ]
            radios = ''
            for opt_val, opt_label in options:
                checked = 'checked' if val == opt_val else ''
                color_class = ''
                if opt_val == 'A':
                    color_class = 'radio-a'
                elif opt_val == 'B':
                    color_class = 'radio-b'
                else:
                    color_class = 'radio-tie'
                radios += f'''
                <label class="radio-label {color_class} {'selected' if checked else ''}">
                    <input type="radio" name="{dimension}_{pair_idx}" value="{opt_val}" {checked}
                           onchange="submitRating({pair_idx}, '{dimension}', this.value)">
                    <span>{opt_label}</span>
                </label>'''
            return radios

        card = f'''
        <div class="pair-card {'annotated' if is_annotated else ''}" id="pair-{i}">
            <div class="pair-header">
                <div class="pair-number">#{i+1}</div>
                <div class="pair-title">{html_lib.escape(pair.get('research_question', f'Pair #{i+1}'))}</div>
                <div class="pair-status {'done' if is_annotated else 'pending'}">
                    {'Annotated' if is_annotated else 'Pending'}
                </div>
            </div>
            <div class="columns">
                <div class="column column-a">
                    <div class="column-label">Proposal A</div>
                    {f'<div class="mini-summary"><strong>Summary:</strong> {side_a_summary}</div>' if side_a_summary else ''}
                    <div class="column-content">{side_a_html}</div>
                </div>
                <div class="column column-b">
                    <div class="column-label">Proposal B</div>
                    {f'<div class="mini-summary"><strong>Summary:</strong> {side_b_summary}</div>' if side_b_summary else ''}
                    <div class="column-content">{side_b_html}</div>
                </div>
            </div>
            <div class="rating-bar">
                <div class="rating-group">
                    <div class="rating-dimension">Soundness</div>
                    <div class="rating-hint">Which proposal is more technically sound and internally consistent?</div>
                    <div class="rating-options">{radio_group(i, 'soundness')}</div>
                </div>
                <div class="rating-group">
                    <div class="rating-dimension">Excitement</div>
                    <div class="rating-hint">Which proposal is more exciting / promising as a publishable direction?</div>
                    <div class="rating-options">{radio_group(i, 'excitement')}</div>
                </div>
                <div class="rating-group">
                    <div class="rating-dimension">Overall</div>
                    <div class="rating-hint">If you could only advance one to a serious project proposal, which would you choose?</div>
                    <div class="rating-options">{radio_group(i, 'overall')}</div>
                </div>
            </div>
        </div>'''
        pair_cards.append(card)

    all_cards = '\n'.join(pair_cards)

    page = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Proposal Comparison - {html_lib.escape(annotator_display)}</title>
<style>
:root {{
    --bg: #f7f9fc;
    --surface: #ffffff;
    --surface2: #f3f6fb;
    --border: #d7deea;
    --text: #1f2937;
    --text-dim: #4b5563;
    --accent-a: #3b82f6;
    --accent-b: #ef4444;
    --accent-tie: #8b5cf6;
    --green: #16a34a;
    --yellow: #ca8a04;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
}}
.top-bar {{
    position: sticky;
    top: 0;
    z-index: 100;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 12px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    backdrop-filter: blur(12px);
}}
.top-bar h1 {{
    font-size: 18px;
    font-weight: 600;
}}
.annotator-badge {{
    background: #e0f2fe;
    color: #0369a1;
    padding: 4px 12px;
    border-radius: 16px;
    font-size: 13px;
    font-weight: 500;
}}
.progress {{
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 14px;
    color: var(--text-dim);
}}
.progress-bar {{
    width: 200px;
    height: 6px;
    background: var(--border);
    border-radius: 3px;
    overflow: hidden;
}}
.progress-fill {{
    height: 100%;
    background: var(--green);
    border-radius: 3px;
    transition: width 0.3s;
    width: {num_annotated / max(len(pairs), 1) * 100:.1f}%;
}}
.progress-text {{
    font-variant-numeric: tabular-nums;
}}
.nav-buttons {{
    display: flex;
    gap: 8px;
}}
.nav-buttons button {{
    background: var(--surface2);
    color: var(--text);
    border: 1px solid var(--border);
    padding: 6px 14px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 13px;
    transition: background 0.15s;
}}
.nav-buttons button:hover {{
    background: var(--border);
}}
.container {{
    max-width: 1600px;
    margin: 0 auto;
    padding: 24px;
}}
.pair-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    margin-bottom: 24px;
    overflow: hidden;
    transition: border-color 0.2s;
}}
.pair-card.annotated {{
    border-color: var(--green);
    border-width: 1px;
}}
.pair-header {{
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 16px 20px;
    background: var(--surface2);
    border-bottom: 1px solid var(--border);
}}
.pair-number {{
    background: var(--accent-a);
    color: white;
    font-weight: 700;
    font-size: 13px;
    padding: 3px 10px;
    border-radius: 20px;
    flex-shrink: 0;
}}
.pair-title {{
    font-weight: 500;
    font-size: 14px;
    flex: 1;
    line-height: 1.4;
    color: var(--text-dim);
}}
.pair-status {{
    font-size: 12px;
    padding: 3px 10px;
    border-radius: 20px;
    font-weight: 600;
    flex-shrink: 0;
}}
.pair-status.done {{ background: rgba(74,222,128,0.15); color: var(--green); }}
.pair-status.pending {{ background: rgba(251,191,36,0.15); color: var(--yellow); }}
.comparison-type {{
    font-size: 11px;
    padding: 3px 10px;
    border-radius: 20px;
    font-weight: 600;
    flex-shrink: 0;
}}
.comparison-type.type-human {{ background: rgba(147,51,234,0.15); color: #7c3aed; }}
.comparison-type.type-prompting {{ background: rgba(236,72,153,0.15); color: #db2777; }}
.rq-bar {{
    padding: 12px 20px;
    background: rgba(108,140,255,0.06);
    border-bottom: 1px solid var(--border);
    font-size: 14px;
    color: var(--text-dim);
}}
.rq-bar strong {{ color: var(--text); }}
.columns {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0;
}}
.column {{
    padding: 20px;
    min-height: 200px;
    max-height: 600px;
    overflow-y: auto;
    font-size: 14px;
    line-height: 1.7;
}}
.column-a {{
    border-right: 1px solid var(--border);
}}
.column-label {{
    font-weight: 700;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 12px;
    padding: 4px 10px;
    border-radius: 4px;
    display: inline-block;
}}
.column-a .column-label {{ background: rgba(108,140,255,0.15); color: var(--accent-a); }}
.column-b .column-label {{ background: rgba(255,126,108,0.15); color: var(--accent-b); }}
.column-content {{
    color: var(--text-dim);
}}
.mini-summary {{
    margin: 2px 0 12px;
    padding: 8px 10px;
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent-a);
    border-radius: 6px;
    background: #f9fbff;
    color: var(--text);
    font-size: 13px;
}}
.column-content h3 {{ color: var(--text); margin: 14px 0 6px; font-size: 15px; }}
.column-content h4 {{ color: var(--text); margin: 10px 0 4px; font-size: 14px; }}
.column-content strong {{ color: var(--text); }}
.column-content ul {{ padding-left: 20px; margin: 6px 0; }}
.column-content li {{ margin: 3px 0; }}
.column-content p {{ margin: 6px 0; }}
mark {{
    background: #fff3a3;
    color: #111827;
    padding: 0 2px;
    border-radius: 3px;
}}
.section-divider {{
    border: none;
    border-top: 1px solid var(--border);
    margin: 16px 0 12px;
}}
.field-label {{
    color: var(--accent-a);
    font-weight: 700;
    font-size: 13px;
}}
.rating-bar {{
    display: flex;
    gap: 0;
    border-top: 1px solid var(--border);
    background: var(--surface2);
}}
.rating-group {{
    flex: 1;
    padding: 14px 20px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    border-right: 1px solid var(--border);
}}
.rating-group:last-child {{ border-right: none; }}
.rating-dimension {{
    font-weight: 700;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-dim);
}}
.rating-hint {{
    font-size: 11px;
    color: var(--text-dim);
    line-height: 1.5;
    max-width: 280px;
    text-align: center;
    margin-bottom: 4px;
}}
.rating-options {{
    display: flex;
    gap: 6px;
}}
.radio-label {{
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 5px 12px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 13px;
    border: 1px solid var(--border);
    transition: all 0.15s;
    user-select: none;
}}
.radio-label input {{ display: none; }}
.radio-label:hover {{ border-color: var(--text-dim); }}
.radio-a.selected {{ background: rgba(108,140,255,0.2); border-color: var(--accent-a); color: var(--accent-a); }}
.radio-b.selected {{ background: rgba(255,126,108,0.2); border-color: var(--accent-b); color: var(--accent-b); }}
.radio-tie.selected {{ background: rgba(167,139,250,0.2); border-color: var(--accent-tie); color: var(--accent-tie); }}

/* Toast notification */
.toast {{
    position: fixed;
    bottom: 24px;
    right: 24px;
    background: var(--green);
    color: #ffffff;
    padding: 10px 20px;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 600;
    opacity: 0;
    transform: translateY(10px);
    transition: all 0.3s;
    z-index: 200;
    pointer-events: none;
}}
.toast.show {{
    opacity: 1;
    transform: translateY(0);
}}

/* Scrollbar */
.column::-webkit-scrollbar {{ width: 6px; }}
.column::-webkit-scrollbar-track {{ background: transparent; }}
.column::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; }}
.column::-webkit-scrollbar-thumb:hover {{ background: var(--text-dim); }}

/* Export button */
.export-btn {{
    background: var(--green);
    color: #ffffff;
    border: none;
    padding: 6px 14px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 13px;
    font-weight: 600;
    transition: opacity 0.15s;
}}
.export-btn:hover {{ opacity: 0.85; }}
</style>
</head>
<body>

<div class="top-bar">
    <h1>Research Proposal Comparison</h1>
    <span class="annotator-badge">Annotator: {html_lib.escape(annotator_display)}</span>
    <div class="progress">
        <div class="progress-bar"><div class="progress-fill" id="progress-fill"></div></div>
        <span class="progress-text" id="progress-text">{num_annotated}/{len(pairs)} annotated</span>
    </div>
    <div class="nav-buttons">
        <button onclick="jumpToNext()">Jump to Next Unannotated</button>
        <button class="export-btn" onclick="exportAnnotations()">Export Results</button>
    </div>
</div>

<div class="container" id="container">
{all_cards}
</div>

<div class="toast" id="toast">Saved!</div>

<script>
const totalPairs = {len(pairs)};
const annotatorId = '{annotator_display}';

function submitRating(pairIdx, dimension, value) {{
    // Update UI immediately
    const radios = document.querySelectorAll(`input[name="${{dimension}}_${{pairIdx}}"]`);
    radios.forEach(r => {{
        const label = r.closest('.radio-label');
        if (r.value === value) {{
            label.classList.add('selected');
        }} else {{
            label.classList.remove('selected');
        }}
    }});

    // Send to server
    fetch('/api/rate', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{pair_id: pairIdx, dimension: dimension, value: value, annotator: annotatorId}})
    }}).then(r => r.json()).then(data => {{
        // Update card status
        const card = document.getElementById(`pair-${{pairIdx}}`);
        if (data.is_complete) {{
            card.classList.add('annotated');
            card.querySelector('.pair-status').className = 'pair-status done';
            card.querySelector('.pair-status').textContent = 'Annotated';
        }}
        // Update progress
        document.getElementById('progress-text').textContent = `${{data.num_annotated}}/${{totalPairs}} annotated`;
        document.getElementById('progress-fill').style.width = `${{data.num_annotated / totalPairs * 100}}%`;
        // Toast
        showToast('Saved!');
    }});
}}

function showToast(msg) {{
    const toast = document.getElementById('toast');
    toast.textContent = msg;
    toast.classList.add('show');
    setTimeout(() => toast.classList.remove('show'), 1500);
}}

function jumpToNext() {{
    const cards = document.querySelectorAll('.pair-card:not(.annotated)');
    if (cards.length > 0) {{
        cards[0].scrollIntoView({{ behavior: 'smooth', block: 'start' }});
    }} else {{
        showToast('All annotated!');
    }}
}}

function exportAnnotations() {{
    window.open('/api/export?annotator=' + encodeURIComponent(annotatorId), '_blank');
}}
</script>
</body>
</html>'''
    return page


class ComparisonHandler(SimpleHTTPRequestHandler):
    """HTTP handler for the comparison tool."""

    pairs = []
    config = {}
    annotations_by_annotator = {}  # {annotator_id: {pair_id: {dimension: value}}}
    annotations_dir = ''
    predictions_path = ''
    default_annotator = None  # Set via --annotator flag

    @classmethod
    def get_annotations_path(cls, annotator_id):
        """Get the annotations file path for a given annotator."""
        base = Path(cls.predictions_path).stem
        # Save to annotations/ subdirectory
        ann_subdir = os.path.join(cls.annotations_dir, "annotations")
        os.makedirs(ann_subdir, exist_ok=True)
        return os.path.join(ann_subdir, f"{base}_annotations_{annotator_id}.json")

    @classmethod
    def load_annotator_data(cls, annotator_id):
        """Load annotations for a specific annotator."""
        if annotator_id not in cls.annotations_by_annotator:
            ann_path = cls.get_annotations_path(annotator_id)
            if os.path.exists(ann_path):
                with open(ann_path, 'r') as f:
                    cls.annotations_by_annotator[annotator_id] = json.load(f)
            else:
                cls.annotations_by_annotator[annotator_id] = {}
        return cls.annotations_by_annotator[annotator_id]

    @classmethod
    def save_annotator_data(cls, annotator_id):
        """Save annotations for a specific annotator."""
        ann_path = cls.get_annotations_path(annotator_id)
        with open(ann_path, 'w') as f:
            json.dump(cls.annotations_by_annotator.get(annotator_id, {}), f, indent=2)

    def do_GET(self):
        parsed = urlparse(self.path)
        query_params = parse_qs(parsed.query)
        
        # Get annotator from URL or use default
        annotator_id = query_params.get('annotator', [self.default_annotator])[0]

        if parsed.path == '/' or parsed.path == '':
            if not annotator_id:
                # Show annotator selection page
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.end_headers()
                page = self._build_annotator_prompt()
                self.wfile.write(page.encode('utf-8'))
                return
            
            annotations = self.load_annotator_data(annotator_id)
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            page = build_html(
                self.pairs, self.config, annotations, self.predictions_path, annotator_id
            )
            self.wfile.write(page.encode('utf-8'))

        elif parsed.path == '/api/export':
            if not annotator_id:
                self.send_response(400)
                self.end_headers()
                return
            annotations = self.load_annotator_data(annotator_id)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Disposition',
                             f'attachment; filename="annotations_{Path(self.predictions_path).stem}_{annotator_id}.json"')
            self.end_headers()
            export = {
                'config': self.config,
                'predictions_file': self.predictions_path,
                'annotator': annotator_id,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'annotations': [],
            }
            for i, pair in enumerate(self.pairs):
                ann = annotations.get(str(i), {})
                export['annotations'].append({
                    'pair_id': i,
                    'root_title': pair['root_title'],
                    'research_question': pair['research_question'],
                    'generated_is_a': pair['generated_is_a'],
                    'soundness': ann.get('soundness', ''),
                    'excitement': ann.get('excitement', ''),
                    'overall': ann.get('overall', ''),
                })
            self.wfile.write(json.dumps(export, indent=2).encode('utf-8'))

        else:
            self.send_response(404)
            self.end_headers()

    def _build_annotator_prompt(self):
        """Build a simple page asking for annotator name."""
        return '''<!DOCTYPE html>
<html><head><title>Enter Your Name</title>
<style>
body { font-family: -apple-system, sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; background: #f7f9fc; }
.box { background: white; padding: 40px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); text-align: center; }
h1 { margin: 0 0 20px; font-size: 24px; }
input { padding: 12px 16px; font-size: 16px; border: 1px solid #ddd; border-radius: 8px; width: 200px; }
button { padding: 12px 24px; font-size: 16px; background: #3b82f6; color: white; border: none; border-radius: 8px; cursor: pointer; margin-top: 16px; }
button:hover { background: #2563eb; }
</style>
</head><body>
<div class="box">
<h1>Research Proposal Annotation</h1>
<p>Please enter your name to start:</p>
<input type="text" id="name" placeholder="Your name" onkeypress="if(event.key==='Enter')go()">
<br><button onclick="go()">Start Annotation</button>
</div>
<script>
function go() {
    var name = document.getElementById('name').value.trim().toLowerCase().replace(/\\s+/g, '_');
    if (name) window.location.href = '/?annotator=' + encodeURIComponent(name);
}
</script>
</body></html>'''

    def do_POST(self):
        if self.path == '/api/rate':
            content_len = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_len)
            data = json.loads(body)

            annotator_id = data.get('annotator')
            if not annotator_id:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(b'{"error": "Missing annotator"}')
                return

            pair_id = str(data['pair_id'])
            dimension = data['dimension']
            value = data['value']

            # Load and update annotations
            annotations = self.load_annotator_data(annotator_id)
            if pair_id not in annotations:
                annotations[pair_id] = {}
            annotations[pair_id][dimension] = value
            
            # Save to file
            self.save_annotator_data(annotator_id)

            # Check if this pair is fully annotated
            ann = annotations[pair_id]
            is_complete = all(ann.get(k) for k in ['soundness', 'excitement', 'overall'])

            # Count total annotated
            num_annotated = sum(
                1 for a in annotations.values()
                if all(a.get(k) for k in ['soundness', 'excitement', 'overall'])
            )

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                'ok': True,
                'is_complete': is_complete,
                'num_annotated': num_annotated,
            }).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        """Suppress default logging for clean output."""
        pass


NLP_AREA_PATTERNS = [
    'large_language_models',
    'foundation or frontier models',
    'language_speech_and_dialog',
    'natural language',
]

NLP_KEYWORD_TERMS = [
    'language model', 'llm', 'nlp', 'text', 'translation', 
    'question answering', 'summarization', 'dialogue', 'sentiment',
    'named entity', 'transformer', 'bert', 'gpt', 'instruction',
    'prompt', 'reasoning', 'retrieval', 'rag', 'knowledge graph',
]

MULTIMODAL_CV_AREA_PATTERNS = [
    'computer vision',
    'multimodal',
    'vision',
    'image',
    'video',
]

# Papers that have been verified as NOT NLP/CV by GPT-4.1
EXCLUDED_NON_NLP_CV_PAPERS = [
    'FlowDec: A flow-based full-band general audio codec with high perceptual quality',
    'Interactive Speculative Planning: Enhance Agent Efficiency through Co-design of System and User Interface',
    'Near-Optimal Sample Complexity for MDPs via Anchoring',
    'Learning Task-Agnostic Representations through Multi-Teacher Distillation',
    'Faster Fixed-Point Methods for Multichain MDPs',
]

def prepare_human_annotation_data(
    stepwise_predictions_path: str,
    prompting_predictions_path: str,
    subset_path: str,
    sampled_papers_path: str,
    output_path: str,
    num_samples: int = 60,
    nlp_ratio: float = 0.7,
    seed: int = 42,
    strip_reasoning: bool = True,
):
    """
    Prepare comparison data for human annotation.
    
    Creates pairs for:
    1. stepwise-cot vs human (ground truth)
    2. stepwise-cot vs prompting (untuned model)
    
    Filters for NLP papers with some multimodal/CV papers.
    """
    import random
    rng = random.Random(seed)
    
    print(f"Loading stepwise-cot predictions from {stepwise_predictions_path}")
    with open(stepwise_predictions_path) as f:
        stepwise_data = json.load(f)
    stepwise_preds = {p['root_title']: p for p in stepwise_data.get('predictions', [])}
    print(f"  Loaded {len(stepwise_preds)} stepwise-cot predictions")
    
    print(f"Loading prompting predictions from {prompting_predictions_path}")
    with open(prompting_predictions_path) as f:
        prompting_data = json.load(f)
    prompting_preds = {p['root_title']: p for p in prompting_data.get('predictions', [])}
    print(f"  Loaded {len(prompting_preds)} prompting predictions")
    
    print(f"Loading 35% subset from {subset_path}")
    with open(subset_path) as f:
        subset_data = json.load(f)
    selected_titles = set(subset_data.get('selected_titles', []))
    print(f"  Subset contains {len(selected_titles)} titles")
    
    print(f"Loading sampled papers metadata from {sampled_papers_path}")
    with open(sampled_papers_path) as f:
        sampled_data = json.load(f)
    papers_by_title = {p['title']: p for p in sampled_data.get('papers', [])}
    print(f"  Loaded {len(papers_by_title)} papers with metadata")
    
    required_sections = ['research question', 'hypothesis', 'proposed method', 'novelty']
    
    def has_all_fields(text):
        """Check if text has required proposal sections (flexible matching)."""
        text_lower = text.lower()
        for section in required_sections:
            # Match patterns like "Research Question:", "### Research Question", "**Research Question**"
            if section not in text_lower:
                return False
        return True
    
    def get_area_category(paper):
        """Return 'nlp', 'multimodal_cv', or 'other'."""
        area = (paper.get('primary_area') or '').lower()
        keywords = [k.lower() for k in (paper.get('keywords') or [])]
        title = (paper.get('title') or '').lower()
        
        # Check NLP by primary area
        for pattern in NLP_AREA_PATTERNS:
            if pattern.lower() in area:
                return 'nlp'
        
        # Check NLP by keywords or title
        for term in NLP_KEYWORD_TERMS:
            for kw in keywords:
                if term in kw:
                    return 'nlp'
            if term in title:
                return 'nlp'
        
        # Check multimodal/CV by primary area
        for pattern in MULTIMODAL_CV_AREA_PATTERNS:
            if pattern.lower() in area:
                return 'multimodal_cv'
        
        # Check multimodal/CV by keywords or title
        for kw in keywords:
            if any(term in kw for term in ['vision', 'image', 'video', 'multimodal', 'visual']):
                return 'multimodal_cv'
        if any(term in title for term in ['vision', 'image', 'video', 'multimodal', 'visual']):
            return 'multimodal_cv'
        
        return 'other'
    
    nlp_candidates = []
    mm_cv_candidates = []
    excluded_count = 0
    
    for title in selected_titles:
        if title not in stepwise_preds or title not in prompting_preds:
            continue
        
        # Skip papers verified as non-NLP/CV
        if title in EXCLUDED_NON_NLP_CV_PAPERS:
            excluded_count += 1
            continue
        
        stepwise_pred = stepwise_preds[title]
        prompting_pred = prompting_preds[title]
        
        stepwise_text = stepwise_pred.get('prediction', '')
        prompting_text = prompting_pred.get('prediction', '')
        ground_truth = stepwise_pred.get('ground_truth', '')
        
        if strip_reasoning:
            stepwise_text = strip_reasoning_from_text(stepwise_text)
            prompting_text = strip_reasoning_from_text(prompting_text)
        
        if not (has_all_fields(stepwise_text) and has_all_fields(prompting_text) and has_all_fields(ground_truth)):
            continue
        
        paper_meta = papers_by_title.get(title, {})
        area_cat = get_area_category(paper_meta)
        
        candidate = {
            'title': title,
            'stepwise_pred': stepwise_pred,
            'prompting_pred': prompting_pred,
            'stepwise_text': stepwise_text,
            'prompting_text': prompting_text,
            'ground_truth': ground_truth,
            'research_question': stepwise_pred.get('research_question', ''),
            'area_category': area_cat,
            'primary_area': paper_meta.get('primary_area', ''),
            'keywords': paper_meta.get('keywords', []),
        }
        
        if area_cat == 'nlp':
            nlp_candidates.append(candidate)
        elif area_cat == 'multimodal_cv':
            mm_cv_candidates.append(candidate)
    
    if excluded_count:
        print(f"\nExcluded {excluded_count} papers verified as non-NLP/CV")
    print(f"Found {len(nlp_candidates)} NLP candidates, {len(mm_cv_candidates)} multimodal/CV candidates")
    
    num_nlp = int(num_samples * nlp_ratio)
    num_mm_cv = num_samples - num_nlp
    
    if len(nlp_candidates) < num_nlp:
        print(f"  Warning: Only {len(nlp_candidates)} NLP candidates available, adjusting...")
        num_nlp = len(nlp_candidates)
        num_mm_cv = min(num_samples - num_nlp, len(mm_cv_candidates))
    
    if len(mm_cv_candidates) < num_mm_cv:
        print(f"  Warning: Only {len(mm_cv_candidates)} multimodal/CV candidates available, adjusting...")
        num_mm_cv = len(mm_cv_candidates)
        num_nlp = min(num_samples - num_mm_cv, len(nlp_candidates))
    
    rng.shuffle(nlp_candidates)
    rng.shuffle(mm_cv_candidates)
    
    selected = nlp_candidates[:num_nlp] + mm_cv_candidates[:num_mm_cv]
    rng.shuffle(selected)
    
    print(f"\nSelected {len(selected)} data points ({num_nlp} NLP, {num_mm_cv} multimodal/CV)")
    
    pairs = []
    for idx, cand in enumerate(selected):
        generated_is_a_vs_human = rng.random() < 0.5
        generated_is_a_vs_prompting = rng.random() < 0.5
        
        stepwise_clean, gt_clean = normalize_pair_for_display(cand['stepwise_text'], cand['ground_truth'])
        stepwise_clean2, prompting_clean = normalize_pair_for_display(cand['stepwise_text'], cand['prompting_text'])
        
        pair_vs_human = {
            'id': f'{idx}_vs_human',
            'data_point_id': idx,
            'comparison_type': 'stepwise_vs_human',
            'root_title': cand['title'],
            'research_question': cand['research_question'],
            'side_a': stepwise_clean if generated_is_a_vs_human else gt_clean,
            'side_b': gt_clean if generated_is_a_vs_human else stepwise_clean,
            'generated_is_a': generated_is_a_vs_human,
            'side_a_label': 'AI-Generated' if generated_is_a_vs_human else 'Human',
            'side_b_label': 'Human' if generated_is_a_vs_human else 'AI-Generated',
            'area_category': cand['area_category'],
            'primary_area': cand['primary_area'],
        }
        
        pair_vs_prompting = {
            'id': f'{idx}_vs_prompting',
            'data_point_id': idx,
            'comparison_type': 'stepwise_vs_prompting',
            'root_title': cand['title'],
            'research_question': cand['research_question'],
            'side_a': stepwise_clean2 if generated_is_a_vs_prompting else prompting_clean,
            'side_b': prompting_clean if generated_is_a_vs_prompting else stepwise_clean2,
            'stepwise_is_a': generated_is_a_vs_prompting,
            'side_a_label': 'Stepwise-CoT' if generated_is_a_vs_prompting else 'Prompting',
            'side_b_label': 'Prompting' if generated_is_a_vs_prompting else 'Stepwise-CoT',
            'area_category': cand['area_category'],
            'primary_area': cand['primary_area'],
        }
        
        pairs.append(pair_vs_human)
        pairs.append(pair_vs_prompting)
    
    output_data = {
        'config': {
            'stepwise_predictions': stepwise_predictions_path,
            'prompting_predictions': prompting_predictions_path,
            'subset_path': subset_path,
            'num_data_points': len(selected),
            'num_pairs': len(pairs),
            'nlp_count': num_nlp,
            'multimodal_cv_count': num_mm_cv,
            'seed': seed,
        },
        'pairs': pairs,
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nSaved {len(pairs)} comparison pairs to {output_path}")
    print(f"  - {len(selected)} data points")
    print(f"  - {len(pairs) // 2} stepwise vs human pairs")
    print(f"  - {len(pairs) // 2} stepwise vs prompting pairs")
    
    area_counts = {}
    for cand in selected:
        area = cand['area_category']
        area_counts[area] = area_counts.get(area, 0) + 1
    print(f"\nArea distribution:")
    for area, count in sorted(area_counts.items()):
        print(f"  {area}: {count}")
    
    return output_data


def load_human_annotation_data(
    comparison_data_path: str,
    seed: int = 42,
    use_llm_readability: bool = True,
    readability_model: str = "gpt-4.1",
    generate_missing_readability: bool = False,
):
    """Load comparison pairs prepared for human annotation with optional LLM readability enhancement."""
    with open(comparison_data_path) as f:
        data = json.load(f)
    
    pairs = data.get('pairs', [])
    config = data.get('config', {})
    
    # Cache for LLM readability enhancements
    cache_path = str(Path(comparison_data_path).with_suffix("")) + "_readability_cache.json"
    readability_cache = {}
    if use_llm_readability and os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                readability_cache = json.load(f)
            print(f"Loaded readability cache: {cache_path} ({len(readability_cache)} entries)")
        except Exception:
            readability_cache = {}
    cache_updated = False
    
    formatted_pairs = []
    for pair in pairs:
        fp = {
            'id': pair['id'],
            'root_title': pair.get('root_title', 'Unknown'),
            'research_question': pair.get('research_question', ''),
            'side_a': pair.get('side_a', ''),
            'side_b': pair.get('side_b', ''),
            'comparison_type': pair.get('comparison_type', ''),
            'side_a_label': pair.get('side_a_label', 'A'),
            'side_b_label': pair.get('side_b_label', 'B'),
            'generated_is_a': pair.get('generated_is_a', pair.get('stepwise_is_a', False)),
            'prompt_mode': 'human_annotation',
            'side_a_summary': '',
            'side_a_keywords': [],
            'side_b_summary': '',
            'side_b_keywords': [],
        }
        
        if use_llm_readability:
            for side_key, sum_key, kw_key in [
                ('side_a', 'side_a_summary', 'side_a_keywords'),
                ('side_b', 'side_b_summary', 'side_b_keywords'),
            ]:
                text = fp[side_key]
                fingerprint = _text_fingerprint(text)
                cached = readability_cache.get(fingerprint)
                if cached:
                    fp[sum_key] = cached.get("summary", "")
                    fp[kw_key] = cached.get("keywords", [])
                    continue
                if generate_missing_readability:
                    try:
                        enh = generate_readability_enhancement(text, model=readability_model)
                        fp[sum_key] = enh.get("summary", "")
                        fp[kw_key] = enh.get("keywords", [])
                        readability_cache[fingerprint] = {
                            "summary": fp[sum_key],
                            "keywords": fp[kw_key],
                        }
                        cache_updated = True
                    except Exception as e:
                        print(f"Readability enhancement skipped: {e}")
        
        formatted_pairs.append(fp)
    
    if use_llm_readability and cache_updated:
        try:
            with open(cache_path, "w") as f:
                json.dump(readability_cache, f, indent=2)
            print(f"Saved readability cache: {cache_path} ({len(readability_cache)} entries)")
        except Exception as e:
            print(f"Could not save readability cache: {e}")
    
    return formatted_pairs, config


def split_annotation_batches(
    comparison_data_path: str,
    pairs_per_batch: int = 30,
    output_dir: str = None,
):
    """
    Split comparison data into batches for multiple annotators.
    
    Args:
        comparison_data_path: Path to the comparison JSON file
        pairs_per_batch: Number of pairs per batch (default 30)
        output_dir: Output directory for batch files (default: same as input)
    """
    with open(comparison_data_path) as f:
        data = json.load(f)
    
    pairs = data.get('pairs', [])
    config = data.get('config', {})
    
    if output_dir is None:
        output_dir = str(Path(comparison_data_path).parent)
    
    num_batches = (len(pairs) + pairs_per_batch - 1) // pairs_per_batch
    
    print(f"Splitting {len(pairs)} pairs into {num_batches} batches of ~{pairs_per_batch} pairs each")
    
    # Also load the readability cache to split it
    cache_path = str(Path(comparison_data_path).with_suffix("")) + "_readability_cache.json"
    readability_cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            readability_cache = json.load(f)
        print(f"Loaded readability cache with {len(readability_cache)} entries")
    
    batch_files = []
    for batch_idx in range(num_batches):
        start = batch_idx * pairs_per_batch
        end = min(start + pairs_per_batch, len(pairs))
        batch_pairs = pairs[start:end]
        
        batch_letter = chr(ord('A') + batch_idx)
        batch_filename = f"comparison_batch_{batch_letter}.json"
        batch_path = os.path.join(output_dir, batch_filename)
        
        batch_data = {
            'config': {
                **config,
                'batch': batch_letter,
                'batch_start': start,
                'batch_end': end,
                'pairs_in_batch': len(batch_pairs),
            },
            'pairs': batch_pairs,
        }
        
        with open(batch_path, 'w') as f:
            json.dump(batch_data, f, indent=2)
        
        # Create batch-specific readability cache
        if readability_cache:
            batch_cache = {}
            for pair in batch_pairs:
                for side in ['side_a', 'side_b']:
                    fp = _text_fingerprint(pair[side])
                    if fp in readability_cache:
                        batch_cache[fp] = readability_cache[fp]
            
            batch_cache_path = os.path.join(output_dir, f"comparison_batch_{batch_letter}_readability_cache.json")
            with open(batch_cache_path, 'w') as f:
                json.dump(batch_cache, f, indent=2)
        
        batch_files.append(batch_path)
        print(f"  Batch {batch_letter}: pairs {start+1}-{end} ({len(batch_pairs)} pairs) -> {batch_filename}")
    
    print(f"\nCreated {num_batches} batch files in {output_dir}")
    print("\nTo serve each batch:")
    for i, bf in enumerate(batch_files):
        batch_letter = chr(ord('A') + i)
        port = 8900 + i
        print(f"  python human_evaluation/serve_comparison.py --comparison-data {bf} --port {port}")
    
    return batch_files


def main():
    parser = argparse.ArgumentParser(description="Serve comparison UI for proposal evaluation")
    parser.add_argument(
        '--predictions',
        default='predictions/qwen-14b-stepwise-cot.json',
        help='Path to predictions JSON file'
    )
    parser.add_argument(
        '--port', type=int, default=8899,
        help='Port to serve on (default: 8899)'
    )
    parser.add_argument(
        '--annotator',
        default=None,
        help='Annotator ID (creates separate annotation file per annotator, e.g., --annotator alice)'
    )
    parser.add_argument(
        '--no-strip-reasoning', action='store_true',
        help='Do not strip reasoning from predictions'
    )
    parser.add_argument(
        '--annotations',
        default=None,
        help='Path to annotations JSON (default: auto-generated next to predictions)'
    )
    parser.add_argument(
        '--no-llm-readability',
        action='store_true',
        help='Disable LLM-generated short summaries and keyword highlighting'
    )
    parser.add_argument(
        '--readability-model',
        default='gpt-4.1-nano',
        help='Model for readability enhancement (default: gpt-4.1-nano)'
    )
    parser.add_argument(
        '--generate-missing-readability',
        action='store_true',
        help='Generate missing readability cache entries during startup (slower)'
    )
    parser.add_argument(
        '--precompute-readability',
        action='store_true',
        help='Precompute readability cache and exit (recommended before serving)'
    )
    
    # Human annotation mode arguments
    parser.add_argument(
        '--prepare-annotation-data',
        action='store_true',
        help='Prepare comparison data for human annotation and exit'
    )
    parser.add_argument(
        '--stepwise-predictions',
        default='predictions/qwen-14b-stepwise-cot.json',
        help='Path to stepwise-cot predictions (for --prepare-annotation-data)'
    )
    parser.add_argument(
        '--prompting-predictions',
        default='predictions/qwen-14b-base.json',
        help='Path to prompting/untuned predictions (for --prepare-annotation-data)'
    )
    parser.add_argument(
        '--subset-path',
        default='evaluation/selected_subset_35pct.json',
        help='Path to subset selection file (for --prepare-annotation-data)'
    )
    parser.add_argument(
        '--sampled-papers',
        default='data/test_set/sampled_papers_3000.json',
        help='Path to sampled papers metadata (for --prepare-annotation-data)'
    )
    parser.add_argument(
        '--output-comparison-data',
        default='human_evaluation/comparison_human_annotation.json',
        help='Output path for comparison data (for --prepare-annotation-data)'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=60,
        help='Number of data points to select (for --prepare-annotation-data)'
    )
    parser.add_argument(
        '--nlp-ratio',
        type=float,
        default=0.7,
        help='Ratio of NLP papers vs multimodal/CV (for --prepare-annotation-data)'
    )
    parser.add_argument(
        '--comparison-data',
        default=None,
        help='Path to prepared comparison data JSON (alternative to --predictions)'
    )

    args = parser.parse_args()

    # Handle annotation data preparation mode
    if args.prepare_annotation_data:
        prepare_human_annotation_data(
            stepwise_predictions_path=args.stepwise_predictions,
            prompting_predictions_path=args.prompting_predictions,
            subset_path=args.subset_path,
            sampled_papers_path=args.sampled_papers,
            output_path=args.output_comparison_data,
            num_samples=args.num_samples,
            nlp_ratio=args.nlp_ratio,
            strip_reasoning=not args.no_strip_reasoning,
        )
        return

    if args.precompute_readability:
        if args.comparison_data:
            print(f"Precomputing readability cache from {args.comparison_data}...")
            pairs, _ = load_human_annotation_data(
                args.comparison_data,
                use_llm_readability=True,
                readability_model=args.readability_model,
                generate_missing_readability=True,
            )
        else:
            print(f"Precomputing readability cache from {args.predictions}...")
            pairs, _ = load_data(
                args.predictions,
                strip_reasoning=not args.no_strip_reasoning,
                use_llm_readability=not args.no_llm_readability,
                readability_model=args.readability_model,
                generate_missing_readability=True,
            )
        print(f"Done. Processed {len(pairs)} pairs.")
        return

    # Load data - either from comparison data file or predictions file
    if args.comparison_data:
        print(f"Loading comparison data from {args.comparison_data}...")
        pairs, config = load_human_annotation_data(
            args.comparison_data,
            use_llm_readability=not args.no_llm_readability,
            readability_model=args.readability_model,
            generate_missing_readability=args.generate_missing_readability,
        )
        print(f"Loaded {len(pairs)} pairs")
        # Use comparison data path for annotations
        args.predictions = args.comparison_data
    else:
        print(f"Loading predictions from {args.predictions}...")
        pairs, config = load_data(
            args.predictions,
            strip_reasoning=not args.no_strip_reasoning,
            use_llm_readability=not args.no_llm_readability,
            readability_model=args.readability_model,
            generate_missing_readability=args.generate_missing_readability,
        )
        print(f"Loaded {len(pairs)} pairs")

    # Annotations directory - use same directory as predictions file
    annotations_dir = str(Path(args.predictions).parent)
    
    # Set class variables
    ComparisonHandler.pairs = pairs
    ComparisonHandler.config = config
    ComparisonHandler.annotations_dir = annotations_dir
    ComparisonHandler.predictions_path = args.predictions
    ComparisonHandler.default_annotator = args.annotator  # May be None
    
    # If default annotator specified, preload their data
    if args.annotator:
        ann_path = ComparisonHandler.get_annotations_path(args.annotator)
        ComparisonHandler.load_annotator_data(args.annotator)
        num_done = sum(1 for a in ComparisonHandler.annotations_by_annotator.get(args.annotator, {}).values()
                       if all(a.get(k) for k in ['soundness', 'excitement', 'overall']))
        if num_done > 0:
            print(f"Loaded {num_done} existing annotations for {args.annotator}")

    server = HTTPServer(('0.0.0.0', args.port), ComparisonHandler)
    print(f"\n{'='*60}")
    print(f"  Comparison UI running at http://localhost:{args.port}")
    if args.annotator:
        print(f"  Default annotator: {args.annotator} (direct access)")
    else:
        print(f"  Multi-annotator mode: users enter name via URL")
        print(f"  Example: http://localhost:{args.port}?annotator=alice")
    print(f"  Predictions: {args.predictions}")
    print(f"  Annotations dir: {annotations_dir}/annotations/")
    print(f"  Pairs: {len(pairs)}")
    print(f"{'='*60}\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == '__main__':
    main()
