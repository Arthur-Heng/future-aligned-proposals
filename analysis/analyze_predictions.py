#!/usr/bin/env python3
"""
Analyze prediction files for statistics and repetition patterns.

Usage:
    python analyze_predictions.py predictions/llama-8b-stepwise-cot.json
    python analyze_predictions.py predictions/*.json  # Compare multiple files
"""

import json, re, argparse, numpy as np
from pathlib import Path
from collections import Counter


def strip_reasoning(text):
    """Extract only the proposal content, removing reasoning process."""
    if not text:
        return text
    
    def _strip_steps(t):
        return re.sub(
            r'###\s*Step\s*\d+[^\n]*\n'
            r'(?:(?!##\s|###\s*(?!Step)|Research Question:|Hypothesis:|Proposed Method:|Novelty Claims?:|Experiment Details?:).*\n)*',
            '', t)
    
    if len(re.findall(r'###\s*Step\s*\d', text)) >= 2:
        c = _strip_steps(text)
        m = re.search(r'(##\s*Proposed Research|##\s*Research Question|\*\*Research Question)', c, re.I)
        if m:
            c = c[m.start():]
        if c.strip():
            return c.strip()
    
    for p in [r'##\s*Proposed Research\s*(?:Idea)?', r'##\s*Proposal', r'\*\*Proposed Research']:
        m = re.search(p, text, re.I)
        if m:
            return _strip_steps(text[m.end():]).strip()
    return text


def find_repeats(text, min_len=30):
    """Find sentences that appear more than once."""
    sents = re.split(r'(?<=[.!?])\s+', text)
    cnts = Counter(s.strip().lower() for s in sents if len(s.strip()) >= min_len)
    return {s: c for s, c in cnts.items() if c > 1}


def analyze(fp):
    """Analyze a single prediction file."""
    data = json.load(open(fp))
    preds = data.get('predictions', [])
    if not preds:
        return None
    
    r = {
        'file': Path(fp).name, 'n': len(preds),
        'raw': [], 'strip': [],
        'rep': 0, 'excess': 0, 'examples': [],
        'sec': {'rq': 0, 'hyp': 0, 'meth': 0, 'nov': 0, 'exp': 0}
    }
    
    for i, p in enumerate(preds):
        raw = p.get('prediction', '')
        s = strip_reasoning(raw)
        r['raw'].append(len(raw))
        r['strip'].append(len(s))
        
        # Check sections
        if 'Research Question' in s: r['sec']['rq'] += 1
        if 'Hypothesis' in s: r['sec']['hyp'] += 1
        if 'Proposed Method' in s: r['sec']['meth'] += 1
        if 'Novelty Claim' in s: r['sec']['nov'] += 1
        if 'Experiment Detail' in s: r['sec']['exp'] += 1
        
        # Check repetition
        reps = find_repeats(s)
        if reps:
            r['rep'] += 1
            r['excess'] += sum(reps.values()) - len(reps)
            mx = max(reps.values())
            if mx >= 3 and len(r['examples']) < 3:
                r['examples'].append({
                    'id': i, 'mx': mx,
                    's': max(reps, key=lambda k: reps[k])[:60]
                })
    return r


def show(r):
    """Print analysis results."""
    n = r['n']
    print(f"\n{'='*60}")
    print(f"File: {r['file']}")
    print(f"{'='*60}")
    
    print(f"\n--- Length Statistics ---")
    print(f"{'':20}{'Raw':>12}{'Stripped':>12}")
    print(f"{'Mean':20}{np.mean(r['raw']):>12.0f}{np.mean(r['strip']):>12.0f}")
    print(f"{'Median':20}{np.median(r['raw']):>12.0f}{np.median(r['strip']):>12.0f}")
    print(f"{'Min':20}{min(r['raw']):>12}{min(r['strip']):>12}")
    print(f"{'Max':20}{max(r['raw']):>12}{max(r['strip']):>12}")
    
    print(f"\n--- Sections ({n} predictions) ---")
    print(f"  Research Question: {r['sec']['rq']:>4}/{n} ({100*r['sec']['rq']/n:.1f}%)")
    print(f"  Hypothesis:        {r['sec']['hyp']:>4}/{n} ({100*r['sec']['hyp']/n:.1f}%)")
    print(f"  Proposed Method:   {r['sec']['meth']:>4}/{n} ({100*r['sec']['meth']/n:.1f}%)")
    print(f"  Novelty Claims:    {r['sec']['nov']:>4}/{n} ({100*r['sec']['nov']/n:.1f}%)")
    print(f"  Experiment Details:{r['sec']['exp']:>4}/{n} ({100*r['sec']['exp']/n:.1f}%)")
    
    print(f"\n--- Repetition Analysis ---")
    print(f"  With repeated sentences: {r['rep']}/{n} ({100*r['rep']/n:.1f}%)")
    print(f"  Total excess repeated: {r['excess']}")
    
    if r['examples']:
        print(f"\n  Examples:")
        for ex in r['examples']:
            print(f"    Pred {ex['id']}: {ex['mx']}x \"{ex['s']}...\"")


def compare(rs):
    """Print comparison table."""
    if len(rs) < 2:
        return
    
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    
    print(f"\n{'Metric':25}", end='')
    for r in rs:
        print(f"{r['file'][:18]:>20}", end='')
    print()
    print('-' * (25 + 20 * len(rs)))
    
    rows = [
        ('N predictions', lambda x: x['n']),
        ('Mean raw len', lambda x: np.mean(x['raw'])),
        ('Mean stripped len', lambda x: np.mean(x['strip'])),
        ('Stripping reduction %', lambda x: 100*(1-np.mean(x['strip'])/np.mean(x['raw']))),
        ('With repetition %', lambda x: 100*x['rep']/x['n']),
    ]
    
    for nm, fn in rows:
        print(f"{nm:25}", end='')
        for r in rs:
            v = fn(r)
            print(f"{v:>20.1f}" if isinstance(v, float) else f"{v:>20}", end='')
        print()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Analyze prediction files")
    p.add_argument("files", nargs="+", help="Prediction JSON files")
    p.add_argument("--compare-only", action="store_true")
    args = p.parse_args()
    
    rs = []
    for f in args.files:
        try:
            r = analyze(f)
            if r:
                rs.append(r)
                if not args.compare_only:
                    show(r)
        except Exception as e:
            print(f"Error {f}: {e}")
    
    compare(rs)