#!/usr/bin/env python3
"""Split comparison data into batches for annotators."""
import json
import os
import sys
import hashlib
from pathlib import Path


def _text_fingerprint(text):
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()


def split_batches(input_path, pairs_per_batch=30):
    with open(input_path) as f:
        data = json.load(f)
    
    pairs = data.get('pairs', [])
    config = data.get('config', {})
    output_dir = str(Path(input_path).parent)
    
    num_batches = (len(pairs) + pairs_per_batch - 1) // pairs_per_batch
    print(f"Splitting {len(pairs)} pairs into {num_batches} batches of ~{pairs_per_batch} pairs each")
    
    # Load readability cache
    cache_path = str(Path(input_path).with_suffix("")) + "_readability_cache.json"
    readability_cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            readability_cache = json.load(f)
        print(f"Loaded readability cache with {len(readability_cache)} entries")
    
    for batch_idx in range(num_batches):
        start = batch_idx * pairs_per_batch
        end = min(start + pairs_per_batch, len(pairs))
        batch_pairs = pairs[start:end]
        
        batch_letter = chr(ord('A') + batch_idx)
        batch_path = os.path.join(output_dir, f"comparison_batch_{batch_letter}.json")
        
        batch_data = {
            'config': {**config, 'batch': batch_letter, 'batch_start': start, 'batch_end': end},
            'pairs': batch_pairs,
        }
        
        with open(batch_path, 'w') as f:
            json.dump(batch_data, f, indent=2)
        
        # Split cache
        if readability_cache:
            batch_cache = {}
            for pair in batch_pairs:
                for side in ['side_a', 'side_b']:
                    fp = _text_fingerprint(pair[side])
                    if fp in readability_cache:
                        batch_cache[fp] = readability_cache[fp]
            cache_out = os.path.join(output_dir, f"comparison_batch_{batch_letter}_readability_cache.json")
            with open(cache_out, 'w') as f:
                json.dump(batch_cache, f, indent=2)
        
        print(f"  Batch {batch_letter}: pairs {start+1}-{end} ({len(batch_pairs)} pairs)")
    
    print(f"\nTo serve each batch:")
    for i in range(num_batches):
        letter = chr(ord('A') + i)
        port = 8900 + i
        print(f"  python serve_comparison.py --comparison-data comparison_batch_{letter}.json --port {port}")


if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else "comparison_human_annotation.json"
    pairs_per = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    split_batches(input_file, pairs_per)
