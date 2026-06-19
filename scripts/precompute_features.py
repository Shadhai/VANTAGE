#!/usr/bin/env python3
"""Pre-compute all candidate features and text frequencies."""

import sys
import time
from pathlib import Path
from collections import Counter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import (
    CANDIDATES_FILE, PROCESSED_DIR, 
    CANDIDATE_FEATURES_FILE, TEXT_FREQ_FILE
)
from src.features.extractor import FeatureExtractor
from src.utils.io_utils import (
    stream_candidates, save_features_to_parquet, save_json
)
from src.utils.text_utils import concatenate_career_text, hash_text


def main():
    print("=" * 60)
    print("Pre-computing Candidate Features")
    print("=" * 60)
    
    # Check if candidates file exists
    if not CANDIDATES_FILE.exists():
        print(f"ERROR: Candidates file not found: {CANDIDATES_FILE}")
        sys.exit(1)
    
    # Pass 1: Compute text frequencies
    print("\n[1/3] Computing text frequencies...")
    text_hashes = Counter()
    total_candidates = 0
    
    for candidate in tqdm(stream_candidates(CANDIDATES_FILE), 
                          desc="  Hashing"):
        career_text = concatenate_career_text(candidate.get("career_history", []))
        if career_text.strip():
            text_hash = hash_text(career_text)
            text_hashes[text_hash] += 1
        total_candidates += 1
    
    print(f"  Processed {total_candidates} candidates")
    
    # Convert to regular dict and save
    text_freq_dict = dict(text_hashes)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    save_json(text_freq_dict, TEXT_FREQ_FILE)
    print(f"  Found {len(text_freq_dict)} unique career text patterns")
    if text_hashes:
        print(f"  Most common: {text_hashes.most_common(3)}")
    
    # Pass 2: Extract features
    print("\n[2/3] Extracting features...")
    extractor = FeatureExtractor(text_frequencies=text_freq_dict)
    all_features = []
    
    start_time = time.time()
    
    for candidate in tqdm(stream_candidates(CANDIDATES_FILE), 
                          desc="  Extracting"):
        features = extractor.extract_all(candidate)
        all_features.append(features)
    
    elapsed = time.time() - start_time
    print(f"  Extracted features for {len(all_features)} candidates "
          f"in {elapsed:.1f} seconds")
    
    # Save to parquet
    print("\n[3/3] Saving to parquet...")
    save_features_to_parquet(all_features, CANDIDATE_FEATURES_FILE)
    
    # Print file size
    size_mb = CANDIDATE_FEATURES_FILE.stat().st_size / 1024 / 1024
    print(f"✓ Features saved to {CANDIDATE_FEATURES_FILE}")
    print(f"  File size: {size_mb:.1f} MB")
    if all_features:
        print(f"  Features per candidate: {len(all_features[0])}")


if __name__ == "__main__":
    main()