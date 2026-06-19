#!/usr/bin/env python3
"""Pre-compute career text embeddings for all 100K candidates.
   
   Uses GPU if available for fast encoding.
   Runtime: ~30 sec with GPU, ~10 min with CPU.
   Output: ~77MB float16 numpy file.
"""

import sys
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import CANDIDATES_FILE, PROCESSED_DIR, EMBEDDINGS_FILE, CANDIDATE_IDS_FILE
from src.utils.io_utils import stream_candidates, save_embeddings, save_json
from src.utils.text_utils import concatenate_career_text


def main():
    print("=" * 60)
    print("Pre-computing Candidate Embeddings")
    print("=" * 60)
    
    if not CANDIDATES_FILE.exists():
        print(f"ERROR: Candidates file not found: {CANDIDATES_FILE}")
        sys.exit(1)
    
    # Check GPU availability
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n  Device: {device.upper()}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load model
    print("\n[1/3] Loading embedding model...")
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    print(f"  ✓ all-MiniLM-L6-v2 loaded on {device.upper()}")
    
    # Collect texts
    print("\n[2/3] Reading candidate career texts...")
    career_texts = []
    candidate_ids = []
    
    for candidate in tqdm(stream_candidates(CANDIDATES_FILE), desc="  Reading"):
        cid = candidate.get("candidate_id", "")
        career_text = concatenate_career_text(candidate.get("career_history", []))
        summary = candidate.get("profile", {}).get("summary", "")
        combined = f"{summary} {career_text}"
        
        candidate_ids.append(cid)
        career_texts.append(combined if combined.strip() else "no career data")
    
    print(f"  ✓ Collected {len(career_texts)} texts")
    
    # Generate embeddings with GPU
    print(f"\n[3/3] Generating embeddings on {device.upper()}...")
    start_time = time.time()
    
    # GPU can handle much larger batches
    batch_size = 1024 if device == "cuda" else 128
    
    all_embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(career_texts), batch_size), desc="  Encoding"):
            batch = career_texts[i:i + batch_size]
            embeddings = model.encode(
                batch,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            all_embeddings.append(embeddings)
    
    embeddings_array = np.concatenate(all_embeddings, axis=0).astype(np.float16)
    elapsed = time.time() - start_time
    
    print(f"  ✓ Generated {embeddings_array.shape[0]} embeddings in {elapsed:.1f}s")
    print(f"  Shape: {embeddings_array.shape}")
    print(f"  Memory: {embeddings_array.nbytes / 1024 / 1024:.1f} MB")
    print(f"  Speed: {len(career_texts) / elapsed:.0f} texts/second")
    
    # Save
    print("\nSaving...")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    save_embeddings(embeddings_array, EMBEDDINGS_FILE)
    save_json(candidate_ids, CANDIDATE_IDS_FILE)
    
    print(f"✓ Embeddings saved to {EMBEDDINGS_FILE}")
    print(f"✓ Candidate IDs saved to {CANDIDATE_IDS_FILE}")


if __name__ == "__main__":
    main()