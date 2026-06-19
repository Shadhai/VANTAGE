"""I/O utilities for streaming large JSONL files and reading parquet."""

import json
import orjson
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Iterator, Dict, Any, List, Optional


def stream_candidates(filepath: Path, max_records: Optional[int] = None) -> Iterator[Dict]:
    """Stream candidates from JSONL file.
    
    Memory-efficient: only one record in memory at a time.
    Handles both plain .jsonl and gzipped .jsonl.gz files.
    """
    # Check if gzipped
    if str(filepath).endswith('.gz'):
        import gzip
        open_func = gzip.open
        mode = 'rt'
    else:
        open_func = open
        mode = 'r'
    
    with open_func(filepath, mode, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_records and i >= max_records:
                break
            line = line.strip()
            if line:
                try:
                    yield orjson.loads(line)
                except (orjson.JSONDecodeError, json.JSONDecodeError):
                    # Fallback to standard json
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue


def load_candidates_batch(filepath: Path, max_records: Optional[int] = None) -> List[Dict]:
    """Load candidates into memory (for smaller batches)."""
    candidates = []
    for candidate in stream_candidates(filepath, max_records):
        candidates.append(candidate)
    return candidates


def count_candidates(filepath: Path) -> int:
    """Count total candidates in JSONL file."""
    count = 0
    for _ in stream_candidates(filepath):
        count += 1
    return count


def save_embeddings(embeddings: np.ndarray, filepath: Path):
    """Save embeddings to numpy file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(filepath), embeddings)


def load_embeddings(filepath: Path) -> np.ndarray:
    """Load embeddings from numpy file."""
    if not filepath.exists():
        raise FileNotFoundError(f"Embeddings file not found: {filepath}")
    return np.load(str(filepath))


def save_features_to_parquet(features_list: List[Dict], filepath: Path):
    """Save feature dictionaries to parquet file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(features_list)
    df.to_parquet(filepath, index=False, engine='pyarrow')


def load_features_from_parquet(filepath: Path) -> pd.DataFrame:
    """Load features from parquet file."""
    if not filepath.exists():
        raise FileNotFoundError(f"Features file not found: {filepath}")
    return pd.read_parquet(filepath, engine='pyarrow')


def save_json(data: Any, filepath: Path):
    """Save data to JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath: Path) -> Any:
    """Load data from JSON file."""
    if not filepath.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_csv(rows: List[Dict], filepath: Path, columns: List[str]):
    """Save list of dicts to CSV file."""
    import csv
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def get_candidate_ids(filepath: Path) -> List[str]:
    """Extract all candidate IDs from JSONL file."""
    ids = []
    for candidate in stream_candidates(filepath):
        ids.append(candidate.get("candidate_id", ""))
    return ids