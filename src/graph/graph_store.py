"""Simple persistence helpers for graph objects."""
import pickle
from pathlib import Path
from typing import Any


class GraphStore:
    @staticmethod
    def save(obj: Any, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    @staticmethod
    def load(path: Path):
        with open(path, "rb") as f:
            return pickle.load(f)
