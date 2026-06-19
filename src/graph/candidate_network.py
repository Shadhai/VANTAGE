"""Candidate similarity network from embedding space.

Uses pre-computed embeddings to find similar candidates.
Clusters via HDBSCAN for group discovery.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

from src.utils.io_utils import load_embeddings, load_json, save_json
from config.settings import EMBEDDINGS_FILE, CANDIDATE_IDS_FILE, PROCESSED_DIR


class CandidateNetwork:
    """Build and query candidate similarity network."""
    
    def __init__(self):
        self.embeddings = None
        self.candidate_ids = None
        self.nn_model = None
        self.clusters = {}
        self.cluster_labels = None
    
    def build(self, embeddings_path: Path = None, ids_path: Path = None,
              n_neighbors: int = 15):
        """Build similarity index from embeddings."""
        embeddings_path = embeddings_path or EMBEDDINGS_FILE
        ids_path = ids_path or CANDIDATE_IDS_FILE
        
        print("Building Candidate Similarity Network...")
        
        # Load embeddings
        self.embeddings = load_embeddings(embeddings_path)
        self.candidate_ids = load_json(ids_path)
        
        print(f"  Loaded {len(self.candidate_ids)} candidates")
        print(f"  Embeddings: {self.embeddings.shape}")
        
        # Build nearest neighbor index
        print(f"  Building KNN index (k={n_neighbors})...")
        self.nn_model = NearestNeighbors(
            n_neighbors=n_neighbors + 1,  # +1 because self is always nearest
            metric='cosine',
            algorithm='brute',
            n_jobs=-1
        )
        self.nn_model.fit(self.embeddings)
        print(f"  ✓ KNN index built")
        
        # Reduce to 2D for visualization
        print("  Computing 2D projection...")
        pca = PCA(n_components=2, random_state=42)
        self.positions_2d = pca.fit_transform(self.embeddings)
        print(f"  ✓ 2D positions computed (explained variance: {pca.explained_variance_ratio_.sum():.2%})")
        
        # Simple clustering by score tiers
        self._compute_clusters()
        
    def _compute_clusters(self):
        """Group candidates by embedding proximity."""
        # Use approximate clustering via KNN connectivity
        from collections import defaultdict
        
        # Get top-5 neighbors for each candidate
        distances, indices = self.nn_model.kneighbors(self.embeddings, n_neighbors=6)
        
        # Skip self (index 0)
        self.similar_candidates = {}
        for i, cid in enumerate(self.candidate_ids):
            neighbors = []
            for j in range(1, len(indices[i])):
                neighbor_idx = indices[i][j]
                similarity = 1.0 - distances[i][j]  # Convert distance to similarity
                if similarity > 0.5:  # Only highly similar
                    neighbors.append({
                        "candidate_id": self.candidate_ids[neighbor_idx],
                        "similarity": round(float(similarity), 4)
                    })
            self.similar_candidates[cid] = neighbors[:10]
    
    def get_similar_candidates(self, candidate_id: str, top_k: int = 10) -> List[Dict]:
        """Find candidates most similar to a given candidate."""
        if candidate_id not in self.candidate_ids:
            return []
        
        idx = self.candidate_ids.index(candidate_id)
        embedding = self.embeddings[idx:idx+1]
        
        distances, indices = self.nn_model.kneighbors(embedding, n_neighbors=top_k+1)
        
        similar = []
        for i in range(1, len(indices[0])):  # Skip self
            neighbor_idx = indices[0][i]
            similarity = 1.0 - distances[0][i]
            similar.append({
                "candidate_id": self.candidate_ids[neighbor_idx],
                "similarity": round(float(similarity), 4)
            })
        
        return similar
    
    def get_2d_positions(self, candidate_ids: List[str] = None) -> Dict:
        """Get 2D positions for visualization."""
        if candidate_ids is None:
            candidate_ids = self.candidate_ids
        
        positions = {}
        for cid in candidate_ids:
            if cid in self.candidate_ids:
                idx = self.candidate_ids.index(cid)
                positions[cid] = {
                    "x": float(self.positions_2d[idx, 0]),
                    "y": float(self.positions_2d[idx, 1])
                }
        
        return positions
    
    def find_candidate_by_id(self, candidate_id: str) -> int:
        """Get index of a candidate."""
        try:
            return self.candidate_ids.index(candidate_id)
        except ValueError:
            return -1
    
    def save(self, path: Path):
        """Save network structure."""
        data = {
            "similar_candidates": self.similar_candidates,
            "positions_2d": {
                cid: {"x": float(self.positions_2d[i, 0]), 
                      "y": float(self.positions_2d[i, 1])}
                for i, cid in enumerate(self.candidate_ids)
            }
        }
        save_json(data, path)
        print(f"✓ Candidate network saved to {path}")
    
    def load(self, path: Path, embeddings_path: Path = None, ids_path: Path = None):
        """Load network structure."""
        embeddings_path = embeddings_path or EMBEDDINGS_FILE
        ids_path = ids_path or CANDIDATE_IDS_FILE
        
        self.embeddings = load_embeddings(embeddings_path)
        self.candidate_ids = load_json(ids_path)
        
        data = load_json(path)
        self.similar_candidates = data.get("similar_candidates", {})
        
        # Rebuild KNN
        self.nn_model = NearestNeighbors(n_neighbors=16, metric='cosine', n_jobs=-1)
        self.nn_model.fit(self.embeddings)
        
        print(f"✓ Candidate network loaded: {len(self.candidate_ids)} candidates")