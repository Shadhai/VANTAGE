"""Skill transferability graph from 100K career histories.

Nodes: Skills (from candidates.jsonl)
Edges: Co-occurrence strength + transfer difficulty
Layout: UMAP projection of skill embeddings
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from config.settings import PROCESSED_DIR, RANKING_TERMS, UNWANTED_TERMS
from src.utils.io_utils import stream_candidates, save_json, load_json


class SkillGraph:
    """Build and query a skill transferability network."""
    
    def __init__(self):
        self.nodes = []          # Skill names
        self.node_ids = {}       # Skill → index
        self.co_occurrence = {}  # (skill_a, skill_b) → count
        self.skill_freq = {}     # skill → total count
        self.edges = []          # [(source, target, weight, transfer_difficulty)]
        self.embeddings = None   # Node positions via UMAP
        
    def build(self, candidates_path: Path, max_candidates: int = None):
        """Build graph from candidate skill lists and career evidence."""
        print("Building Skill Transferability Graph...")
        
        # Pass 1: Count skill co-occurrence
        skill_pairs = Counter()
        skill_total = Counter()
        verified_skills = defaultdict(set)
        
        count = 0
        for candidate in stream_candidates(candidates_path):
            skills = candidate.get("skills", [])
            career_text = " ".join([
                j.get("description", "") 
                for j in candidate.get("career_history", [])
            ]).lower()
            
            skill_names = []
            for s in skills:
                name = s.get("name", "").strip()
                if name:
                    skill_names.append(name)
                    skill_total[name] += 1
                    # Check if verified in career
                    if name.lower() in career_text:
                        verified_skills[candidate.get("candidate_id")].add(name)
            
            # Count pairs
            for i in range(len(skill_names)):
                for j in range(i+1, len(skill_names)):
                    pair = tuple(sorted([skill_names[i], skill_names[j]]))
                    skill_pairs[pair] += 1
            
            count += 1
            if max_candidates and count >= max_candidates:
                break
        
        print(f"  Processed {count} candidates")
        print(f"  Unique skills: {len(skill_total)}")
        print(f"  Skill pairs: {len(skill_pairs)}")
        
        # Build nodes (top 500 skills by frequency)
        top_skills = [s for s, _ in skill_total.most_common(500)]
        self.nodes = top_skills
        self.node_ids = {s: i for i, s in enumerate(top_skills)}
        self.skill_freq = {s: skill_total[s] for s in top_skills}
        
        # Build edges
        self.edges = []
        for (s1, s2), count in skill_pairs.items():
            if s1 in self.node_ids and s2 in self.node_ids:
                # Edge weight = Jaccard similarity
                union = skill_total[s1] + skill_total[s2] - count
                weight = count / max(union, 1)
                
                # Transfer difficulty (approximate)
                # Low = similar skills, High = very different skills
                difficulty = 1.0 - weight
                
                if weight > 0.01:  # Filter noise
                    self.edges.append({
                        "source": self.node_ids[s1],
                        "target": self.node_ids[s2],
                        "weight": round(weight, 4),
                        "difficulty": round(difficulty, 4),
                        "count": count
                    })
        
        print(f"  Edges: {len(self.edges)} (weight > 0.01)")
        
        # Compute node categories
        self._categorize_nodes()
        
    def _categorize_nodes(self):
        """Tag skills by category for coloring."""
        self.categories = {}
        
        ranking_set = set(t.lower() for t in RANKING_TERMS)
        unwanted_set = set(t.lower() for t in UNWANTED_TERMS)
        
        for skill in self.nodes:
            sl = skill.lower()
            if any(t in sl for t in ["python", "pytorch", "tensorflow", "sklearn", "xgboost"]):
                self.categories[skill] = "ml_framework"
            elif any(t in sl for t in ranking_set):
                self.categories[skill] = "ranking_ir"
            elif any(t in sl for t in ["embedding", "vector", "semantic"]):
                self.categories[skill] = "embeddings"
            elif any(t in sl for t in ["llm", "gpt", "bert", "transformer", "fine-tuning", "lora"]):
                self.categories[skill] = "llm"
            elif any(t in sl for t in ["faiss", "pinecone", "weaviate", "qdrant", "milvus"]):
                self.categories[skill] = "vector_db"
            elif any(t in sl for t in unwanted_set):
                self.categories[skill] = "unwanted"
            elif any(t in sl for t in ["docker", "kubernetes", "aws", "gcp", "azure"]):
                self.categories[skill] = "infrastructure"
            elif any(t in sl for t in ["sql", "spark", "airflow", "kafka", "pipeline"]):
                self.categories[skill] = "data_engineering"
            else:
                self.categories[skill] = "other"
    
    def get_transfer_path(self, skill_a: str, skill_b: str) -> Dict:
        """Find shortest transfer path between two skills."""
        # BFS on the skill graph
        if skill_a not in self.node_ids or skill_b not in self.node_ids:
            return {"path": [], "difficulty": 1.0, "exists": False}
        
        start = self.node_ids[skill_a]
        end = self.node_ids[skill_b]
        
        # Build adjacency
        adj = defaultdict(list)
        for edge in self.edges:
            adj[edge["source"]].append((edge["target"], edge["difficulty"]))
            adj[edge["target"]].append((edge["source"], edge["difficulty"]))
        
        # BFS
        from collections import deque
        queue = deque([(start, [start], 0.0)])
        visited = {start: 0.0}
        
        while queue:
            node, path, cost = queue.popleft()
            if node == end:
                path_names = [self.nodes[n] for n in path]
                return {
                    "path": path_names,
                    "difficulty": round(cost, 4),
                    "steps": len(path) - 1,
                    "exists": True
                }
            
            for neighbor, difficulty in adj[node]:
                new_cost = cost + difficulty
                if neighbor not in visited or new_cost < visited[neighbor]:
                    visited[neighbor] = new_cost
                    queue.append((neighbor, path + [neighbor], new_cost))
        
        return {"path": [], "difficulty": 1.0, "exists": False}
    
    def get_related_skills(self, skill: str, top_k: int = 10) -> List[Dict]:
        """Get most related skills to a given skill."""
        if skill not in self.node_ids:
            return []
        
        skill_id = self.node_ids[skill]
        related = []
        
        for edge in self.edges:
            if edge["source"] == skill_id:
                related.append({
                    "skill": self.nodes[edge["target"]],
                    "weight": edge["weight"],
                    "difficulty": edge["difficulty"]
                })
            elif edge["target"] == skill_id:
                related.append({
                    "skill": self.nodes[edge["source"]],
                    "weight": edge["weight"],
                    "difficulty": edge["difficulty"]
                })
        
        related.sort(key=lambda x: x["weight"], reverse=True)
        return related[:top_k]
    
    def save(self, path: Path):
        """Save graph to JSON."""
        data = {
            "nodes": self.nodes,
            "node_ids": self.node_ids,
            "edges": self.edges,
            "skill_freq": self.skill_freq,
            "categories": self.categories
        }
        save_json(data, path)
        print(f"✓ Skill graph saved to {path}")
    
    def load(self, path: Path):
        """Load graph from JSON."""
        data = load_json(path)
        self.nodes = data["nodes"]
        self.node_ids = data["node_ids"]
        self.edges = data["edges"]
        self.skill_freq = data["skill_freq"]
        self.categories = data.get("categories", {})
        print(f"✓ Skill graph loaded: {len(self.nodes)} nodes, {len(self.edges)} edges")