#!/usr/bin/env python3
"""Build all three knowledge graph structures.

Generates:
1. Skill Transferability Graph (skill_graph.json)
2. Career Flow Graph (career_flow_graph.json)  
3. Candidate Similarity Network (candidate_network.json)

Runtime: ~5 minutes. Run once.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import CANDIDATES_FILE, PROCESSED_DIR
from src.graph.skill_graph import SkillGraph
from src.graph.career_flow_graph import CareerFlowGraph
from src.graph.candidate_network import CandidateNetwork


def main():
    print("=" * 60)
    print("Building VANTAGE Knowledge Graph")
    print("=" * 60)
    
    start = time.time()
    
    # 1. Skill Graph
    print("\n[1/3] Skill Transferability Graph")
    skill_graph = SkillGraph()
    skill_graph.build(CANDIDATES_FILE)
    skill_graph.save(PROCESSED_DIR / "skill_graph.json")
    
    # 2. Career Flow Graph
    print("\n[2/3] Career Flow Graph")
    career_graph = CareerFlowGraph()
    career_graph.build(CANDIDATES_FILE)
    career_graph.save(PROCESSED_DIR / "career_flow_graph.json")
    
    # 3. Candidate Network
    print("\n[3/3] Candidate Similarity Network")
    candidate_network = CandidateNetwork()
    candidate_network.build()
    candidate_network.save(PROCESSED_DIR / "candidate_network.json")
    
    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"✓ Knowledge graph built in {elapsed:.1f} seconds")
    print(f"  skill_graph.json          — Skill transferability network")
    print(f"  career_flow_graph.json    — Company talent flows")
    print(f"  candidate_network.json    — Candidate similarity clusters")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()