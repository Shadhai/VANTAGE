"""Career flow graph showing talent movement between companies.

Nodes: Companies
Edges: Number of candidates who moved from Company A → Company B
"""

import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List

from src.utils.io_utils import stream_candidates, save_json, load_json
from config.settings import PRODUCT_COMPANIES, CONSULTING_COMPANIES


class CareerFlowGraph:
    """Build and query a company-to-company talent flow network."""
    
    def __init__(self):
        self.nodes = []
        self.node_ids = {}
        self.flows = []          # [(source, target, count)]
        self.company_stats = {}  # company → {total, product, consulting}
        self.categories = {}
    
    def build(self, candidates_path: Path, max_candidates: int = None):
        """Build flow graph from career transitions."""
        print("Building Career Flow Graph...")
        
        transitions = Counter()
        company_total = Counter()
        
        count = 0
        for candidate in stream_candidates(candidates_path):
            career = candidate.get("career_history", [])
            companies = [j.get("company", "") for j in career if j.get("company")]
            
            for c in companies:
                company_total[c] += 1
            
            # Count transitions (company A → company B)
            for i in range(len(companies) - 1):
                source = companies[i]
                target = companies[i + 1]
                if source and target:
                    transitions[(source, target)] += 1
            
            count += 1
            if max_candidates and count >= max_candidates:
                break
        
        print(f"  Processed {count} candidates")
        print(f"  Unique companies: {len(company_total)}")
        print(f"  Transitions: {len(transitions)}")
        
        # Build nodes (companies with ≥5 candidates)
        self.nodes = [c for c, cnt in company_total.items() if cnt >= 5]
        self.node_ids = {c: i for i, c in enumerate(self.nodes)}
        
        # Categorize
        for company in self.nodes:
            if company in PRODUCT_COMPANIES:
                self.categories[company] = "product"
            elif company in CONSULTING_COMPANIES:
                self.categories[company] = "consulting"
            else:
                self.categories[company] = "other"
        
        # Build flows
        self.flows = []
        for (source, target), count in transitions.items():
            if source in self.node_ids and target in self.node_ids:
                if count >= 2:  # Filter noise
                    self.flows.append({
                        "source": self.node_ids[source],
                        "target": self.node_ids[target],
                        "count": count
                    })
        
        # Company stats
        for company in self.nodes:
            self.company_stats[company] = {
                "total_candidates": company_total[company],
                "category": self.categories.get(company, "other")
            }
        
        print(f"  Nodes: {len(self.nodes)}")
        print(f"  Flows: {len(self.flows)} (count ≥ 2)")
    
    def get_talent_inflow(self, company: str) -> List[Dict]:
        """Which companies do candidates come from?"""
        if company not in self.node_ids:
            return []
        
        company_id = self.node_ids[company]
        inflows = []
        
        for flow in self.flows:
            if flow["target"] == company_id:
                inflows.append({
                    "from": self.nodes[flow["source"]],
                    "count": flow["count"]
                })
        
        inflows.sort(key=lambda x: x["count"], reverse=True)
        return inflows
    
    def get_talent_outflow(self, company: str) -> List[Dict]:
        """Where do candidates go after leaving?"""
        if company not in self.node_ids:
            return []
        
        company_id = self.node_ids[company]
        outflows = []
        
        for flow in self.flows:
            if flow["source"] == company_id:
                outflows.append({
                    "to": self.nodes[flow["target"]],
                    "count": flow["count"]
                })
        
        outflows.sort(key=lambda x: x["count"], reverse=True)
        return outflows
    
    def save(self, path: Path):
        """Save graph to JSON."""
        data = {
            "nodes": self.nodes,
            "node_ids": self.node_ids,
            "flows": self.flows,
            "company_stats": self.company_stats,
            "categories": self.categories
        }
        save_json(data, path)
        print(f"✓ Career flow graph saved to {path}")
    
    def load(self, path: Path):
        """Load graph from JSON."""
        data = load_json(path)
        self.nodes = data["nodes"]
        self.node_ids = data["node_ids"]
        self.flows = data["flows"]
        self.company_stats = data["company_stats"]
        self.categories = data.get("categories", {})
        print(f"✓ Career flow graph loaded: {len(self.nodes)} companies, {len(self.flows)} flows")