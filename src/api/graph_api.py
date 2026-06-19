"""FastAPI endpoints for graph queries (minimal stubs)."""
from fastapi import FastAPI, HTTPException
from typing import List
from pathlib import Path
from src.graph.graph_store import GraphStore

app = FastAPI(title="Graph API")


@app.get("/graph/skill/{skill_name}")
def skill_neighbors(skill_name: str):
    # Placeholder: load a stored skill graph if available
    p = Path("./models/skill_graph.pkl")
    if not p.exists():
        raise HTTPException(status_code=404, detail="Skill graph not found")
    g = GraphStore.load(p)
    return {"skill": skill_name, "neighbors": g.get(skill_name, {})}


@app.get("/graph/career_flow")
def career_flow():
    p = Path("./models/career_flow.pkl")
    if not p.exists():
        raise HTTPException(status_code=404, detail="Career flow graph not found")
    g = GraphStore.load(p)
    return {"nodes": list(g.keys())}
