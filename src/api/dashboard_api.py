"""Dashboard data endpoints used by the Streamlit app."""
from fastapi import FastAPI, HTTPException
from pathlib import Path
from src.graph.graph_store import GraphStore

app = FastAPI(title="Dashboard API")


@app.get("/dashboard/summary")
def summary():
    p = Path("./models/summary.pkl")
    if not p.exists():
        raise HTTPException(status_code=404, detail="Summary not found")
    return GraphStore.load(p)
