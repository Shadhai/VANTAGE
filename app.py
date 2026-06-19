"""VANTAGE — HuggingFace Sandbox."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="VANTAGE", page_icon="●", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&display=swap');
* { font-family: 'IBM Plex Mono', monospace; }
.stApp { background: #0a0a0a; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 style="font-weight:400;color:#fff;">VANTAGE</h1>', unsafe_allow_html=True)
st.caption("Talent Intelligence · Redrob Hackathon")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Candidates", "100,000")
c2.metric("Runtime", "156s")
c3.metric("Memory", "500 MB")
c4.metric("Pipeline", "CPU-only")

st.divider()
st.subheader("Architecture")
st.code("""
JD -> Parse -> Semantic Embeddings -> Multi-Dimensional Scoring -> XGBoost -> Top 100
       |              |                        |
  86 terms    384-dim vectors          Career evidence
  5 gates     Cosine similarity        Skill verification
  6 prefs     100K pre-computed        Behavioral signals
                                       Honeypot detection
""")

st.divider()

try:
    df = pd.read_csv("output/submission.csv")
    st.subheader("Sample Output")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['rank'], y=df['score'], mode='lines', line=dict(color='#00ff88', width=2), fill='tozeroy', fillcolor='rgba(0,255,136,0.08)'))
    fig.update_layout(template='plotly_dark', height=300, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)
except:
    st.info("Run pipeline to generate output. Full reproduction at Stage 3 evaluation.")

st.divider()
st.caption("VANTAGE · CPU-ONLY · 500MB RAM · ZERO API CALLS · FULLY OFFLINE")
