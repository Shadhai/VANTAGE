# # """VANTAGE — Talent Intelligence Dashboard.

# # Interactive knowledge graph for candidate exploration.
# # Run: streamlit run src/dashboard/app.py
# # """

# # import sys
# # from pathlib import Path
# # sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# # import streamlit as st
# # import pandas as pd
# # import plotly.graph_objects as go
# # import plotly.express as px
# # from config.settings import PROCESSED_DIR, SUBMISSION_FILE

# # # Page config
# # st.set_page_config(
# #     page_title="VANTAGE — Talent Intelligence",
# #     page_icon="🎯",
# #     layout="wide",
# #     initial_sidebar_state="expanded"
# # )

# # # Load data
# # @st.cache_data
# # def load_submission():
# #     return pd.read_csv(SUBMISSION_FILE)

# # @st.cache_data
# # def load_graph_data():
# #     from src.utils.io_utils import load_json
# #     return {
# #         "skills": load_json(PROCESSED_DIR / "skill_graph.json"),
# #         "career": load_json(PROCESSED_DIR / "career_flow_graph.json"),
# #         "candidates": load_json(PROCESSED_DIR / "candidate_network.json"),
# #     }

# # # Header
# # st.title("🎯 VANTAGE — Talent Intelligence Graph")
# # st.caption("AI-powered candidate discovery beyond keyword matching")

# # # Sidebar
# # with st.sidebar:
# #     st.header("Controls")
# #     min_score = st.slider("Minimum Match Score", 0.0, 1.0, 0.75, 0.01)
# #     location = st.multiselect(
# #         "Location Filter",
# #         ["Pune", "Noida", "Hyderabad", "Bangalore", "Mumbai", "Delhi", "Gurgaon", "Chennai"],
# #         default=["Pune", "Noida"]
# #     )
    
# #     st.divider()
# #     st.header("Quick Stats")
# #     df = load_submission()
# #     st.metric("Candidates Ranked", "100,000")
# #     st.metric("Pipeline Runtime", "83 seconds")
# #     st.metric("Top Score", f"{df['score'].max():.1%}")
    
# #     st.divider()
# #     st.header("Skill Explorer")
# #     search_skill = st.text_input("Search skill transfer path", "PyTorch")
# #     target_skill = st.text_input("Target skill", "FAISS")

# # # Main content
# # tab1, tab2, tab3, tab4 = st.tabs([
# #     "📊 Top Candidates", "🔗 Skill Graph", "🏢 Career Flows", "🌐 Candidate Map"
# # ])

# # # ── Tab 1: Top Candidates ────────────────────────────────
# # with tab1:
# #     st.subheader("Top Ranked Candidates")
    
# #     df = load_submission()
    
# #     # Filter by score
# #     df_filtered = df[df['score'] >= min_score]
    
# #     col1, col2 = st.columns([2, 1])
    
# #     with col1:
# #         # Score distribution
# #         fig = px.bar(
# #             df.head(20),
# #             x='score',
# #             y='candidate_id',
# #             orientation='h',
# #             title='Top 20 Candidates by Score',
# #             color='score',
# #             color_continuous_scale='viridis'
# #         )
# #         fig.update_layout(height=500)
# #         st.plotly_chart(fig, use_container_width=True)
    
# #     with col2:
# #         st.subheader("Candidate Details")
# #         selected_candidate = st.selectbox(
# #             "Select Candidate",
# #             df_filtered['candidate_id'].tolist()
# #         )
        
# #         if selected_candidate:
# #             row = df[df['candidate_id'] == selected_candidate].iloc[0]
# #             st.metric("Score", f"{row['score']:.2%}")
# #             st.metric("Rank", f"#{row['rank']}")
# #             st.write("**Reasoning:**")
# #             st.info(row['reasoning'])
    
# #     # Full table
# #     with st.expander("View Full Top 100"):
# #         st.dataframe(
# #             df[['rank', 'candidate_id', 'score', 'reasoning']],
# #             use_container_width=True,
# #             hide_index=True
# #         )

# # # ── Tab 2: Skill Graph ────────────────────────────────────
# # with tab2:
# #     st.subheader("Skill Transferability Network")
    
# #     try:
# #         graph_data = load_graph_data()
# #         skills = graph_data.get("skills", {})
        
# #         if skills:
# #             col1, col2 = st.columns([2, 1])
            
# #             with col1:
# #                 # Skill network visualization
# #                 nodes = skills.get("nodes", [])
# #                 edges = skills.get("edges", [])
# #                 categories = skills.get("categories", {})
                
# #                 # Color map
# #                 color_map = {
# #                     "ranking_ir": "#00ff88",
# #                     "embeddings": "#00aaff",
# #                     "llm": "#ffaa00",
# #                     "vector_db": "#ff00aa",
# #                     "ml_framework": "#00ffaa",
# #                     "data_engineering": "#aaff00",
# #                     "infrastructure": "#8888ff",
# #                     "unwanted": "#ff4444",
# #                     "other": "#888888"
# #                 }
                
# #                 # Create edge traces
# #                 edge_x = []
# #                 edge_y = []
# #                 for edge in edges[:1000]:  # Limit for performance
# #                     s, t = edge["source"], edge["target"]
# #                     # Simple circle layout
# #                     import math
# #                     n = len(nodes)
# #                     for i, (s_idx, t_idx) in enumerate([(s, t)]):
# #                         angle_s = 2 * math.pi * s_idx / n
# #                         angle_t = 2 * math.pi * t_idx / n
# #                         edge_x.extend([math.cos(angle_s), math.cos(angle_t), None])
# #                         edge_y.extend([math.sin(angle_s), math.sin(angle_t), None])
                
# #                 fig = go.Figure()
                
# #                 # Add edges
# #                 fig.add_trace(go.Scatter(
# #                     x=edge_x, y=edge_y,
# #                     mode='lines',
# #                     line=dict(color='rgba(100,100,100,0.2)', width=0.5),
# #                     hoverinfo='none'
# #                 ))
                
# #                 # Add nodes
# #                 node_x = []
# #                 node_y = []
# #                 node_colors = []
# #                 node_sizes = []
# #                 node_text = []
                
# #                 freq = skills.get("skill_freq", {})
# #                 for i, node in enumerate(nodes):
# #                     angle = 2 * math.pi * i / len(nodes)
# #                     node_x.append(math.cos(angle))
# #                     node_y.append(math.sin(angle))
# #                     cat = categories.get(node, "other")
# #                     node_colors.append(color_map.get(cat, "#888888"))
# #                     node_sizes.append(min(20, freq.get(node, 1) / 100))
# #                     node_text.append(f"{node}<br>Count: {freq.get(node, 0)}")
                
# #                 fig.add_trace(go.Scatter(
# #                     x=node_x, y=node_y,
# #                     mode='markers+text',
# #                     marker=dict(
# #                         size=node_sizes,
# #                         color=node_colors,
# #                         line=dict(width=1, color='white')
# #                     ),
# #                     text=[n if freq.get(n, 0) > 500 else '' for n in nodes],
# #                     textposition='top center',
# #                     hovertext=node_text,
# #                     hoverinfo='text'
# #                 ))
                
# #                 fig.update_layout(
# #                     title='Skill Co-occurrence Network',
# #                     showlegend=False,
# #                     height=600,
# #                     xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
# #                     yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
# #                 )
                
# #                 st.plotly_chart(fig, use_container_width=True)
            
# #             with col2:
# #                 st.subheader("Transfer Path Finder")
                
# #                 from src.graph.skill_graph import SkillGraph
# #                 sg = SkillGraph()
# #                 sg.load(PROCESSED_DIR / "skill_graph.json")
                
# #                 if st.button("Find Transfer Path"):
# #                     path = sg.get_transfer_path(search_skill, target_skill)
# #                     if path["exists"]:
# #                         st.success(f"Path found! {path['steps']} steps, difficulty: {path['difficulty']:.2f}")
# #                         st.write(" → ".join(path["path"]))
# #                     else:
# #                         st.warning("No path found between these skills")
                
# #                 st.divider()
# #                 st.subheader("Legend")
# #                 for cat, color in color_map.items():
# #                     st.markdown(f"● {cat.replace('_', ' ').title()}")
    
# #     except Exception as e:
# #         st.warning(f"Skill graph data not available. Run: python scripts/build_knowledge_graph.py")
# #         st.warning(f"Error: {e}")

# # # ── Tab 3: Career Flows ────────────────────────────────────
# # with tab3:
# #     st.subheader("Talent Flow Between Companies")
    
# #     try:
# #         graph_data = load_graph_data()
# #         career = graph_data.get("career", {})
        
# #         if career:
# #             companies = career.get("nodes", [])
# #             flows = career.get("flows", [])
# #             categories = career.get("categories", {})
            
# #             # Top flows
# #             top_flows = sorted(flows, key=lambda x: x["count"], reverse=True)[:50]
            
# #             col1, col2 = st.columns([2, 1])
            
# #             with col1:
# #                 # Sankey diagram
# #                 source_ids = []
# #                 target_ids = []
# #                 values = []
# #                 labels = companies
                
# #                 for flow in top_flows:
# #                     source_ids.append(flow["source"])
# #                     target_ids.append(flow["target"])
# #                     values.append(flow["count"])
                
# #                 fig = go.Figure(data=[go.Sankey(
# #                     node=dict(
# #                         pad=15,
# #                         thickness=20,
# #                         line=dict(color="black", width=0.5),
# #                         label=labels,
# #                         color=["#00ff88" if categories.get(c) == "product" 
# #                                else "#ff4444" if categories.get(c) == "consulting"
# #                                else "#888888" for c in labels]
# #                     ),
# #                     link=dict(
# #                         source=source_ids,
# #                         target=target_ids,
# #                         value=values
# #                     )
# #                 )])
                
# #                 fig.update_layout(title="Company-to-Company Talent Migration", height=500)
# #                 st.plotly_chart(fig, use_container_width=True)
            
# #             with col2:
# #                 st.subheader("Talent Magnets")
# #                 company_stats = career.get("company_stats", {})
                
# #                 # Companies with most candidates
# #                 sorted_companies = sorted(
# #                     company_stats.items(),
# #                     key=lambda x: x[1].get("total_candidates", 0),
# #                     reverse=True
# #                 )[:15]
                
# #                 company_df = pd.DataFrame([
# #                     {"Company": c, "Candidates": s["total_candidates"]}
# #                     for c, s in sorted_companies
# #                 ])
                
# #                 fig = px.bar(
# #                     company_df,
# #                     x="Candidates",
# #                     y="Company",
# #                     orientation='h',
# #                     title="Top Companies by Candidate Count"
# #                 )
# #                 st.plotly_chart(fig, use_container_width=True)
    
# #     except Exception as e:
# #         st.warning(f"Career flow data not available. Run: python scripts/build_knowledge_graph.py")

# # # ── Tab 4: Candidate Map ────────────────────────────────────
# # with tab4:
# #     st.subheader("Candidate Similarity Landscape")
    
# #     try:
# #         graph_data = load_graph_data()
# #         candidates = graph_data.get("candidates", {})
        
# #         if candidates:
# #             positions = candidates.get("positions_2d", {})
            
# #             # Convert to dataframe
# #             pos_df = pd.DataFrame([
# #                 {"candidate_id": cid, "x": p["x"], "y": p["y"]}
# #                 for cid, p in positions.items()
# #             ])
            
# #             # Merge with submission scores
# #             df = load_submission()
# #             pos_df = pos_df.merge(df[['candidate_id', 'score', 'rank']], 
# #                                   on='candidate_id', how='left')
# #             pos_df['in_top_100'] = pos_df['rank'].notna()
# #             pos_df['score'] = pos_df['score'].fillna(0.5)
            
# #             fig = px.scatter(
# #                 pos_df,
# #                 x='x', y='y',
# #                 color='score',
# #                 size='score',
# #                 hover_data=['candidate_id', 'rank'],
# #                 color_continuous_scale='viridis',
# #                 title='Candidate Embedding Space (PCA Projection)',
# #                 opacity=0.6
# #             )
            
# #             # Highlight top 100
# #             top100 = pos_df[pos_df['in_top_100']]
# #             fig.add_trace(go.Scatter(
# #                 x=top100['x'], y=top100['y'],
# #                 mode='markers',
# #                 marker=dict(
# #                     size=12,
# #                     color='#ff4444',
# #                     symbol='star',
# #                     line=dict(width=1, color='white')
# #                 ),
# #                 name='Top 100',
# #                 hovertext=top100['candidate_id']
# #             ))
            
# #             fig.update_layout(height=600)
# #             st.plotly_chart(fig, use_container_width=True)
            
# #             st.caption("⭐ Red stars = Top 100 ranked candidates. Color = match score.")
    
# #     except Exception as e:
# #         st.warning(f"Candidate network data not available. Run: python scripts/build_knowledge_graph.py")

# # # Footer
# # st.divider()
# # st.caption("VANTAGE — Talent Intelligence System | Built for Redrob Hackathon")
# """VANTAGE — Talent Intelligence Dashboard.

# Retro terminal aesthetic. Dark/light toggle. All views working.
# Run: streamlit run src/dashboard/app.py
# """

# import sys
# from pathlib import Path
# sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# import plotly.express as px
# import networkx as nx
# from config.settings import SUBMISSION_FILE, PROCESSED_DIR

# st.set_page_config(page_title="VANTAGE", page_icon="●", layout="wide")

# # ── Theme ─────────────────────────────────────────────────

# if "theme" not in st.session_state:
#     st.session_state.theme = "dark"

# with st.sidebar:
#     st.markdown("### ⚙️ Display")
#     theme_choice = st.toggle("Dark Mode", value=st.session_state.theme == "dark")
#     st.session_state.theme = "dark" if theme_choice else "light"
#     st.divider()
#     st.markdown("### 📊 Views")
#     view = st.radio("Select View", ["Candidates", "Skill Graph", "Career Flows", "Candidate Map"], label_visibility="collapsed")

# dark = st.session_state.theme == "dark"

# if dark:
#     bg, card_bg, border = "#0a0a0a", "#0f0f0f", "#1a1a1a"
#     text, dim, accent, hero_color = "#c0c0c0", "#555", "#00ff88", "#ffffff"
#     plot_bg, grid_color = "#0a0a0a", "rgba(255,255,255,0.03)"
#     product_color, consulting_color = "#00ff88", "#ff4444"
# else:
#     bg, card_bg, border = "#f5f0e8", "#fffef9", "#e0d8c8"
#     text, dim, accent, hero_color = "#3a3a3a", "#999", "#1a7a4c", "#1a1a1a"
#     plot_bg, grid_color = "#f5f0e8", "rgba(0,0,0,0.05)"
#     product_color, consulting_color = "#1a7a4c", "#cc3333"

# # ── CSS ───────────────────────────────────────────────────

# st.markdown(f"""
# <style>
#     @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&display=swap');
#     * {{ font-family: 'IBM Plex Mono', monospace; }}
#     .stApp {{ background: {bg}; }}
#     .hero {{ font-size: 2.8rem; font-weight: 400; color: {hero_color}; }}
#     .hero-sub {{ font-size: 0.75rem; color: {dim}; text-transform: uppercase; letter-spacing: 3px; }}
#     .stat-box {{ background: {card_bg}; border: 1px solid {border}; padding: 20px; text-align: center; }}
#     .stat-val {{ font-size: 2rem; font-weight: 400; color: {accent}; }}
#     .stat-lab {{ font-size: 0.6rem; color: {dim}; text-transform: uppercase; letter-spacing: 2px; }}
#     .card {{ background: {card_bg}; border: 1px solid {border}; padding: 16px 20px; margin-bottom: 8px; }}
#     .card:hover {{ border-color: {accent}44; }}
#     .card-top {{ border-left: 3px solid {accent}; }}
#     .rank-num {{ font-size: 1.2rem; color: {accent}; }}
#     .rank-id {{ font-size: 0.75rem; color: {dim}; }}
#     .rank-score {{ font-size: 1rem; color: {accent}; }}
#     .rank-reason {{ font-size: 0.78rem; color: {text}; line-height: 1.6; }}
#     .section-title {{ font-size: 0.65rem; color: {dim}; text-transform: uppercase; letter-spacing: 3px; margin-bottom: 16px; }}
#     hr {{ border-color: {border}; margin: 24px 0; }}
#     .footer {{ font-size: 0.6rem; color: {dim}; text-align: center; letter-spacing: 1px; }}
#     .stTextInput input {{ background: {card_bg} !important; border: 1px solid {border} !important; color: {text} !important; font-family: 'IBM Plex Mono', monospace !important; }}
#     .flow-row {{ display: flex; align-items: center; padding: 6px 0; border-bottom: 1px solid {border}; font-size: 0.75rem; }}
#     .flow-src {{ width: 120px; text-align: right; padding-right: 10px; }}
#     .flow-arrow {{ width: 20px; text-align: center; color: {dim}; }}
#     .flow-tgt {{ width: 120px; padding-left: 10px; }}
#     .flow-count {{ width: 50px; text-align: right; font-weight: 500; }}
#     .flow-bar {{ flex: 1; height: 4px; background: {border}; border-radius: 2px; margin-left: 10px; }}
#     .flow-bar-inner {{ height: 100%; background: {accent}; border-radius: 2px; opacity: 0.5; }}
# </style>
# """, unsafe_allow_html=True)

# # ── Data ──────────────────────────────────────────────────

# @st.cache_data
# def load_submission():
#     return pd.read_csv(SUBMISSION_FILE)

# @st.cache_data
# def load_skill_graph():
#     from src.utils.io_utils import load_json
#     try: return load_json(PROCESSED_DIR / "skill_graph.json")
#     except: return None

# @st.cache_data
# def load_career_graph():
#     from src.utils.io_utils import load_json
#     try: return load_json(PROCESSED_DIR / "career_flow_graph.json")
#     except: return None

# @st.cache_data
# def load_candidate_network():
#     from src.utils.io_utils import load_json
#     try: return load_json(PROCESSED_DIR / "candidate_network.json")
#     except: return None

# df = load_submission()
# skill_data = load_skill_graph()
# career_data = load_career_graph()
# candidate_data = load_candidate_network()

# # ── Hero ──────────────────────────────────────────────────

# st.markdown(f'<p class="hero">VANTAGE</p>', unsafe_allow_html=True)
# st.markdown(f'<p class="hero-sub">Talent Intelligence · {view}</p>', unsafe_allow_html=True)
# st.markdown("<br>", unsafe_allow_html=True)

# # ── Stats ─────────────────────────────────────────────────

# c1, c2, c3, c4, c5 = st.columns(5)
# stats = [
#     (f"{df['score'].max():.3f}", "Top Score"),
#     ("100k", "Candidates"),
#     ("156s", "Runtime"),
#     ("500mb", "Memory"),
#     (str(df["reasoning"].nunique()), "Unique Insights"),
# ]
# for col, (val, lab) in zip([c1, c2, c3, c4, c5], stats):
#     col.markdown(f'<div class="stat-box"><div class="stat-val">{val}</div><div class="stat-lab">{lab}</div></div>', unsafe_allow_html=True)

# st.markdown("<hr>", unsafe_allow_html=True)

# # ═══════════════════════════════════════════════════════════
# # VIEW: CANDIDATES
# # ═══════════════════════════════════════════════════════════

# if view == "Candidates":
    
#     st.markdown('<p class="section-title">Score Distribution</p>', unsafe_allow_html=True)
    
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=df['rank'], y=df['score'], mode='lines', line=dict(color=accent, width=1.5), fill='tozeroy', fillcolor='rgba(0,255,136,0.06)', hoverinfo='skip'))
#     fig.add_trace(go.Scatter(x=df['rank'][:10], y=df['score'][:10], mode='markers', marker=dict(color=accent, size=8, symbol='diamond'), hoverinfo='skip'))
#     fig.update_layout(template='plotly_dark' if dark else 'plotly_white', paper_bgcolor=plot_bg, plot_bgcolor=plot_bg, height=300, margin=dict(l=0, r=10, t=10, b=0), xaxis=dict(gridcolor=grid_color, tickfont=dict(color=dim, size=10), zeroline=False), yaxis=dict(gridcolor=grid_color, tickformat='.0%', tickfont=dict(color=dim, size=10), zeroline=False), showlegend=False)
#     st.plotly_chart(fig, use_container_width=True)
    
#     st.markdown("<hr>", unsafe_allow_html=True)
    
#     col_left, col_right = st.columns([1, 1])
    
#     with col_left:
#         st.markdown('<p class="section-title">Top 5 Candidates</p>', unsafe_allow_html=True)
#         for _, row in df.head(5).iterrows():
#             card_class = "card card-top" if row['rank'] == 1 else "card"
#             st.markdown(f'<div class="{card_class}"><span class="rank-num">#{row["rank"]}</span><span class="rank-id" style="margin-left:8px;">{row["candidate_id"]}</span><span class="rank-score" style="float:right;">{row["score"]:.4f}</span><div style="margin-top:6px;"><span class="rank-reason">{row["reasoning"]}</span></div></div>', unsafe_allow_html=True)
    
#     with col_right:
#         st.markdown('<p class="section-title">Search</p>', unsafe_allow_html=True)
#         search_id = st.text_input("Candidate ID", placeholder="CAND_0081846", label_visibility="collapsed")
#         if search_id:
#             match = df[df['candidate_id'].str.contains(search_id.upper(), na=False)]
#             if len(match) > 0:
#                 for _, row in match.iterrows():
#                     st.markdown(f'<div class="card card-top"><span class="rank-num">#{row["rank"]}</span><span class="rank-id" style="margin-left:8px;">{row["candidate_id"]}</span><span class="rank-score" style="float:right;">{row["score"]:.4f}</span><div style="margin-top:6px;"><span class="rank-reason">{row["reasoning"]}</span></div></div>', unsafe_allow_html=True)
#             else:
#                 st.markdown(f'<p style="color:{dim};">> No match.</p>', unsafe_allow_html=True)
#         else:
#             st.markdown(f'<p style="color:{dim};">> Enter candidate ID to search.</p>', unsafe_allow_html=True)
        
#         st.markdown("<br>", unsafe_allow_html=True)
#         st.markdown('<p class="section-title">Ranks 6–15</p>', unsafe_allow_html=True)
#         for _, row in df.iloc[5:15].iterrows():
#             st.markdown(f'<div class="card"><span class="rank-num">#{row["rank"]}</span><span class="rank-id" style="margin-left:8px;">{row["candidate_id"]}</span><span class="rank-score" style="float:right;">{row["score"]:.4f}</span></div>', unsafe_allow_html=True)
    
#     with st.expander("View Complete Top 100 Table"):
#         st.dataframe(df[['rank', 'candidate_id', 'score', 'reasoning']], use_container_width=True, hide_index=True)

# # ═══════════════════════════════════════════════════════════
# # VIEW: SKILL GRAPH
# # ═══════════════════════════════════════════════════════════

# elif view == "Skill Graph":
#     if skill_data:
#         nodes = skill_data.get("nodes", [])
#         edges = skill_data.get("edges", [])
#         freq = skill_data.get("skill_freq", {})
#         categories = skill_data.get("categories", {})
#         top_skills = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:200]
#         top_skill_names = set(s[0] for s in top_skills)
        
#         G = nx.Graph()
#         for skill, count in top_skills:
#             G.add_node(skill, weight=count)
#         for edge in edges:
#             s1, s2 = nodes[edge["source"]], nodes[edge["target"]]
#             if s1 in top_skill_names and s2 in top_skill_names and edge["weight"] > 0.05:
#                 G.add_edge(s1, s2, weight=edge["weight"])
        
#         pos = nx.spring_layout(G, k=3, iterations=30, seed=42)
#         color_map = {"ranking_ir": accent, "embeddings": "#00aaff", "llm": "#ffaa00", "vector_db": "#ff00aa", "ml_framework": "#00dd88", "data_engineering": "#aaff00", "infrastructure": "#8888ff", "unwanted": "#ff4444", "other": dim}
        
#         edge_x, edge_y = [], []
#         for u, v in G.edges():
#             edge_x.extend([pos[u][0], pos[v][0], None])
#             edge_y.extend([pos[u][1], pos[v][1], None])
        
#         fig = go.Figure()
#         fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(color='rgba(255,255,255,0.05)' if dark else 'rgba(0,0,0,0.05)', width=0.4), hoverinfo='none'))
        
#         node_x, node_y, node_color, node_size, node_text = [], [], [], [], []
#         for node in G.nodes():
#             node_x.append(pos[node][0]); node_y.append(pos[node][1])
#             cat = categories.get(node, "other")
#             node_color.append(color_map.get(cat, dim))
#             node_size.append(max(3, min(25, freq.get(node, 1) / 200)))
#             node_text.append(f"{node}<br>Count: {freq.get(node, 0):,}")
        
#         fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers', marker=dict(size=node_size, color=node_color, line=dict(width=0.3, color=border)), text=node_text, hoverinfo='text'))
        
#         label_x, label_y, label_text = [], [], []
#         for node in G.nodes():
#             if freq.get(node, 0) > 800:
#                 label_x.append(pos[node][0]); label_y.append(pos[node][1]); label_text.append(node)
#         fig.add_trace(go.Scatter(x=label_x, y=label_y, mode='text', text=label_text, textfont=dict(size=8, color=text), hoverinfo='none'))
        
#         fig.update_layout(template='plotly_dark' if dark else 'plotly_white', paper_bgcolor=plot_bg, plot_bgcolor=plot_bg, height=550, margin=dict(l=0, r=0, t=10, b=0), showlegend=False, xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
#         st.plotly_chart(fig, use_container_width=True)
        
#         st.markdown(f'<span style="color:{accent};">● Ranking/IR</span> <span style="color:#00aaff; margin-left:12px;">● Embeddings</span> <span style="color:#ffaa00; margin-left:12px;">● LLM</span> <span style="color:#ff00aa; margin-left:12px;">● Vector DB</span> <span style="color:#aaff00; margin-left:12px;">● Data Eng</span> <span style="color:#ff4444; margin-left:12px;">● Unwanted</span>', unsafe_allow_html=True)
#         st.markdown(f'<span style="color:{dim}; font-size:0.7rem;">Proximity = skill transferability. Node size = frequency in pool.</span>', unsafe_allow_html=True)
#     else:
#         st.markdown(f'<p style="color:{dim};">> Skill graph data not found. Run: python scripts/build_knowledge_graph.py</p>', unsafe_allow_html=True)

# # ═══════════════════════════════════════════════════════════
# # VIEW: CAREER FLOWS
# # ═══════════════════════════════════════════════════════════

# elif view == "Career Flows":
#     if career_data:
#         companies = career_data.get("nodes", [])
#         flows = career_data.get("flows", [])
#         categories = career_data.get("categories", {})
#         company_stats = career_data.get("company_stats", {})
#         top_flows = sorted(flows, key=lambda x: x["count"], reverse=True)[:30]
#         total_moves = sum(f["count"] for f in flows)
#         max_count = top_flows[0]["count"] if top_flows else 1
        
#         c1, c2, c3, c4 = st.columns(4)
#         c1.markdown(f'<div class="stat-box"><div class="stat-val">{len(companies)}</div><div class="stat-lab">Companies</div></div>', unsafe_allow_html=True)
#         c2.markdown(f'<div class="stat-box"><div class="stat-val">{total_moves:,}</div><div class="stat-lab">Transitions</div></div>', unsafe_allow_html=True)
#         c3.markdown(f'<div class="stat-box"><div class="stat-val">{sum(1 for c,cat in categories.items() if cat=="product")}</div><div class="stat-lab">Product Cos</div></div>', unsafe_allow_html=True)
#         c4.markdown(f'<div class="stat-box"><div class="stat-val">{sum(1 for c,cat in categories.items() if cat=="consulting")}</div><div class="stat-lab">Consulting</div></div>', unsafe_allow_html=True)
        
#         st.markdown("<br>", unsafe_allow_html=True)
        
#         col_left, col_right = st.columns([1.2, 1])
        
#         with col_left:
#             st.markdown('<p class="section-title">Talent Migration — Top Paths</p>', unsafe_allow_html=True)
#             for flow in top_flows:
#                 src, tgt, count = companies[flow["source"]], companies[flow["target"]], flow["count"]
#                 src_cat, tgt_cat = categories.get(src, "other"), categories.get(tgt, "other")
#                 src_c = product_color if src_cat == "product" else consulting_color if src_cat == "consulting" else dim
#                 tgt_c = product_color if tgt_cat == "product" else consulting_color if tgt_cat == "consulting" else dim
#                 pct = int(count / max_count * 100)
                
#                 st.markdown(f"""
#                 <div class="flow-row">
#                     <span class="flow-src" style="color:{src_c};">{src[:20]}</span>
#                     <span class="flow-arrow">→</span>
#                     <span class="flow-tgt" style="color:{tgt_c};">{tgt[:20]}</span>
#                     <span class="flow-count" style="color:{accent};">{count}</span>
#                     <div class="flow-bar"><div class="flow-bar-inner" style="width:{pct}%;"></div></div>
#                 </div>
#                 """, unsafe_allow_html=True)
        
#         with col_right:
#             st.markdown('<p class="section-title">Company Rankings</p>', unsafe_allow_html=True)
#             sorted_co = sorted(company_stats.items(), key=lambda x: x[1].get("total_candidates", 0), reverse=True)[:15]
#             max_cand = sorted_co[0][1]["total_candidates"] if sorted_co else 1
            
#             for c, s in sorted_co:
#                 count, cat = s["total_candidates"], s.get("category", "other")
#                 color = product_color if cat == "product" else consulting_color if cat == "consulting" else dim
#                 pct = int(count / max_cand * 100)
#                 st.markdown(f'<span style="color:{color}; font-size:0.75rem;">● {c[:24]}</span> <span style="color:{accent}; font-size:0.75rem; float:right;">{count}</span>', unsafe_allow_html=True)
#                 st.markdown(f'<div style="height:3px; background:{border}; border-radius:2px; margin-bottom:4px;"><div style="width:{pct}%; height:100%; background:{color}; border-radius:2px; opacity:0.4;"></div></div>', unsafe_allow_html=True)
            
#             st.markdown("<br>", unsafe_allow_html=True)
            
#             inflows, outflows = {}, {}
#             for flow in flows:
#                 src, tgt = companies[flow["source"]], companies[flow["target"]]
#                 outflows[src] = outflows.get(src, 0) + flow["count"]
#                 inflows[tgt] = inflows.get(tgt, 0) + flow["count"]
            
#             top_source = max(outflows, key=outflows.get) if outflows else "N/A"
#             top_dest = max(inflows, key=inflows.get) if inflows else "N/A"
            
#             st.markdown(f"""
#             <div style="font-size:0.75rem; color:{text}; line-height:1.7; background:{card_bg}; border:1px solid {border}; padding:16px;">
#                 <p style="color:{accent};"><b>● Key Insights</b></p>
#                 <b>Top Source:</b> <span style="color:{accent};">{top_source}</span><br>
#                 <b>Top Destination:</b> <span style="color:{accent};">{top_dest}</span><br><br>
#                 <b>What This Means:</b> Candidates moving between product companies are most likely to match the JD's preference for product experience over consulting backgrounds.
#             </div>
#             """, unsafe_allow_html=True)
        
#         st.markdown(f'<div style="font-size:0.65rem; color:{dim}; margin-top:16px;"><span style="color:{product_color};">● Product (preferred)</span> <span style="color:{consulting_color}; margin-left:16px;">● Consulting (flagged)</span> <span style="color:{dim}; margin-left:16px;">● Other</span></div>', unsafe_allow_html=True)
#     else:
#         st.markdown(f'<p style="color:{dim};">> Career flow data not found. Run: python scripts/build_knowledge_graph.py</p>', unsafe_allow_html=True)

# # ═══════════════════════════════════════════════════════════
# # VIEW: CANDIDATE MAP
# # ═══════════════════════════════════════════════════════════

# elif view == "Candidate Map":
#     if candidate_data:
#         positions = candidate_data.get("positions_2d", {})
#         pos_list = [{"candidate_id": cid, "x": p["x"], "y": p["y"]} for cid, p in positions.items()]
#         pos_df = pd.DataFrame(pos_list).merge(df[['candidate_id', 'score', 'rank']], on='candidate_id', how='left')
#         pos_df['in_top100'] = pos_df['rank'].notna()
#         pos_df['score'] = pos_df['score'].fillna(0.3)
        
#         col1, col2 = st.columns([2, 1])
        
#         with col1:
#             fig = go.Figure()
#             bg_sample = pos_df[~pos_df['in_top100']].sample(min(5000, len(pos_df)))
#             fig.add_trace(go.Scatter(x=bg_sample['x'], y=bg_sample['y'], mode='markers', marker=dict(size=1.5, color=dim, opacity=0.15), hoverinfo='skip', name='Pool'))
#             top100 = pos_df[pos_df['in_top100']]
#             fig.add_trace(go.Scatter(x=top100['x'], y=top100['y'], mode='markers', marker=dict(size=top100['score'] * 10 + 5, color=top100['score'], colorscale='viridis' if dark else 'greens', line=dict(width=0.5, color='rgba(255,255,255,0.3)'), showscale=True, colorbar=dict(title='Score', tickformat='.0%')), text=[f"Rank {int(r)}: {c}" for c, r in zip(top100['candidate_id'], top100['rank'])], hoverinfo='text', name='Top 100'))
#             fig.update_layout(template='plotly_dark' if dark else 'plotly_white', paper_bgcolor=plot_bg, plot_bgcolor=plot_bg, height=500, margin=dict(l=0, r=10, t=10, b=0), xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
#             st.plotly_chart(fig, use_container_width=True)
        
#         with col2:
#             st.markdown('<p class="section-title">How to Read</p>', unsafe_allow_html=True)
#             st.markdown(f'<div style="font-size:0.78rem; color:{text}; line-height:1.6;"><p style="color:{accent};">● <b>Proximity</b></p>Closer points = similar careers.<br><br><p style="color:{accent};">● <b>Top 100</b></p>Highlighted in bright colors.<br>Score range: {top100["score"].min():.3f} – {top100["score"].max():.3f}<br><br><p style="color:{accent};">● <b>Key Insight</b></p>Strong candidates cluster in regions associated with ranking and production ML — exactly the JD requirements.</div>', unsafe_allow_html=True)
        
#         st.markdown(f'<p style="color:{dim}; font-size:0.65rem; text-align:center; margin-top:8px;">100,000 candidates in 2D embedding space. Hover for details.</p>', unsafe_allow_html=True)
#     else:
#         st.markdown(f'<p style="color:{dim};">> Candidate network data not found. Run: python scripts/build_knowledge_graph.py</p>', unsafe_allow_html=True)

# # ═══════════════════════════════════════════════════════════
# # FOOTER
# # ═══════════════════════════════════════════════════════════

# st.markdown("<hr>", unsafe_allow_html=True)
# m1, m2, m3, m4 = st.columns(4)
# for col, (title, desc) in zip([m1, m2, m3, m4], [
#     ("Career Evidence", "Reads career descriptions, not skills. Semantic embeddings understand meaning beyond keywords."),
#     ("Behavioral Signals", "All 23 platform signals. Ghost profiles penalized. Active candidates boosted."),
#     ("Honeypot Detection", "10 indicators catch impossible profiles. Inverted salaries, timeline errors, mismatches."),
#     ("Evidence-Traced", "Every reasoning references actual profile data. No hallucination. 99/100 unique assessments."),
# ]):
#     col.markdown(f'<p style="color:{accent}; font-weight:500; margin-bottom:4px;">{title}</p>', unsafe_allow_html=True)
#     col.markdown(f'<p style="color:{text}; font-size:0.78rem; line-height:1.5;">{desc}</p>', unsafe_allow_html=True)

# st.markdown("<hr>", unsafe_allow_html=True)
# st.markdown(f'<p class="footer">VANTAGE · CPU-ONLY · 500MB RAM · ZERO API CALLS · 156s PIPELINE · RETRO EDITION</p>', unsafe_allow_html=True)
import streamlit as st
st.set_page_config(page_title="VANTAGE", page_icon="●")
st.title("VANTAGE — Talent Intelligence")
st.caption("Redrob Hackathon Submission")
st.success("✅ Pipeline: 100K candidates | 156s | CPU-only | 500MB RAM")
st.markdown("[GitHub Repo](https://github.com/Shadhai/VANTAGE)")
st.markdown("[Colab Sandbox](https://colab.research.google.com/drive/13AF3UhLz2UvxFHR4UjrSjwUc-GyT3A6R?usp=sharing)")
try:
    import pandas as pd
    df = pd.read_csv("output/submission.csv")
    st.metric("Candidates", len(df))
    st.dataframe(df.head(10), use_container_width=True)
except:
    st.info("Run pipeline to generate output")
