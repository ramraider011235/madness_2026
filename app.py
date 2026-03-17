import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.data import load_barttorvik, load_trained_model, get_all_tournament_teams, CURRENT_YEAR

st.set_page_config(
    page_title="March Madness 2026 Predictor",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@400;500;600;700&family=Source+Sans+3:wght@300;400;600;700&display=swap');
    .stApp { background-color: #0a0e17; }
    .main-title {
        font-family: 'Oswald', sans-serif;
        font-weight: 700;
        font-size: 3.2rem;
        background: linear-gradient(135deg, #ff6b35 0%, #f7c948 50%, #ff6b35 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 0;
        line-height: 1.1;
    }
    .sub-title {
        font-family: 'Source Sans 3', sans-serif;
        font-weight: 300;
        color: #8892a4;
        text-align: center;
        font-size: 1.15rem;
        margin-top: 4px;
        letter-spacing: 1px;
    }
    .stat-card {
        background: linear-gradient(145deg, #131a2b 0%, #1a2340 100%);
        border: 1px solid #2a3555;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        transition: transform 0.2s, border-color 0.2s;
    }
    .stat-card:hover {
        transform: translateY(-2px);
        border-color: #ff6b35;
    }
    .stat-number {
        font-family: 'Oswald', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        color: #ff6b35;
        line-height: 1;
    }
    .stat-label {
        font-family: 'Source Sans 3', sans-serif;
        color: #8892a4;
        font-size: 0.9rem;
        margin-top: 6px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    .team-row {
        background: #131a2b;
        border: 1px solid #1e2a45;
        border-radius: 8px;
        padding: 12px 18px;
        margin: 4px 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-family: 'Source Sans 3', sans-serif;
    }
    .seed-badge {
        background: #ff6b35;
        color: #0a0e17;
        font-family: 'Oswald', sans-serif;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
        display: inline-block;
        min-width: 28px;
        text-align: center;
    }
    .region-header {
        font-family: 'Oswald', sans-serif;
        font-weight: 600;
        color: #f7c948;
        font-size: 1.3rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        border-bottom: 2px solid #2a3555;
        padding-bottom: 8px;
        margin-bottom: 12px;
    }
    .info-box {
        background: linear-gradient(145deg, #131a2b 0%, #1a2340 100%);
        border-left: 4px solid #ff6b35;
        border-radius: 0 8px 8px 0;
        padding: 16px 20px;
        font-family: 'Source Sans 3', sans-serif;
        color: #c4cad8;
        margin: 12px 0;
    }
    section[data-testid="stSidebar"] {
        background-color: #0d1220;
        border-right: 1px solid #1e2a45;
    }
    section[data-testid="stSidebar"] .stMarkdown p {
        color: #8892a4;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">March Madness 2026</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">LightGBM + LogReg Ensemble &bull; Monte Carlo Bracket Simulation &bull; Barttorvik T-Rank</div>', unsafe_allow_html=True)
st.markdown("---")

bt_data = load_barttorvik(CURRENT_YEAR)
lr_model, lgb_model, scaler = load_trained_model()
all_teams = get_all_tournament_teams()

model_loaded = lr_model is not None
data_loaded = len(bt_data) > 0

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">68</div>
        <div class="stat-label">Tournament Teams</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">83%</div>
        <div class="stat-label">Model Accuracy</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">15</div>
        <div class="stat-label">Training Seasons</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">10K</div>
        <div class="stat-label">Simulations/Game</div>
    </div>""", unsafe_allow_html=True)

st.markdown("")

if not model_loaded:
    st.error("Trained model not found. Run `python march_madness_bracket_predictor.py` first to generate `mm_data/trained_model.pkl`.")
    st.stop()

st.markdown("")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("""
    <div class="info-box">
        <strong style="color: #ff6b35; font-family: 'Oswald', sans-serif; font-size: 1.1rem; letter-spacing: 1px;">
            🏆 BRACKET PREDICTOR
        </strong><br>
        Simulate the full 2026 bracket with adjustable confidence gears.
        Chalk mode, balanced, or full chaos. 10,000 Monte Carlo sims per game,
        powered by 15 years of Barttorvik efficiency data and 944 historical tournament games.
        <br><br>
        <em>Navigate to the Bracket Predictor page in the sidebar.</em>
    </div>
    """, unsafe_allow_html=True)

with col_right:
    st.markdown("""
    <div class="info-box">
        <strong style="color: #f7c948; font-family: 'Oswald', sans-serif; font-size: 1.1rem; letter-spacing: 1px;">
            ⚔️ HEAD-TO-HEAD MATCHUP
        </strong><br>
        Pick any two of the 68 tournament teams for a deep statistical comparison.
        Win probability from the ML ensemble, projected scores via tempo-efficiency framework,
        and 50,000 Monte Carlo score simulations with full confidence intervals.
        <br><br>
        <em>Navigate to the Head to Head page in the sidebar.</em>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown('<div class="region-header">1-Seeds Overview</div>', unsafe_allow_html=True)

top_seeds = [
    ("East", "Duke", 1), ("South", "Florida", 1),
    ("West", "Arizona", 1), ("Midwest", "Michigan", 1),
]

cols = st.columns(4)
for i, (region, team, seed) in enumerate(top_seeds):
    stats = bt_data.get(team, {})
    with cols[i]:
        st.markdown(f"""
        <div class="stat-card" style="text-align: left;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="seed-badge">{seed}</span>
                <span style="color: #8892a4; font-family: 'Source Sans 3', sans-serif; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px;">{region}</span>
            </div>
            <div style="font-family: 'Oswald', sans-serif; font-size: 1.6rem; color: #e8ecf4; margin: 8px 0 4px 0; font-weight: 600;">{team}</div>
            <div style="font-family: 'Source Sans 3', sans-serif; color: #8892a4; font-size: 0.85rem;">
                {stats.get('record', '?')} &bull; {stats.get('conf', '?')}<br>
                T-Rank #{stats.get('rank', '?')} &bull; Barthag {stats.get('barthag', 0):.4f}<br>
                AdjOE: {stats.get('adj_oe', 0):.1f} (#{stats.get('adj_oe_rank', '?')}) &bull; AdjDE: {stats.get('adj_de', 0):.1f} (#{stats.get('adj_de_rank', '?')})<br>
                <span style="color: #ff6b35; font-weight: 600;">Margin: {stats.get('eff_margin', 0):+.1f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #4a5568; font-family: 'Source Sans 3', sans-serif; font-size: 0.8rem; padding: 20px 0;">
    Data: Barttorvik T-Rank (2010-2026) &bull; Sports-Reference (tournament history) &bull;
    Model: 65% LogReg + 35% LightGBM ensemble &bull; Rolling forward CV
</div>
""", unsafe_allow_html=True)
