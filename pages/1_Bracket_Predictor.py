import streamlit as st
import sys
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.data import load_barttorvik, load_trained_model, BRACKET_2026, CURRENT_YEAR
from core.model import predict_matchup, apply_gear, simulate_region

st.set_page_config(page_title="Bracket Predictor", page_icon="🏆", layout="wide")
format_a = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@400;500;600;700&family=Source+Sans+3:wght@300;400;600;700&display=swap');
    .page-title {
        font-family: 'Oswald', sans-serif;
        font-weight: 700;
        font-size: 2.4rem;
        color: #f30b45;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    .region-title {
        font-family: 'Oswald', sans-serif;
        font-weight: 600;
        font-size: 1.4rem;
        color: #3b6e3f;
        letter-spacing: 2px;
        text-transform: uppercase;
        padding: 8px 0;
        border-bottom: 2px solid #c4cec0;
        margin-bottom: 10px;
    }
    .game-card {
        background: #ffffff;
        border: 1px solid #c4cec0;
        border-radius: 8px;
        padding: 10px 16px;
        margin: 5px 0;
        font-family: 'Source Sans 3', sans-serif;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }
    .winner-name {
        color: #2d5a1e;
        font-weight: 700;
        font-size: 1rem;
    }
    .loser-name {
        color: #888888;
        font-size: 0.9rem;
    }
    .pct-bar {
        height: 6px;
        border-radius: 3px;
        background: #dde3da;
        margin: 4px 0;
        overflow: hidden;
    }
    .pct-fill {
        height: 100%;
        border-radius: 3px;
        background: linear-gradient(90deg, #f30b45, #94a68e);
    }
    .seed-tag {
        background: #f30b45;
        color: #ffffff;
        font-family: 'Oswald', sans-serif;
        font-weight: 600;
        padding: 1px 6px;
        border-radius: 3px;
        font-size: 0.75rem;
        display: inline-block;
        min-width: 22px;
        text-align: center;
    }
    .champion-box {
        background: #ffffff;
        border: 2px solid #3b6e3f;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
    }
    .champion-label {
        font-family: 'Oswald', sans-serif;
        font-weight: 700;
        font-size: 2rem;
        color: #f30b45;
        letter-spacing: 2px;
    }
    .round-header {
        font-family: 'Oswald', sans-serif;
        font-weight: 500;
        color: #555555;
        font-size: 0.85rem;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin: 12px 0 6px 0;
    }
    .ff-card {
        background: #ffffff;
        border: 1px solid #3b6e3f;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        margin: 8px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    section[data-testid="stSidebar"] {
        background-color: #0d1220;
        border-right: 1px solid #1e2a45;
    }
    section[data-testid="stSidebar"] .stMarkdown p {
        color: #8892a4;
    }
</style>
"""
st.markdown(format_a, unsafe_allow_html=True)
st.markdown('<div class="page-title">🏆 Bracket Predictor</div>', unsafe_allow_html=True)

bt_data = load_barttorvik(CURRENT_YEAR)
lr_model, lgb_model, scaler = load_trained_model()

if lr_model is None:
    st.error("No trained model found. Run `python march_madness_bracket_predictor.py` first.")
    st.stop()

st.markdown("---")

col_gear, col_sims, col_run = st.columns([2, 1, 1])

with col_gear:
    gear_labels = {
        -2: "Heavy Chalk",
        -1: "Mild Chalk",
        0: "Balanced",
        1: "Mild Upset",
        2: "Chaos Mode",
    }
    gear = st.select_slider(
        "Confidence Gear",
        options=[-2, -1, 0, 1, 2],
        value=0,
        format_func=lambda x: f"{x:+d}  {gear_labels[x]}",
    )

with col_sims:
    n_sims = st.selectbox("Simulations per game", [1000, 5000, 10000, 25000], index=2)

with col_run:
    st.markdown("<br>", unsafe_allow_html=True)
    run_bracket = st.button("🚀 Simulate Bracket", type="primary", use_container_width=True)

gear_desc = {
    -2: "Strongly favoring higher seeds. Minimal upset potential.",
    -1: "Slightly favoring higher seeds over model output.",
    0: "Pure model output. No adjustment.",
    1: "Probabilities compressed toward 50/50. More upsets.",
    2: "Maximum chaos. Every game is a coin flip area.",
}
st.caption(f"Gear {gear:+d}: {gear_desc[gear]}")

if run_bracket or "bracket_results" in st.session_state:
    if run_bracket:
        np.random.seed(None)
        region_winners = {}
        region_rounds = {}
        progress = st.progress(0, text="Simulating bracket...")
        for i, (region_name, teams) in enumerate(BRACKET_2026.items()):
            progress.progress((i + 1) / 4, text=f"Simulating {region_name} region...")
            rounds, champion = simulate_region(teams, bt_data, lr_model, lgb_model, scaler, gear=gear, n_sims=n_sims)
            region_rounds[region_name] = rounds
            region_winners[region_name] = champion
        progress.empty()
        ff_matchups = [
            (region_winners["East"], region_winners["South"]),
            (region_winners["West"], region_winners["Midwest"]),
        ]
        ff_results = []
        ff_winners = []
        for (sa, na), (sb, nb) in ff_matchups:
            stats_a, stats_b = bt_data.get(na), bt_data.get(nb)
            if stats_a and stats_b:
                prob = apply_gear(predict_matchup(stats_a, sa, stats_b, sb, lr_model, lgb_model, scaler), gear)
            else:
                prob = 0.5
            a_wins = sum(1 for _ in range(n_sims) if np.random.random() < prob)
            a_pct = a_wins / n_sims
            if a_pct >= 0.5:
                ff_winners.append((sa, na))
                ff_results.append({"winner": na, "w_seed": sa, "w_pct": a_pct, "loser": nb, "l_seed": sb, "l_pct": 1 - a_pct})
            else:
                ff_winners.append((sb, nb))
                ff_results.append({"winner": nb, "w_seed": sb, "w_pct": 1 - a_pct, "loser": na, "l_seed": sa, "l_pct": a_pct})
        (sa, na), (sb, nb) = ff_winners[0], ff_winners[1]
        stats_a, stats_b = bt_data.get(na), bt_data.get(nb)
        if stats_a and stats_b:
            prob = apply_gear(predict_matchup(stats_a, sa, stats_b, sb, lr_model, lgb_model, scaler), gear)
        else:
            prob = 0.5
        a_wins = sum(1 for _ in range(n_sims) if np.random.random() < prob)
        a_pct = a_wins / n_sims
        if a_pct >= 0.5:
            champion = (sa, na)
            ncg_result = {"winner": na, "w_seed": sa, "w_pct": a_pct, "loser": nb, "l_seed": sb, "l_pct": 1 - a_pct}
        else:
            champion = (sb, nb)
            ncg_result = {"winner": nb, "w_seed": sb, "w_pct": 1 - a_pct, "loser": na, "l_seed": sa, "l_pct": a_pct}
        st.session_state["bracket_results"] = {
            "region_rounds": region_rounds,
            "region_winners": region_winners,
            "ff_results": ff_results,
            "ncg_result": ncg_result,
            "champion": champion,
            "gear": gear,
        }
    results = st.session_state["bracket_results"]
    region_rounds = results["region_rounds"]
    region_winners = results["region_winners"]
    ff_results = results["ff_results"]
    ncg_result = results["ncg_result"]
    champion = results["champion"]
    st.markdown(f"""
    <div class="champion-box">
        <div style="font-family: 'Oswald', sans-serif; color: #555555; font-size: 0.9rem; letter-spacing: 2px; text-transform: uppercase;">
            2026 National Champion (Gear {results['gear']:+d})
        </div>
        <div class="champion-label">🏆 ({champion[0]}) {champion[1]}</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")
    st.markdown('<div class="region-title">🏟️ Final Four</div>', unsafe_allow_html=True)
    from core.bracket_svg import build_final_four_svg, build_bracket_svg
    ff_svg = build_final_four_svg(ff_results, ncg_result, champion)
    st.markdown(ff_svg, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="region-title">📐 Regional Brackets</div>', unsafe_allow_html=True)
    tab_east, tab_south, tab_west, tab_midwest = st.tabs(["🔵 East", "🔴 South", "🟢 West", "🟡 Midwest"])
    region_tabs = {"East": tab_east, "South": tab_south, "West": tab_west, "Midwest": tab_midwest}
    region_colors = {"East": "#3b82f6", "South": "#ef4444", "West": "#22c55e", "Midwest": "#d97706"}
    for region_name, tab in region_tabs.items():
        with tab:
            color = region_colors[region_name]
            bracket_svg = build_bracket_svg(region_name, BRACKET_2026[region_name], results=region_rounds[region_name], color=color)
            st.markdown(bracket_svg, unsafe_allow_html=True)
