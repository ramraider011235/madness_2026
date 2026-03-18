
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from march_madness_2026_core import bracket_probability_table, current_team_names, fit_or_load_models, get_current_team_table, predict_two_teams, simulate_full_bracket

st.set_page_config(page_title="March Madness 2026 Lab", page_icon="🏀", layout="wide")
st.title("March Madness 2026 Lab")
st.caption("Calibrated matchup model, full bracket simulation, current-team head-to-head analysis")

@st.cache_resource
def load_everything(kaggle_dir, cache_dir):
    matchup_model, score_model = fit_or_load_models(model_dir="artifacts/models", kaggle_dir=kaggle_dir or None, cache_dir=cache_dir, force=False)
    team_table = get_current_team_table(year=2026, cache_dir=cache_dir)
    return matchup_model, score_model, team_table

with st.sidebar:
    kaggle_dir = st.text_input("Optional Kaggle data directory", value="")
    cache_dir = st.text_input("Cache directory", value="cache")
    gear = st.slider("Gear", min_value=-2, max_value=2, value=0, step=1)
    recent_form_weight = st.slider("Recent-form weight", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
    program_weight = st.slider("Program-weight", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
    upset_factor = st.slider("Upset-factor", min_value=-1.0, max_value=1.0, value=0.0, step=0.1)
    roster_weight = st.slider("Roster-weight", min_value=0.0, max_value=0.3, value=0.1, step=0.05)
    sims = st.selectbox("Bracket simulations", [5000, 10000, 20000, 50000], index=2)

matchup_model, score_model, team_table = load_everything(kaggle_dir, cache_dir)
model_metrics = matchup_model.get("metrics", {})
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Model mode", matchup_model.get("mode", "unknown"))
with c2:
    st.metric("Training rows", model_metrics.get("n_games", "n/a"))
with c3:
    st.metric("Train log loss", round(model_metrics["log_loss"], 4) if "log_loss" in model_metrics else "n/a")

tab1, tab2, tab3 = st.tabs(["Bracket", "Head to Head", "Model Audit"])

with tab1:
    if st.button("Run bracket simulation", type="primary", use_container_width=True):
        advancement, deterministic = simulate_full_bracket(team_table, matchup_model, n_sims=sims, gear=gear, recent_form_weight=recent_form_weight, program_weight=program_weight, upset_factor=upset_factor)
        st.session_state["advancement"] = advancement
        st.session_state["deterministic"] = deterministic
    if "advancement" in st.session_state:
        advancement = st.session_state["advancement"]
        deterministic = st.session_state["deterministic"]
        st.subheader(f"Champion pick: {deterministic['champion']}")
        df = bracket_probability_table(advancement)
        st.dataframe(df, use_container_width=True, hide_index=True)
        fig = go.Figure()
        show = df.head(16)
        fig.add_trace(go.Bar(x=show["team"], y=show["Champ"]))
        fig.update_layout(title="Title probability", xaxis_title="", yaxis_title="Probability")
        st.plotly_chart(fig, use_container_width=True)
        st.json(deterministic)

with tab2:
    names = current_team_names()
    left, right = st.columns(2)
    with left:
        team_a = st.selectbox("Team A", names, index=names.index("Duke") if "Duke" in names else 0)
    with right:
        team_b = st.selectbox("Team B", names, index=names.index("Houston") if "Houston" in names else 1)
    if team_a != team_b and st.button("Run matchup", use_container_width=True):
        result = predict_two_teams(team_a, team_b, team_table, matchup_model, score_model, recent_form_weight=recent_form_weight, program_weight=program_weight, roster_weight=roster_weight, gear=gear, upset_factor=upset_factor)
        st.session_state["h2h"] = result
    if "h2h" in st.session_state:
        result = st.session_state["h2h"]
        score = result["score"]
        a, b = st.columns(2)
        with a:
            st.metric(f"{result['team_a']} win %", f"{result['prob_team_a']*100:.1f}%")
            st.metric(f"{result['team_a']} score", f"{score['team1_score_mean']:.1f}")
        with b:
            st.metric(f"{result['team_b']} win %", f"{result['prob_team_b']*100:.1f}%")
            st.metric(f"{result['team_b']} score", f"{score['team2_score_mean']:.1f}")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=score["team1_scores"], nbinsx=50, name=result["team_a"], opacity=0.6))
        fig.add_trace(go.Histogram(x=score["team2_scores"], nbinsx=50, name=result["team_b"], opacity=0.6))
        fig.update_layout(barmode="overlay", title="Simulated score distribution")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(pd.DataFrame(result["contributions"]), use_container_width=True, hide_index=True)

with tab3:
    st.json({"matchup_model": {k: v for k, v in matchup_model.items() if isinstance(v, (int, float, str, dict))}, "score_model": {k: v for k, v in score_model.items() if isinstance(v, (int, float, str, dict))}})
