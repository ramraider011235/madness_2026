import streamlit as st
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.data import (
    load_barttorvik, load_trained_model, load_tournament_history,
    reconstruct_bracket, get_actual_results, HISTORY_YEARS,
)
from core.model import simulate_region, _get_matchup_prob

st.set_page_config(page_title="Historical Backtest", page_icon="📊", layout="wide")
st.markdown("""
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
    .stat-card {
        background: linear-gradient(145deg, #131a2b 0%, #1a2340 100%);
        border: 1px solid #2a3555;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .stat-number {
        font-family: 'Oswald', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #ff6b35;
        line-height: 1;
    }
    .stat-label {
        font-family: 'Source Sans 3', sans-serif;
        color: #8892a4;
        font-size: 0.85rem;
        margin-top: 6px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    .round-card {
        background: #ffffff;
        border: 1px solid #c4cec0;
        border-radius: 10px;
        padding: 16px;
        margin: 6px 0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }
    .round-header {
        font-family: 'Oswald', sans-serif;
        font-weight: 600;
        font-size: 1rem;
        color: #3b6e3f;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    .correct-pick {
        color: #2d5a1e;
        font-weight: 600;
    }
    .wrong-pick {
        color: #c0392b;
        font-weight: 600;
    }
    .game-row {
        font-family: 'Source Sans 3', sans-serif;
        font-size: 0.9rem;
        padding: 4px 0;
        border-bottom: 1px solid #f0f0f0;
    }
    .region-title {
        font-family: 'Oswald', sans-serif;
        font-weight: 600;
        font-size: 1.2rem;
        color: #3b6e3f;
        letter-spacing: 2px;
        text-transform: uppercase;
        padding: 8px 0;
        border-bottom: 2px solid #c4cec0;
        margin-bottom: 10px;
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
st.markdown('<div class="page-title">📊 Historical Backtest</div>', unsafe_allow_html=True)
st.markdown("Apply the prediction model to past tournaments and compare against actual results.")

lr_model, lgb_model, scaler = load_trained_model()
if lr_model is None:
    st.error("No trained model found. Run the training notebook first.")
    st.stop()

history = load_tournament_history()
if not history:
    st.error("tournament_history.json not found in mm_data/.")
    st.stop()

st.markdown("---")

ROUND_DISPLAY = {
    "R64": "Round of 64",
    "R32": "Round of 32",
    "S16": "Sweet 16",
    "E8": "Elite Eight",
    "F4": "Final Four",
    "NCG": "Championship",
}
ROUND_POINTS = {"R64": 1, "R32": 2, "S16": 4, "E8": 8, "F4": 16, "NCG": 32}
ROUND_ORDER = ["R64", "R32", "S16", "E8", "F4", "NCG"]

col_year, col_sims, col_run = st.columns([2, 1, 1])
with col_year:
    selected_year = st.selectbox("Tournament Year", sorted(HISTORY_YEARS, reverse=True))
with col_sims:
    bt_sims = st.selectbox("Simulations", [500, 1000, 5000], index=1, key="bt_sims")
with col_run:
    st.markdown("<br>", unsafe_allow_html=True)
    run_backtest = st.button("🔬 Run Backtest", type="primary", width='stretch')


def run_historical_prediction(year, n_sims):
    bt_data = load_barttorvik(year)
    if not bt_data:
        return None, "Could not load Barttorvik data for this year."
    regions = reconstruct_bracket(year, history)
    if regions is None:
        return None, "Could not reconstruct bracket for this year."
    actual = get_actual_results(year, history)
    predicted_advancers = {rnd: set() for rnd in ROUND_ORDER}
    region_results = []
    region_champs = []
    for ri, region_teams in enumerate(regions):
        rounds, champion = simulate_region(
            region_teams, bt_data, lr_model, lgb_model, scaler, gear=0, n_sims=n_sims
        )
        region_results.append(rounds)
        if champion:
            region_champs.append(champion)
        for rd_idx, rd_data in enumerate(rounds):
            rnd_key = ROUND_ORDER[rd_idx] if rd_idx < len(ROUND_ORDER) else None
            if rnd_key and rnd_key != "Region Champ":
                for g in rd_data["games"]:
                    predicted_advancers[rnd_key].add(g["winner"])
        if champion:
            predicted_advancers["E8"].add(champion[1])
    if len(region_champs) >= 4:
        prob_cache = {}
        ff_pairs = [(region_champs[0], region_champs[1]), (region_champs[2], region_champs[3])]
        ff_slot = [{} for _ in range(2)]
        ncg_slot = {}
        for _ in range(n_sims):
            ff_w = []
            for fi, ((sa, na), (sb, nb)) in enumerate(ff_pairs):
                prob = _get_matchup_prob(sa, na, sb, nb, bt_data, lr_model, lgb_model, scaler, 0, prob_cache)
                if np.random.random() < prob:
                    ff_w.append((sa, na))
                    ff_slot[fi][(na, nb)] = ff_slot[fi].get((na, nb), 0) + 1
                else:
                    ff_w.append((sb, nb))
                    ff_slot[fi][(nb, na)] = ff_slot[fi].get((nb, na), 0) + 1
            (sa2, na2), (sb2, nb2) = ff_w[0], ff_w[1]
            prob2 = _get_matchup_prob(sa2, na2, sb2, nb2, bt_data, lr_model, lgb_model, scaler, 0, prob_cache)
            if np.random.random() < prob2:
                ncg_slot[na2] = ncg_slot.get(na2, 0) + 1
            else:
                ncg_slot[nb2] = ncg_slot.get(nb2, 0) + 1
        for fi in range(2):
            best = max(ff_slot[fi], key=ff_slot[fi].get)
            predicted_advancers["F4"].add(best[0])
        best_champ = max(ncg_slot, key=ncg_slot.get)
        predicted_advancers["NCG"].add(best_champ)
        for (sa, na), (sb, nb) in ff_pairs:
            predicted_advancers["F4"].add(na)
            predicted_advancers["F4"].add(nb)
    round_scores = {}
    for rnd in ROUND_ORDER:
        actual_winners = actual.get(rnd, set())
        predicted_winners = predicted_advancers.get(rnd, set())
        correct = actual_winners & predicted_winners
        wrong_actual = actual_winners - predicted_winners
        wrong_predicted = predicted_winners - actual_winners
        n_games = len(actual_winners)
        n_correct = len(correct)
        round_scores[rnd] = {
            "correct": sorted(correct),
            "wrong_actual": sorted(wrong_actual),
            "wrong_predicted": sorted(wrong_predicted),
            "n_correct": n_correct,
            "n_games": n_games,
            "pct": n_correct / n_games if n_games > 0 else 0,
            "points": n_correct * ROUND_POINTS[rnd],
            "max_points": n_games * ROUND_POINTS[rnd],
        }
    total_correct = sum(v["n_correct"] for v in round_scores.values())
    total_games = sum(v["n_games"] for v in round_scores.values())
    total_points = sum(v["points"] for v in round_scores.values())
    max_points = sum(v["max_points"] for v in round_scores.values())
    actual_champ = None
    ncg_games = [g for g in history if g["year"] == year and g["round"] == "NCG"]
    if ncg_games:
        actual_champ = ncg_games[0]["w_team"]
    predicted_champ = best_champ if len(region_champs) >= 4 else None
    return {
        "round_scores": round_scores,
        "total_correct": total_correct,
        "total_games": total_games,
        "total_points": total_points,
        "max_points": max_points,
        "actual_champ": actual_champ,
        "predicted_champ": predicted_champ,
        "champ_correct": actual_champ == predicted_champ if actual_champ and predicted_champ else False,
    }, None


if run_backtest or "backtest_results" in st.session_state:
    if run_backtest:
        with st.spinner(f"Simulating {selected_year} tournament ({bt_sims} sims)..."):
            result, error = run_historical_prediction(selected_year, bt_sims)
        if error:
            st.error(error)
            st.stop()
        st.session_state["backtest_results"] = result
        st.session_state["backtest_year"] = selected_year
    result = st.session_state["backtest_results"]
    year = st.session_state.get("backtest_year", selected_year)
    rs = result["round_scores"]
    st.markdown("---")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""<div class="stat-card">
            <div class="stat-number">{year}</div>
            <div class="stat-label">Tournament</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        pct = result["total_correct"] / result["total_games"] * 100 if result["total_games"] > 0 else 0
        st.markdown(f"""<div class="stat-card">
            <div class="stat-number">{pct:.1f}%</div>
            <div class="stat-label">Overall Accuracy</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="stat-card">
            <div class="stat-number">{result['total_correct']}/{result['total_games']}</div>
            <div class="stat-label">Correct Picks</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="stat-card">
            <div class="stat-number">{result['total_points']}</div>
            <div class="stat-label">Bracket Points</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        champ_icon = "✅" if result["champ_correct"] else "❌"
        st.markdown(f"""<div class="stat-card">
            <div class="stat-number">{champ_icon}</div>
            <div class="stat-label">Champion Correct</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("")
    champ_col1, champ_col2 = st.columns(2)
    with champ_col1:
        st.markdown(f"""<div class="round-card">
            <span class="round-header">Predicted Champion:</span>
            <span style="font-size:1.1rem; font-weight:700; color:#f30b45;"> {result['predicted_champ'] or 'N/A'}</span>
        </div>""", unsafe_allow_html=True)
    with champ_col2:
        st.markdown(f"""<div class="round-card">
            <span class="round-header">Actual Champion:</span>
            <span style="font-size:1.1rem; font-weight:700; color:#3b6e3f;"> {result['actual_champ'] or 'N/A'}</span>
        </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="region-title">Per-Round Breakdown</div>', unsafe_allow_html=True)
    round_df_rows = []
    for rnd in ROUND_ORDER:
        s = rs[rnd]
        round_df_rows.append({
            "Round": ROUND_DISPLAY[rnd],
            "Correct": s["n_correct"],
            "Total": s["n_games"],
            "Accuracy": f"{s['pct'] * 100:.1f}%",
            "Points": f"{s['points']}/{s['max_points']}",
        })
    round_df = pd.DataFrame(round_df_rows)
    bar_data = pd.DataFrame([
        {"Round": ROUND_DISPLAY[rnd], "Accuracy": rs[rnd]["pct"] * 100}
        for rnd in ROUND_ORDER
    ])
    st.bar_chart(bar_data, x="Round", y="Accuracy", color="#ff6b35", height=300)
    st.dataframe(round_df, width='stretch', hide_index=True)
    st.markdown("---")
    for rnd in ROUND_ORDER:
        s = rs[rnd]
        with st.expander(f"{ROUND_DISPLAY[rnd]}: {s['n_correct']}/{s['n_games']} correct"):
            if s["correct"]:
                correct_html = " ".join(
                    [f'<span class="correct-pick">✅ {t}</span>' for t in s["correct"]]
                )
                st.markdown(f"**Correct:** {correct_html}", unsafe_allow_html=True)
            if s["wrong_actual"]:
                missed_html = " ".join(
                    [f'<span class="wrong-pick">❌ {t}</span>' for t in s["wrong_actual"]]
                )
                st.markdown(f"**Missed (actual winners):** {missed_html}", unsafe_allow_html=True)
            if s["wrong_predicted"]:
                over_html = " ".join(
                    [f'<span class="wrong-pick">⚠️ {t}</span>' for t in s["wrong_predicted"]]
                )
                st.markdown(f"**Wrong picks (model predicted):** {over_html}", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="region-title">Multi-Year Summary</div>', unsafe_allow_html=True)
    if st.button("Run All Years", type="secondary"):
        all_results = []
        prog = st.progress(0, text="Running backtests...")
        for yi, y in enumerate(HISTORY_YEARS):
            prog.progress((yi + 1) / len(HISTORY_YEARS), text=f"Backtesting {y}...")
            res, err = run_historical_prediction(y, min(bt_sims, 1000))
            if res:
                pct_val = res["total_correct"] / res["total_games"] * 100 if res["total_games"] > 0 else 0
                all_results.append({
                    "Year": y,
                    "Correct": res["total_correct"],
                    "Total": res["total_games"],
                    "Accuracy": f"{pct_val:.1f}%",
                    "Points": res["total_points"],
                    "Champion": "✅" if res["champ_correct"] else "❌",
                    "Predicted": res["predicted_champ"] or "",
                    "Actual": res["actual_champ"] or "",
                })
        prog.empty()
        if all_results:
            summary_df = pd.DataFrame(all_results)
            st.dataframe(summary_df, width='stretch', hide_index=True)
            accs = [float(r["Accuracy"].replace("%", "")) for r in all_results]
            mean_acc = np.mean(accs)
            st.markdown(f"**Mean accuracy across {len(all_results)} tournaments: {mean_acc:.1f}%**")
