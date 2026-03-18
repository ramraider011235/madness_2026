import streamlit as st
import sys
import os
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.data import load_barttorvik, load_trained_model, get_all_tournament_teams, CURRENT_YEAR
from core.model import predict_matchup, project_score, monte_carlo_scores

st.set_page_config(page_title="Head-to-Head", page_icon="⚔️", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@400;500;600;700&family=Source+Sans+3:wght@300;400;600;700&display=swap');
    .page-title {
        font-family: 'Oswald', sans-serif;
        font-weight: 700;
        font-size: 2.4rem;
        color: #3b6e3f;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    .vs-box {
        background: #ffffff;
        border: 1px solid #c4cec0;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .team-header {
        font-family: 'Oswald', sans-serif;
        font-weight: 700;
        font-size: 1.8rem;
        letter-spacing: 1px;
    }
    .prob-display {
        font-family: 'Oswald', sans-serif;
        font-weight: 700;
        font-size: 3rem;
        line-height: 1;
    }
    .prob-bar-container {
        height: 16px;
        background: #dde3da;
        border-radius: 8px;
        overflow: hidden;
        margin: 8px 0;
    }
    .prob-bar-a {
        height: 100%;
        border-radius: 8px 0 0 8px;
        float: left;
    }
    .score-box {
        background: #ffffff;
        border: 1px solid #c4cec0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .score-big {
        font-family: 'Oswald', sans-serif;
        font-weight: 700;
        font-size: 2.6rem;
        line-height: 1;
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
    section[data-testid="stSidebar"] {
        background-color: #0d1220;
        border-right: 1px solid #1e2a45;
    }
    section[data-testid="stSidebar"] .stMarkdown p {
        color: #8892a4;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-title">⚔️ Head-to-Head Matchup</div>', unsafe_allow_html=True)

bt_data = load_barttorvik(CURRENT_YEAR)
lr_model, lgb_model, scaler = load_trained_model()
all_teams = get_all_tournament_teams()

if lr_model is None:
    st.error("No trained model found. Run the bracket predictor script first.")
    st.stop()

team_names = sorted(all_teams.keys())
team_display = [f"({all_teams[t]['seed']:>2d}) {t}  [{all_teams[t]['region']}]" for t in team_names]
team_lookup = {d: n for d, n in zip(team_display, team_names)}

st.markdown("---")

col_a, col_vs, col_b = st.columns([5, 1, 5])

with col_a:
    sel_a = st.selectbox("Team A", team_display, index=team_names.index("Duke") if "Duke" in team_names else 0)
with col_vs:
    st.markdown("<div style='text-align: center; padding-top: 28px; font-family: Oswald; font-size: 1.8rem; color: #f30b45; font-weight: 700;'>VS</div>", unsafe_allow_html=True)
with col_b:
    default_b = team_names.index("Arizona") if "Arizona" in team_names else 1
    sel_b = st.selectbox("Team B", team_display, index=default_b)

team_a = team_lookup[sel_a]
team_b = team_lookup[sel_b]

if team_a == team_b:
    st.warning("Select two different teams.")
    st.stop()

a_seed = all_teams[team_a]["seed"]
b_seed = all_teams[team_b]["seed"]
a_stats = bt_data.get(team_a)
b_stats = bt_data.get(team_b)

if not a_stats or not b_stats:
    st.error("Stats not found for one or both teams.")
    st.stop()

run = st.button("⚡ Run Matchup Analysis", type="primary", width='stretch')

if run or "h2h_results" in st.session_state and st.session_state.get("h2h_teams") == (team_a, team_b):
    if run:
        ml_prob = predict_matchup(a_stats, a_seed, b_stats, b_seed, lr_model, lgb_model, scaler)
        proj_a, proj_b, poss = project_score(a_stats, b_stats)
        mc = monte_carlo_scores(proj_a, proj_b, n_sims=50000)
        combined = 0.6 * ml_prob + 0.4 * mc["a_win_pct"]
        st.session_state["h2h_results"] = {
            "ml_prob": ml_prob, "mc": mc, "combined": combined,
            "proj_a": proj_a, "proj_b": proj_b, "poss": poss,
        }
        st.session_state["h2h_teams"] = (team_a, team_b)
    r = st.session_state["h2h_results"]
    ml_prob = r["ml_prob"]
    mc = r["mc"]
    combined = r["combined"]
    proj_a = r["proj_a"]
    proj_b = r["proj_b"]
    poss = r["poss"]
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; margin: 10px 0;">
        <span class="team-header" style="color: #2d5a1e;">({a_seed}) {team_a}</span>
        <span style="color: #888888; font-family: 'Oswald'; font-size: 1.2rem; margin: 0 16px;">vs</span>
        <span class="team-header" style="color: #f30b45;">({b_seed}) {team_b}</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")
    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown(f"""
        <div class="vs-box">
            <div style="color: #555555; font-family: 'Source Sans 3'; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px;">ML Model</div>
            <div class="prob-display" style="color: #2d5a1e;">{ml_prob * 100:.1f}%</div>
            <div style="color: #555555; font-size: 0.85rem;">{team_a}</div>
            <div class="prob-bar-container">
                <div class="prob-bar-a" style="width: {ml_prob * 100:.0f}%; background: linear-gradient(90deg, #2d5a1e, #94a68e);"></div>
            </div>
            <div class="prob-display" style="color: #f30b45;">{(1 - ml_prob) * 100:.1f}%</div>
            <div style="color: #555555; font-size: 0.85rem;">{team_b}</div>
        </div>
        """, unsafe_allow_html=True)
    with p2:
        st.markdown(f"""
        <div class="vs-box" style="border-color: #f30b45;">
            <div style="color: #f30b45; font-family: 'Oswald'; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px;">Combined</div>
            <div class="prob-display" style="color: {'#2d5a1e' if combined >= 0.5 else '#f30b45'};">{max(combined, 1 - combined) * 100:.1f}%</div>
            <div style="color: #000000; font-family: 'Oswald'; font-size: 1.1rem; font-weight: 600;">
                {team_a if combined >= 0.5 else team_b} WINS
            </div>
            <div class="prob-bar-container" style="height: 20px; margin: 12px 0;">
                <div class="prob-bar-a" style="width: {combined * 100:.0f}%; background: linear-gradient(90deg, #2d5a1e, #94a68e);"></div>
            </div>
            <div style="color: #555555; font-size: 0.8rem;">60% ML + 40% Monte Carlo</div>
        </div>
        """, unsafe_allow_html=True)
    with p3:
        st.markdown(f"""
        <div class="vs-box">
            <div style="color: #555555; font-family: 'Source Sans 3'; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px;">Monte Carlo (50K sims)</div>
            <div class="prob-display" style="color: #2d5a1e;">{mc['a_win_pct'] * 100:.1f}%</div>
            <div style="color: #555555; font-size: 0.85rem;">{team_a}</div>
            <div class="prob-bar-container">
                <div class="prob-bar-a" style="width: {mc['a_win_pct'] * 100:.0f}%; background: linear-gradient(90deg, #2d5a1e, #94a68e);"></div>
            </div>
            <div class="prob-display" style="color: #f30b45;">{mc['b_win_pct'] * 100:.1f}%</div>
            <div style="color: #555555; font-size: 0.85rem;">{team_b}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("")
    sc1, sc2, sc3, sc4 = st.columns([3, 3, 2, 2])
    with sc1:
        st.markdown(f"""
        <div class="score-box">
            <div style="color: #555555; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px;">Projected Score</div>
            <div class="score-big" style="color: #2d5a1e;">{proj_a:.0f}</div>
            <div style="color: #555555; font-size: 0.85rem;">{team_a}</div>
            <div style="color: #888888; font-size: 0.75rem; margin-top: 4px;">
                90% CI: {mc['a_5th']:.0f} - {mc['a_95th']:.0f}
            </div>
        </div>
        """, unsafe_allow_html=True)
    with sc2:
        st.markdown(f"""
        <div class="score-box">
            <div style="color: #555555; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px;">Projected Score</div>
            <div class="score-big" style="color: #f30b45;">{proj_b:.0f}</div>
            <div style="color: #555555; font-size: 0.85rem;">{team_b}</div>
            <div style="color: #888888; font-size: 0.75rem; margin-top: 4px;">
                90% CI: {mc['b_5th']:.0f} - {mc['b_95th']:.0f}
            </div>
        </div>
        """, unsafe_allow_html=True)
    with sc3:
        st.markdown(f"""
        <div class="score-box">
            <div style="color: #555555; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px;">Margin</div>
            <div class="score-big" style="color: #f30b45;">{mc['margin_mean']:+.1f}</div>
            <div style="color: #555555; font-size: 0.8rem;">Std: ±{mc['margin_std']:.1f}</div>
        </div>
        """, unsafe_allow_html=True)
    with sc4:
        st.markdown(f"""
        <div class="score-box">
            <div style="color: #555555; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px;">Possessions</div>
            <div class="score-big" style="color: #3b6e3f;">{poss:.0f}</div>
            <div style="color: #555555; font-size: 0.8rem;">Expected</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("")
    tab_viz, tab_stats, tab_game = st.tabs(["📊 Score Distributions", "📋 Stat Comparison", "🎯 Game Factors"])
    with tab_viz:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=mc["scores_a"], nbinsx=60, name=team_a,
            marker_color="rgba(45, 90, 30, 0.6)",
            histnorm="probability density",
        ))
        fig.add_trace(go.Histogram(
            x=mc["scores_b"], nbinsx=60, name=team_b,
            marker_color="rgba(243, 11, 69, 0.5)",
            histnorm="probability density",
        ))
        fig.add_vline(x=proj_a, line_dash="dash", line_color="#2d5a1e", line_width=2, annotation_text=f"{team_a}: {proj_a:.0f}")
        fig.add_vline(x=proj_b, line_dash="dash", line_color="#f30b45", line_width=2, annotation_text=f"{team_b}: {proj_b:.0f}")
        fig.update_layout(
            title="Projected Score Distributions (50,000 simulations)",
            xaxis_title="Points", yaxis_title="Density",
            barmode="overlay",
            paper_bgcolor="#e8e8e3", plot_bgcolor="#ffffff",
            font=dict(family="Source Sans 3", color="#000000"),
            height=400,
        )
        st.plotly_chart(fig, width='stretch')
        margins = mc["scores_a"] - mc["scores_b"]
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=margins, nbinsx=80, name="Margin",
            marker_color="rgba(148, 166, 142, 0.7)",
            histnorm="probability density",
        ))
        fig2.add_vline(x=0, line_color="#000000", line_width=2)
        fig2.add_vline(x=mc["margin_mean"], line_dash="dash", line_color="#f30b45", line_width=2,
                       annotation_text=f"Mean: {mc['margin_mean']:+.1f}")
        fig2.add_vrect(x0=-3, x1=3, fillcolor="rgba(243, 11, 69, 0.08)", line_width=0,
                       annotation_text="Close game zone")
        fig2.update_layout(
            title=f"Margin Distribution ({team_a} minus {team_b})",
            xaxis_title="Point Margin", yaxis_title="Density",
            paper_bgcolor="#e8e8e3", plot_bgcolor="#ffffff",
            font=dict(family="Source Sans 3", color="#000000"),
            height=400,
        )
        st.plotly_chart(fig2, width='stretch')
    with tab_stats:
        metrics = [
            ("Record", a_stats["record"], b_stats["record"], None),
            ("T-Rank", f"#{a_stats['rank']}", f"#{b_stats['rank']}", "lower"),
            ("Adj. Offensive Eff.", f"{a_stats['adj_oe']:.1f}", f"{b_stats['adj_oe']:.1f}", "higher"),
            ("Adj. Defensive Eff.", f"{a_stats['adj_de']:.1f}", f"{b_stats['adj_de']:.1f}", "lower"),
            ("Efficiency Margin", f"{a_stats['eff_margin']:+.1f}", f"{b_stats['eff_margin']:+.1f}", "higher"),
            ("Barthag (Win Prob)", f"{a_stats['barthag']:.4f}", f"{b_stats['barthag']:.4f}", "higher"),
            ("Adj. Tempo", f"{a_stats['adj_tempo']:.1f}", f"{b_stats['adj_tempo']:.1f}", None),
            ("Win %", f"{a_stats['win_pct'] * 100:.1f}%", f"{b_stats['win_pct'] * 100:.1f}%", "higher"),
            ("Conference", a_stats["conf"], b_stats["conf"], None),
        ]
        comparison_data = []
        for label, val_a, val_b, better in metrics:
            edge = ""
            if better:
                try:
                    na = float(val_a.replace('#', '').replace('%', '').replace('+', ''))
                    nb = float(val_b.replace('#', '').replace('%', '').replace('+', ''))
                    if better == "higher":
                        edge = team_a if na > nb else team_b if nb > na else "Even"
                    else:
                        edge = team_a if na < nb else team_b if nb < na else "Even"
                except Exception as e:
                    print(e)
                    pass
            comparison_data.append({"Metric": label, team_a: val_a, team_b: val_b, "Edge": edge})
        comp_df = pd.DataFrame(comparison_data)
        st.dataframe(comp_df, width='stretch', hide_index=True)
        categories = ["Adj OE", "Adj DE\n(inverted)", "Eff Margin", "Barthag\n(x100)", "Win%\n(x100)"]
        a_radar = [a_stats["adj_oe"], 200 - a_stats["adj_de"], a_stats["eff_margin"], a_stats["barthag"] * 100, a_stats["win_pct"] * 100]
        b_radar = [b_stats["adj_oe"], 200 - b_stats["adj_de"], b_stats["eff_margin"], b_stats["barthag"] * 100, b_stats["win_pct"] * 100]
        fig3 = go.Figure()
        fig3.add_trace(go.Scatterpolar(
            r=a_radar, theta=categories, fill='toself', name=team_a,
            line_color='#2d5a1e', fillcolor='rgba(45, 90, 30, 0.2)',
        ))
        fig3.add_trace(go.Scatterpolar(
            r=b_radar, theta=categories, fill='toself', name=team_b,
            line_color='#f30b45', fillcolor='rgba(243, 11, 69, 0.15)',
        ))
        fig3.update_layout(
            polar=dict(bgcolor="#ffffff", radialaxis=dict(visible=True, gridcolor="#dde3da")),
            paper_bgcolor="#e8e8e3",
            font=dict(family="Source Sans 3", color="#000000"),
            title="Team Profile Radar",
            height=450,
        )
        st.plotly_chart(fig3, width='stretch')
    with tab_game:
        g1, g2, g3, g4 = st.columns(4)
        with g1:
            st.metric("P(Close Game < 3pts)", f"{mc['prob_close'] * 100:.1f}%")
        with g2:
            st.metric("P(Blowout > 10pts)", f"{mc['prob_blowout'] * 100:.1f}%")
        with g3:
            pace = "Fast" if poss > 70 else "Moderate" if poss > 65 else "Slow"
            st.metric("Pace", f"{pace} ({poss:.0f} poss)")
        with g4:
            favorite = team_a if combined >= 0.5 else team_b
            st.metric("Predicted Winner", favorite)
        st.markdown("---")
        st.markdown("**Tempo Analysis**")
        tempo_diff = a_stats["adj_tempo"] - b_stats["adj_tempo"]
        if abs(tempo_diff) < 2:
            st.info(f"Both teams play at similar tempos ({a_stats['adj_tempo']:.1f} vs {b_stats['adj_tempo']:.1f}). Expect a neutral pace game.")
        elif tempo_diff > 0:
            st.info(f"{team_a} prefers a faster pace ({a_stats['adj_tempo']:.1f} vs {b_stats['adj_tempo']:.1f}). If they control tempo, scoring should increase.")
        else:
            st.info(f"{team_b} prefers a faster pace ({b_stats['adj_tempo']:.1f} vs {a_stats['adj_tempo']:.1f}). If they control tempo, scoring should increase.")
        st.markdown("**Efficiency Breakdown**")
        oe_edge = team_a if a_stats["adj_oe"] > b_stats["adj_oe"] else team_b
        de_edge = team_a if a_stats["adj_de"] < b_stats["adj_de"] else team_b
        st.info(f"**Offensive edge:** {oe_edge} ({max(a_stats['adj_oe'], b_stats['adj_oe']):.1f} vs {min(a_stats['adj_oe'], b_stats['adj_oe']):.1f} AdjOE)")
        st.info(f"**Defensive edge:** {de_edge} ({min(a_stats['adj_de'], b_stats['adj_de']):.1f} vs {max(a_stats['adj_de'], b_stats['adj_de']):.1f} AdjDE)")
        margin_edge = a_stats["eff_margin"] - b_stats["eff_margin"]
        if abs(margin_edge) < 3:
            st.success("These teams are extremely closely matched. Expect a competitive game.")
        elif abs(margin_edge) < 8:
            better = team_a if margin_edge > 0 else team_b
            st.warning(f"{better} has a meaningful efficiency advantage (+{abs(margin_edge):.1f} margin differential).")
        else:
            better = team_a if margin_edge > 0 else team_b
            st.error(f"{better} has a dominant efficiency edge (+{abs(margin_edge):.1f} margin differential). Major upset territory for the underdog.")
