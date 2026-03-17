import requests
import json
import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "C:/Users/gordo/Documents/madness_2026/mm_data")
CURRENT_YEAR = 2026
AVG_TEMPO = 67.5
AVG_EFF = 107.0

TOURNAMENT_TEAMS_2026 = {
    "East": {
        "Duke": 1, "Siena": 16, "Ohio St.": 8, "TCU": 9,
        "St. John's": 5, "Northern Iowa": 12, "Kansas": 4, "Cal Baptist": 13,
        "Louisville": 6, "South Florida": 11, "Michigan St.": 3, "North Dakota St.": 14,
        "UCLA": 7, "UCF": 10, "Connecticut": 2, "Furman": 15,
    },
    "South": {
        "Florida": 1, "Lehigh": 16, "Clemson": 8, "Iowa": 9,
        "Vanderbilt": 5, "McNeese St.": 12, "Nebraska": 4, "Troy": 13,
        "North Carolina": 6, "VCU": 11, "Illinois": 3, "Penn": 14,
        "Saint Mary's": 7, "Texas A&M": 10, "Houston": 2, "Idaho": 15,
    },
    "West": {
        "Arizona": 1, "Long Island": 16, "Villanova": 8, "Utah St.": 9,
        "Wisconsin": 5, "High Point": 12, "Arkansas": 4, "Hawaii": 13,
        "BYU": 6, "Texas": 11, "Gonzaga": 3, "Kennesaw St.": 14,
        "Miami FL": 7, "Missouri": 10, "Purdue": 2, "Queens": 15,
    },
    "Midwest": {
        "Michigan": 1, "Howard": 16, "Georgia": 8, "Saint Louis": 9,
        "Texas Tech": 5, "Akron": 12, "Alabama": 4, "Hofstra": 13,
        "Tennessee": 6, "SMU": 11, "Virginia": 3, "Wright St.": 14,
        "Kentucky": 7, "Santa Clara": 10, "Iowa St.": 2, "Tennessee St.": 15,
    },
    "First Four": {
        "UMBC": 16, "N.C. State": 11, "Prairie View A&M": 16,
        "Miami OH": 11,
    },
}

FEATURE_COLS = [
    "seed_diff", "oe_diff", "de_diff", "eff_margin_diff",
    "barthag_diff", "rank_diff", "win_pct_diff", "tempo_diff",
    "oe_rank_diff", "de_rank_diff",
]


def get_all_teams():
    teams = {}
    for region, team_dict in TOURNAMENT_TEAMS_2026.items():
        for team, seed in team_dict.items():
            teams[team] = {"seed": seed, "region": region}
    return teams


def load_barttorvik_2026():
    path = os.path.join(DATA_DIR, f"barttorvik_{CURRENT_YEAR}.json")
    if not os.path.exists(path):
        url = f"https://barttorvik.com/{CURRENT_YEAR}_team_results.json"
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code == 200:
                os.makedirs(DATA_DIR, exist_ok=True)
                with open(path, "w") as f:
                    f.write(r.text)
        except:
            return {}
    with open(path) as f:
        data = json.load(f)
    teams = {}
    for t in data:
        rec = t[3] if len(t) > 3 else "0-0"
        wins, losses = 0, 0
        if isinstance(rec, str) and '-' in rec:
            parts = rec.split('-')
            wins, losses = int(parts[0]), int(parts[1])
        total = wins + losses if (wins + losses) > 0 else 1
        teams[t[1]] = {
            "rank": t[0],
            "conf": t[2],
            "record": rec,
            "wins": wins,
            "losses": losses,
            "win_pct": wins / total,
            "adj_oe": t[4] if len(t) > 4 else 100.0,
            "adj_oe_rank": t[5] if len(t) > 5 else 180,
            "adj_de": t[6] if len(t) > 6 else 100.0,
            "adj_de_rank": t[7] if len(t) > 7 else 180,
            "barthag": t[8] if len(t) > 8 else 0.5,
            "barthag_rank": t[9] if len(t) > 9 else 180,
            "adj_tempo": t[44] if len(t) > 44 else 67.0,
            "eff_margin": (t[4] - t[6]) if len(t) > 6 else 0.0,
        }
    return teams


def load_or_train_model():
    model_path = os.path.join(DATA_DIR, "trained_model.pkl")
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            m = pickle.load(f)
        return m["lr"], m["lgb"], m["scaler"]
    print("No pre-trained model found. Training from scratch...")
    print("Run march_madness_bracket_predictor.py first, or training inline now.\n")
    return train_inline()


def train_inline():
    from march_madness_bracket_predictor import (
        download_tournament_history, load_barttorvik,
        download_barttorvik, build_training_data, train_model, YEARS
    )
    os.makedirs(DATA_DIR, exist_ok=True)
    bt_by_year = {}
    for year in YEARS + [CURRENT_YEAR]:
        download_barttorvik(year)
        bt_by_year[year] = load_barttorvik(year)
    games = download_tournament_history()
    df = build_training_data(games, bt_by_year)
    lr, lgb_m, scaler = train_model(df)
    model_path = os.path.join(DATA_DIR, "trained_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"lr": lr, "lgb": lgb_m, "scaler": scaler}, f)
    return lr, lgb_m, scaler


def predict_h2h(team_a_name, team_a_seed, team_a_stats, team_b_name, team_b_seed, team_b_stats, lr_model, lgb_model, scaler):
    features = np.array([[
        team_a_seed - team_b_seed,
        team_a_stats["adj_oe"] - team_b_stats["adj_oe"],
        team_a_stats["adj_de"] - team_b_stats["adj_de"],
        team_a_stats["eff_margin"] - team_b_stats["eff_margin"],
        team_a_stats["barthag"] - team_b_stats["barthag"],
        team_a_stats["rank"] - team_b_stats["rank"],
        team_a_stats["win_pct"] - team_b_stats["win_pct"],
        team_a_stats["adj_tempo"] - team_b_stats["adj_tempo"],
        team_a_stats["adj_oe_rank"] - team_b_stats["adj_oe_rank"],
        team_a_stats["adj_de_rank"] - team_b_stats["adj_de_rank"],
    ]])
    X = scaler.transform(features)
    lr_prob = lr_model.predict_proba(X)[0][1]
    if lgb_model:
        lgb_prob = lgb_model.predict_proba(X)[0][1]
        prob_a = 0.65 * lr_prob + 0.35 * lgb_prob
    else:
        prob_a = lr_prob
    return prob_a


def project_score(team_a_stats, team_b_stats):
    tempo_a = team_a_stats["adj_tempo"]
    tempo_b = team_b_stats["adj_tempo"]
    expected_possessions = (tempo_a * tempo_b) / AVG_TEMPO
    ppp_a = (team_a_stats["adj_oe"] * team_b_stats["adj_de"]) / (AVG_EFF * 100)
    ppp_b = (team_b_stats["adj_oe"] * team_a_stats["adj_de"]) / (AVG_EFF * 100)
    score_a = ppp_a * expected_possessions
    score_b = ppp_b * expected_possessions
    return score_a, score_b, expected_possessions


def monte_carlo_scores(score_a_mean, score_b_mean, n_sims=50000, std_dev=11.0):
    scores_a = np.random.normal(score_a_mean, std_dev, n_sims)
    scores_b = np.random.normal(score_b_mean, std_dev, n_sims)
    scores_a = np.maximum(scores_a, 30)
    scores_b = np.maximum(scores_b, 30)
    a_wins = np.sum(scores_a > scores_b)
    b_wins = np.sum(scores_b > scores_a)
    ties = n_sims - a_wins - b_wins
    a_wins += ties // 2
    b_wins += ties - ties // 2
    return {
        "a_win_pct": a_wins / n_sims,
        "b_win_pct": b_wins / n_sims,
        "a_score_mean": np.mean(scores_a),
        "b_score_mean": np.mean(scores_b),
        "a_score_median": np.median(scores_a),
        "b_score_median": np.median(scores_b),
        "a_score_5th": np.percentile(scores_a, 5),
        "a_score_95th": np.percentile(scores_a, 95),
        "b_score_5th": np.percentile(scores_b, 5),
        "b_score_95th": np.percentile(scores_b, 95),
        "margin_mean": np.mean(scores_a - scores_b),
        "margin_std": np.std(scores_a - scores_b),
        "prob_ot": np.mean(np.abs(scores_a - scores_b) < 3),
        "prob_blowout_10": np.mean(np.abs(scores_a - scores_b) > 10),
    }


def display_matchup(team_a_name, team_a_seed, team_a_stats, team_b_name, team_b_seed, team_b_stats, ml_prob, mc_results, proj_a, proj_b, possessions):
    w = 70
    print(f"\n{'='*w}")
    print(f"  HEAD-TO-HEAD MATCHUP PREDICTION")
    print(f"{'='*w}")
    print(f"\n  ({team_a_seed}) {team_a_name}  vs  ({team_b_seed}) {team_b_name}")
    print(f"\n{'='*w}")
    print(f"  WIN PROBABILITY")
    print(f"{'='*w}")
    bar_len = 50
    a_bars = int(ml_prob * bar_len)
    b_bars = bar_len - a_bars
    print(f"\n  ML Model:     {team_a_name} {ml_prob*100:5.1f}%  {'█' * a_bars}{'░' * b_bars}  {(1-ml_prob)*100:5.1f}% {team_b_name}")
    mc_prob = mc_results["a_win_pct"]
    a_bars2 = int(mc_prob * bar_len)
    b_bars2 = bar_len - a_bars2
    print(f"  Monte Carlo:  {team_a_name} {mc_prob*100:5.1f}%  {'█' * a_bars2}{'░' * b_bars2}  {(1-mc_prob)*100:5.1f}% {team_b_name}")
    combined = 0.6 * ml_prob + 0.4 * mc_prob
    a_bars3 = int(combined * bar_len)
    b_bars3 = bar_len - a_bars3
    print(f"  Combined:     {team_a_name} {combined*100:5.1f}%  {'█' * a_bars3}{'░' * b_bars3}  {(1-combined)*100:5.1f}% {team_b_name}")
    print(f"\n{'='*w}")
    print(f"  PROJECTED SCORE (50,000 simulations)")
    print(f"{'='*w}")
    print(f"\n  {'':30s}  {'Proj':>6s}  {'Median':>6s}  {'5th':>6s}  {'95th':>6s}")
    print(f"  {'-'*60}")
    print(f"  ({team_a_seed:>2}) {team_a_name:<25s}  {proj_a:6.1f}  {mc_results['a_score_median']:6.1f}  {mc_results['a_score_5th']:6.1f}  {mc_results['a_score_95th']:6.1f}")
    print(f"  ({team_b_seed:>2}) {team_b_name:<25s}  {proj_b:6.1f}  {mc_results['b_score_median']:6.1f}  {mc_results['b_score_5th']:6.1f}  {mc_results['b_score_95th']:6.1f}")
    print(f"\n  Projected margin: {team_a_name} by {mc_results['margin_mean']:+.1f} pts (std: {mc_results['margin_std']:.1f})")
    print(f"  Expected possessions: {possessions:.1f}")
    print(f"  Probability of close game (<3 pts): {mc_results['prob_ot']*100:.1f}%")
    print(f"  Probability of blowout (>10 pts):   {mc_results['prob_blowout_10']*100:.1f}%")
    print(f"\n{'='*w}")
    print(f"  TEAM COMPARISON (Barttorvik T-Rank)")
    print(f"{'='*w}")
    print(f"\n  {'Metric':<25s}  {team_a_name:>18s}  {team_b_name:>18s}  {'Edge':>10s}")
    print(f"  {'-'*75}")
    stats_compare = [
        ("Record", team_a_stats["record"], team_b_stats["record"], ""),
        ("T-Rank", f"#{team_a_stats['rank']}", f"#{team_b_stats['rank']}", team_a_name if team_a_stats['rank'] < team_b_stats['rank'] else team_b_name),
        ("Adj. Off. Eff.", f"{team_a_stats['adj_oe']:.1f} (#{team_a_stats['adj_oe_rank']})", f"{team_b_stats['adj_oe']:.1f} (#{team_b_stats['adj_oe_rank']})", team_a_name if team_a_stats['adj_oe'] > team_b_stats['adj_oe'] else team_b_name),
        ("Adj. Def. Eff.", f"{team_a_stats['adj_de']:.1f} (#{team_a_stats['adj_de_rank']})", f"{team_b_stats['adj_de']:.1f} (#{team_b_stats['adj_de_rank']})", team_a_name if team_a_stats['adj_de'] < team_b_stats['adj_de'] else team_b_name),
        ("Eff. Margin", f"{team_a_stats['eff_margin']:+.1f}", f"{team_b_stats['eff_margin']:+.1f}", team_a_name if team_a_stats['eff_margin'] > team_b_stats['eff_margin'] else team_b_name),
        ("Barthag (Win%)", f"{team_a_stats['barthag']:.4f}", f"{team_b_stats['barthag']:.4f}", team_a_name if team_a_stats['barthag'] > team_b_stats['barthag'] else team_b_name),
        ("Adj. Tempo", f"{team_a_stats['adj_tempo']:.1f}", f"{team_b_stats['adj_tempo']:.1f}", "Faster" if team_a_stats['adj_tempo'] > team_b_stats['adj_tempo'] else "Slower"),
        ("Win %", f"{team_a_stats['win_pct']*100:.1f}%", f"{team_b_stats['win_pct']*100:.1f}%", team_a_name if team_a_stats['win_pct'] > team_b_stats['win_pct'] else team_b_name),
        ("Conference", team_a_stats["conf"], team_b_stats["conf"], ""),
    ]
    for label, val_a, val_b, edge in stats_compare:
        print(f"  {label:<25s}  {str(val_a):>18s}  {str(val_b):>18s}  {edge:>10s}")
    if combined >= 0.5:
        predicted_winner = team_a_name
        win_pct = combined * 100
    else:
        predicted_winner = team_b_name
        win_pct = (1 - combined) * 100
    print(f"\n{'='*w}")
    print(f"  PREDICTION: ({team_a_seed if predicted_winner == team_a_name else team_b_seed}) {predicted_winner} wins ({win_pct:.1f}%)")
    print(f"  Projected final: {team_a_name} {round(proj_a)}, {team_b_name} {round(proj_b)}")
    print(f"{'='*w}")


def interactive_mode(bt_data, all_teams, lr_model, lgb_model, scaler):
    team_list = sorted(all_teams.keys())
    while True:
        print(f"\n{'='*70}")
        print(f"  2026 TOURNAMENT TEAMS ({len(team_list)} teams)")
        print(f"{'='*70}")
        for i, team in enumerate(team_list):
            info = all_teams[team]
            stats = bt_data.get(team, {})
            rank = stats.get("rank", "?")
            rec = stats.get("record", "?")
            print(f"  {i+1:>3}. ({info['seed']:>2}) {team:<25s} [{info['region']:<8s}] T-Rank #{rank}, {rec}")
        print(f"\n  Enter two team numbers separated by comma (e.g. '1,5'), or 'q' to quit:")
        try:
            user_input = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in ('q', 'quit', 'exit'):
            break
        try:
            parts = user_input.replace(' ', '').split(',')
            idx_a, idx_b = int(parts[0]) - 1, int(parts[1]) - 1
            team_a_name = team_list[idx_a]
            team_b_name = team_list[idx_b]
        except:
            print("  Invalid input. Try again.")
            continue
        team_a_seed = all_teams[team_a_name]["seed"]
        team_b_seed = all_teams[team_b_name]["seed"]
        team_a_stats = bt_data.get(team_a_name)
        team_b_stats = bt_data.get(team_b_name)
        if not team_a_stats:
            print(f"  No stats found for {team_a_name}")
            continue
        if not team_b_stats:
            print(f"  No stats found for {team_b_name}")
            continue
        ml_prob = predict_h2h(
            team_a_name, team_a_seed, team_a_stats,
            team_b_name, team_b_seed, team_b_stats,
            lr_model, lgb_model, scaler
        )
        proj_a, proj_b, possessions = project_score(team_a_stats, team_b_stats)
        mc = monte_carlo_scores(proj_a, proj_b, n_sims=50000)
        display_matchup(
            team_a_name, team_a_seed, team_a_stats,
            team_b_name, team_b_seed, team_b_stats,
            ml_prob, mc, proj_a, proj_b, possessions
        )


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    print("=" * 70)
    print("  MARCH MADNESS 2026 HEAD-TO-HEAD MATCHUP PREDICTOR")
    print("  LightGBM/LogReg Ensemble + Monte Carlo Score Simulation")
    print("=" * 70)
    print("\n[1/3] Loading 2026 team data from Barttorvik...")
    bt_data = load_barttorvik_2026()
    all_teams = get_all_teams()
    found = sum(1 for t in all_teams if t in bt_data)
    print(f"  {found}/{len(all_teams)} tournament teams matched in Barttorvik data")
    print("\n[2/3] Loading trained model...")
    lr_model, lgb_model, scaler = load_or_train_model()
    print("  Model loaded successfully")
    print("\n[3/3] Starting interactive matchup predictor...")
    if len(sys.argv) >= 3:
        team_a_name = sys.argv[1]
        team_b_name = sys.argv[2]
        team_a_stats = bt_data.get(team_a_name)
        team_b_stats = bt_data.get(team_b_name)
        if not team_a_stats or not team_b_stats:
            print(f"  Could not find stats for one or both teams")
            return
        team_a_seed = all_teams.get(team_a_name, {}).get("seed", 8)
        team_b_seed = all_teams.get(team_b_name, {}).get("seed", 8)
        ml_prob = predict_h2h(team_a_name, team_a_seed, team_a_stats,
                              team_b_name, team_b_seed, team_b_stats,
                              lr_model, lgb_model, scaler)
        proj_a, proj_b, poss = project_score(team_a_stats, team_b_stats)
        mc = monte_carlo_scores(proj_a, proj_b, n_sims=50000)
        display_matchup(team_a_name, team_a_seed, team_a_stats,
                        team_b_name, team_b_seed, team_b_stats,
                        ml_prob, mc, proj_a, proj_b, poss)
    else:
        interactive_mode(bt_data, all_teams, lr_model, lgb_model, scaler)


if __name__ == "__main__":
    main()
