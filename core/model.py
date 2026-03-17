import numpy as np

FEATURE_COLS = [
    "seed_diff", "oe_diff", "de_diff", "eff_margin_diff",
    "barthag_diff", "rank_diff", "win_pct_diff", "tempo_diff",
    "oe_rank_diff", "de_rank_diff",
]
AVG_TEMPO = 67.5
AVG_EFF = 107.0


def predict_matchup(a_stats, a_seed, b_stats, b_seed, lr_model, lgb_model, scaler):
    features = np.array([[
        a_seed - b_seed,
        a_stats["adj_oe"] - b_stats["adj_oe"],
        a_stats["adj_de"] - b_stats["adj_de"],
        a_stats["eff_margin"] - b_stats["eff_margin"],
        a_stats["barthag"] - b_stats["barthag"],
        a_stats["rank"] - b_stats["rank"],
        a_stats["win_pct"] - b_stats["win_pct"],
        a_stats["adj_tempo"] - b_stats["adj_tempo"],
        a_stats["adj_oe_rank"] - b_stats["adj_oe_rank"],
        a_stats["adj_de_rank"] - b_stats["adj_de_rank"],
    ]])
    X = scaler.transform(features)
    lr_prob = lr_model.predict_proba(X)[0][1]
    if lgb_model:
        lgb_prob = lgb_model.predict_proba(X)[0][1]
        return 0.65 * lr_prob + 0.35 * lgb_prob
    return lr_prob


def apply_gear(prob, gear):
    if gear == 0:
        return prob
    logit = np.log(np.clip(prob, 1e-6, 1 - 1e-6) / (1 - np.clip(prob, 1e-6, 1 - 1e-6)))
    temperature = 1.0 + gear * 0.3
    return 1.0 / (1.0 + np.exp(-logit / temperature))


def project_score(a_stats, b_stats):
    expected_poss = (a_stats["adj_tempo"] * b_stats["adj_tempo"]) / AVG_TEMPO
    ppp_a = (a_stats["adj_oe"] * b_stats["adj_de"]) / (AVG_EFF * 100)
    ppp_b = (b_stats["adj_oe"] * a_stats["adj_de"]) / (AVG_EFF * 100)
    return ppp_a * expected_poss, ppp_b * expected_poss, expected_poss


def monte_carlo_scores(score_a_mean, score_b_mean, n_sims=50000, std_dev=11.0):
    scores_a = np.maximum(np.random.normal(score_a_mean, std_dev, n_sims), 30)
    scores_b = np.maximum(np.random.normal(score_b_mean, std_dev, n_sims), 30)
    a_wins = np.sum(scores_a > scores_b) + np.sum(scores_a == scores_b) // 2
    return {
        "a_win_pct": a_wins / n_sims,
        "b_win_pct": 1 - a_wins / n_sims,
        "a_score_mean": np.mean(scores_a),
        "b_score_mean": np.mean(scores_b),
        "a_score_median": np.median(scores_a),
        "b_score_median": np.median(scores_b),
        "a_5th": np.percentile(scores_a, 5),
        "a_95th": np.percentile(scores_a, 95),
        "b_5th": np.percentile(scores_b, 5),
        "b_95th": np.percentile(scores_b, 95),
        "margin_mean": np.mean(scores_a - scores_b),
        "margin_std": np.std(scores_a - scores_b),
        "prob_close": np.mean(np.abs(scores_a - scores_b) < 3),
        "prob_blowout": np.mean(np.abs(scores_a - scores_b) > 10),
        "scores_a": scores_a,
        "scores_b": scores_b,
    }


def simulate_region(region_teams, bt_data, lr_model, lgb_model, scaler, gear=0, n_sims=10000):
    matchups = [(region_teams[i], region_teams[i + 1]) for i in range(0, len(region_teams), 2)]
    round_names = ["Round of 64", "Round of 32", "Sweet 16", "Elite Eight"]
    all_rounds = []
    current = list(matchups)
    round_num = 0
    while len(current) > 0:
        rname = round_names[round_num] if round_num < len(round_names) else f"Round {round_num}"
        round_results = []
        next_round = []
        for match in current:
            (sa, na), (sb, nb) = match[0], match[1]
            stats_a = bt_data.get(na)
            stats_b = bt_data.get(nb)
            if stats_a and stats_b:
                raw = predict_matchup(stats_a, sa, stats_b, sb, lr_model, lgb_model, scaler)
                prob = apply_gear(raw, gear)
            else:
                prob = 0.5 + (sb - sa) * 0.03
                prob = np.clip(prob, 0.05, 0.95)
            a_wins = sum(1 for _ in range(n_sims) if np.random.random() < prob)
            a_pct = a_wins / n_sims
            if a_pct >= 0.5:
                winner = (sa, na)
                round_results.append({
                    "winner": na, "w_seed": sa, "w_pct": a_pct,
                    "loser": nb, "l_seed": sb, "l_pct": 1 - a_pct,
                })
            else:
                winner = (sb, nb)
                round_results.append({
                    "winner": nb, "w_seed": sb, "w_pct": 1 - a_pct,
                    "loser": na, "l_seed": sa, "l_pct": a_pct,
                })
            next_round.append(winner)
        all_rounds.append({"name": rname, "games": round_results})
        if len(next_round) == 1:
            return all_rounds, next_round[0]
        current = [(next_round[i], next_round[i + 1]) for i in range(0, len(next_round), 2)]
        round_num += 1
    return all_rounds, None
