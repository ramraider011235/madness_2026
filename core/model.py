import numpy as np
import pandas as pd

FEATURE_COLS = [
    "seed_diff", "oe_diff", "de_diff", "eff_margin_diff",
    "barthag_diff", "rank_diff", "win_pct_diff", "tempo_diff",
    "oe_rank_diff", "de_rank_diff", "sos_diff", "luck_diff",
    "experience_diff", "away_oe_diff", "away_de_diff",
    "conf_rank_diff", "seed_eff_margin_interact",
    "tempo_mismatch", "consistency_diff",
]
AVG_TEMPO = 67.5
AVG_EFF = 107.0


def predict_matchup(a_stats, a_seed, b_stats, b_seed, lr_model, lgb_model, scaler):
    seed_diff = a_seed - b_seed
    eff_margin_diff = a_stats["eff_margin"] - b_stats["eff_margin"]
    features = pd.DataFrame([[
        seed_diff,
        a_stats["adj_oe"] - b_stats["adj_oe"],
        a_stats["adj_de"] - b_stats["adj_de"],
        eff_margin_diff,
        a_stats["barthag"] - b_stats["barthag"],
        a_stats["rank"] - b_stats["rank"],
        a_stats["win_pct"] - b_stats["win_pct"],
        a_stats["adj_tempo"] - b_stats["adj_tempo"],
        a_stats["adj_oe_rank"] - b_stats["adj_oe_rank"],
        a_stats["adj_de_rank"] - b_stats["adj_de_rank"],
        a_stats.get("sos", 0.5) - b_stats.get("sos", 0.5),
        a_stats.get("luck", 0.5) - b_stats.get("luck", 0.5),
        a_stats.get("experience", 1.0) - b_stats.get("experience", 1.0),
        a_stats.get("away_oe", a_stats["adj_oe"]) - b_stats.get("away_oe", b_stats["adj_oe"]),
        a_stats.get("away_de", a_stats["adj_de"]) - b_stats.get("away_de", b_stats["adj_de"]),
        a_stats.get("conf_rank", 5.0) - b_stats.get("conf_rank", 5.0),
        seed_diff * eff_margin_diff,
        abs(a_stats["adj_tempo"] - b_stats["adj_tempo"]),
        a_stats.get("consistency", 0) - b_stats.get("consistency", 0),
    ]], columns=FEATURE_COLS)
    X = pd.DataFrame(scaler.transform(features), columns=FEATURE_COLS)
    lr_prob = lr_model.predict_proba(X)[0][1]
    if lgb_model:
        lgb_prob = lgb_model.predict_proba(X)[0][1]
        return 0.55 * lr_prob + 0.45 * lgb_prob
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


def _get_matchup_prob(sa, na, sb, nb, bt_data, lr_model, lgb_model, scaler, gear, cache=None):
    key = (na, nb, gear)
    if cache is not None and key in cache:
        return cache[key]
    stats_a = bt_data.get(na)
    stats_b = bt_data.get(nb)
    if stats_a and stats_b:
        raw = predict_matchup(stats_a, sa, stats_b, sb, lr_model, lgb_model, scaler)
        prob = apply_gear(raw, gear)
    else:
        prob = 0.5 + (sb - sa) * 0.03
        prob = np.clip(prob, 0.05, 0.95)
    if cache is not None:
        cache[key] = prob
    return prob


def simulate_region(region_teams, bt_data, lr_model, lgb_model, scaler, gear=0, n_sims=10000, single_draw=False):
    r1_matchups = [(region_teams[i], region_teams[i + 1]) for i in range(0, len(region_teams), 2)]
    round_names = ["Round of 64", "Round of 32", "Sweet 16", "Elite Eight"]
    prob_cache = {}
    r1_slots = []
    for (sa, na), (sb, nb) in r1_matchups:
        prob = _get_matchup_prob(sa, na, sb, nb, bt_data, lr_model, lgb_model, scaler, gear, prob_cache)
        r1_slots.append(((sa, na), (sb, nb), prob))
    if single_draw:
        all_rounds = []
        prev_winners = []
        round_results = []
        for (sa, na), (sb, nb), prob in r1_slots:
            if np.random.random() < prob:
                winner, loser = (sa, na), (sb, nb)
                w_pct = prob
            else:
                winner, loser = (sb, nb), (sa, na)
                w_pct = 1 - prob
            prev_winners.append(winner)
            round_results.append({
                "winner": winner[1], "w_seed": winner[0], "w_pct": w_pct,
                "loser": loser[1], "l_seed": loser[0], "l_pct": 1 - w_pct,
            })
        all_rounds.append({"name": round_names[0], "games": round_results})
        rnd = 1
        while len(prev_winners) > 1:
            cur_winners = []
            round_results = []
            for gi in range(0, len(prev_winners), 2):
                sa, na = prev_winners[gi]
                sb, nb = prev_winners[gi + 1]
                prob = _get_matchup_prob(sa, na, sb, nb, bt_data, lr_model, lgb_model, scaler, gear, prob_cache)
                if np.random.random() < prob:
                    winner, loser = (sa, na), (sb, nb)
                    w_pct = prob
                else:
                    winner, loser = (sb, nb), (sa, na)
                    w_pct = 1 - prob
                cur_winners.append(winner)
                round_results.append({
                    "winner": winner[1], "w_seed": winner[0], "w_pct": w_pct,
                    "loser": loser[1], "l_seed": loser[0], "l_pct": 1 - w_pct,
                })
            rname = round_names[rnd] if rnd < len(round_names) else f"Round {rnd}"
            all_rounds.append({"name": rname, "games": round_results})
            prev_winners = cur_winners
            rnd += 1
        return all_rounds, prev_winners[0]
    n_r1 = len(r1_slots)
    n_rounds = 0
    tmp = n_r1
    while tmp >= 1:
        n_rounds += 1
        tmp //= 2
    games_per_round = [n_r1 // (2 ** r) for r in range(n_rounds)]
    game_wins = [[{} for _ in range(gpr)] for gpr in games_per_round]
    region_champ_counts = {}
    for _ in range(n_sims):
        prev_winners = []
        for gi, ((sa, na), (sb, nb), prob) in enumerate(r1_slots):
            if np.random.random() < prob:
                prev_winners.append((sa, na))
            else:
                prev_winners.append((sb, nb))
            game_wins[0][gi][prev_winners[gi]] = game_wins[0][gi].get(prev_winners[gi], 0) + 1
        for r in range(1, n_rounds):
            cur_winners = []
            for gi in range(0, len(prev_winners), 2):
                sa, na = prev_winners[gi]
                sb, nb = prev_winners[gi + 1]
                prob = _get_matchup_prob(sa, na, sb, nb, bt_data, lr_model, lgb_model, scaler, gear, prob_cache)
                if np.random.random() < prob:
                    cur_winners.append((sa, na))
                else:
                    cur_winners.append((sb, nb))
                g_idx = gi // 2
                game_wins[r][g_idx][cur_winners[-1]] = game_wins[r][g_idx].get(cur_winners[-1], 0) + 1
            prev_winners = cur_winners
        champ = prev_winners[0]
        region_champ_counts[champ] = region_champ_counts.get(champ, 0) + 1
    all_rounds = []
    for r in range(n_rounds):
        rname = round_names[r] if r < len(round_names) else f"Round {r}"
        round_results = []
        for gi, counts in enumerate(game_wins[r]):
            teams_sorted = sorted(counts.items(), key=lambda x: x[1], reverse=True)
            (ws, wn), w_count = teams_sorted[0]
            if len(teams_sorted) > 1:
                (ls, ln), l_count = teams_sorted[1]
            else:
                ls, ln, l_count = 0, "N/A", 0
            total = w_count + l_count
            round_results.append({
                "winner": wn, "w_seed": ws, "w_pct": w_count / total,
                "loser": ln, "l_seed": ls, "l_pct": l_count / total,
            })
        all_rounds.append({"name": rname, "games": round_results})
    best_champ = max(region_champ_counts, key=region_champ_counts.get)
    return all_rounds, best_champ
