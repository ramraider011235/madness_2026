import requests
import json
import time
import warnings
import sys
import os
import pickle
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
# from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
# from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
warnings.filterwarnings('ignore')
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "C:/Users/gordo/Documents/madness_2026/mm_data")
YEARS = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]
CURRENT_YEAR = 2026

NAME_MAP = {
    "Alabama State": "Alabama St.", "Albany (NY)": "Albany",
    "Arizona State": "Arizona St.", "Arkansas-Pine Bluff": "Arkansas Pine Bluff",
    "Boise State": "Boise St.", "Cal State Bakersfield": "Cal St. Bakersfield",
    "Cal State Fullerton": "Cal St. Fullerton", "Cleveland State": "Cleveland St.",
    "College of Charleston": "Charleston", "Colorado State": "Colorado St.",
    "ETSU": "East Tennessee St.", "FDU": "Fairleigh Dickinson",
    "Florida State": "Florida St.", "Fresno State": "Fresno St.",
    "Gardner-Webb": "Gardner Webb", "Georgia State": "Georgia St.",
    "Grambling": "Grambling St.", "Indiana State": "Indiana St.",
    "Iowa State": "Iowa St.", "Jacksonville State": "Jacksonville St.",
    "Kansas State": "Kansas St.", "Kennesaw State": "Kennesaw St.",
    "Kent State": "Kent St.", "Long Beach State": "Long Beach St.",
    "Loyola (IL)": "Loyola Chicago", "Loyola (MD)": "Loyola MD",
    "McNeese": "McNeese St.", "Miami (FL)": "Miami FL",
    "Michigan State": "Michigan St.", "Mississippi State": "Mississippi St.",
    "Montana State": "Montana St.", "Morehead State": "Morehead St.",
    "Morgan State": "Morgan St.", "Murray State": "Murray St.",
    "NC State": "N.C. State", "New Mexico State": "New Mexico St.",
    "Norfolk State": "Norfolk St.", "North Dakota State": "North Dakota St.",
    "Northwestern State": "Northwestern St.", "Ohio State": "Ohio St.",
    "Oklahoma State": "Oklahoma St.", "Ole Miss": "Mississippi",
    "Omaha": "Nebraska Omaha", "Oregon State": "Oregon St.",
    "Penn State": "Penn St.", "Pitt": "Pittsburgh",
    "SIU-Edwardsville": "SIU Edwardsville", "Sam Houston": "Sam Houston St.",
    "San Diego State": "San Diego St.", "South Dakota State": "South Dakota St.",
    "St. John's (NY)": "St. John's", "Texas A&M-Corpus Christi": "Texas A&M Corpus Chris",
    "UConn": "Connecticut", "UMass": "Massachusetts", "UNC": "North Carolina",
    "Utah State": "Utah St.", "Washington State": "Washington St.",
    "Weber State": "Weber St.", "Wichita State": "Wichita St.",
    "Wright State": "Wright St.",
}

BRACKET_2026 = {
    "East": [
        (1, "Duke"), (16, "Siena"), (8, "Ohio St."), (9, "TCU"),
        (5, "St. John's"), (12, "Northern Iowa"), (4, "Kansas"),
        (13, "Cal Baptist"), (6, "Louisville"), (11, "South Florida"),
        (3, "Michigan St."), (14, "North Dakota St."),
        (7, "UCLA"), (10, "UCF"), (2, "Connecticut"), (15, "Furman"),
    ],
    "South": [
        (1, "Florida"), (16, "Lehigh"), (8, "Clemson"), (9, "Iowa"),
        (5, "Vanderbilt"), (12, "McNeese St."), (4, "Nebraska"),
        (13, "Troy"), (6, "North Carolina"), (11, "VCU"),
        (3, "Illinois"), (14, "Penn"), (7, "Saint Mary's"),
        (10, "Texas A&M"), (2, "Houston"), (15, "Idaho"),
    ],
    "West": [
        (1, "Arizona"), (16, "Long Island"), (8, "Villanova"),
        (9, "Utah St."), (5, "Wisconsin"), (12, "High Point"),
        (4, "Arkansas"), (13, "Hawaii"), (6, "BYU"), (11, "Texas"),
        (3, "Gonzaga"), (14, "Kennesaw St."), (7, "Miami FL"),
        (10, "Missouri"), (2, "Purdue"), (15, "Queens"),
    ],
    "Midwest": [
        (1, "Michigan"), (16, "Howard"), (8, "Georgia"),
        (9, "Saint Louis"), (5, "Texas Tech"), (12, "Akron"),
        (4, "Alabama"), (13, "Hofstra"), (6, "Tennessee"),
        (11, "SMU"), (3, "Virginia"), (14, "Wright St."),
        (7, "Kentucky"), (10, "Santa Clara"), (2, "Iowa St."),
        (15, "Tennessee St."),
    ],
}


def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def download_barttorvik(year):
    path = os.path.join(DATA_DIR, f"barttorvik_{year}.json")
    if os.path.exists(path):
        return True
    url = f"https://barttorvik.com/{year}_team_results.json"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code == 200:
            with open(path, "w") as f:
                f.write(r.text)
            return True
    except:
        pass
    return False


def load_barttorvik(year):
    path = os.path.join(DATA_DIR, f"barttorvik_{year}.json")
    if not os.path.exists(path):
        if not download_barttorvik(year):
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


def parse_sref_bracket(html_text, year):
    soup = BeautifulSoup(html_text, 'html.parser')
    games = []
    rounds_16 = {0: "R64", 1: "R32", 2: "S16", 3: "E8"}
    rounds_4 = {0: "F4", 1: "NCG"}
    def parse_team_div(div):
        is_winner = 'winner' in div.get('class', [])
        seed, name, score = None, None, None
        for s in div.find_all('span', recursive=False):
            txt = s.get_text(strip=True)
            if txt.isdigit():
                seed = int(txt)
        for link in div.find_all('a', recursive=False):
            href = link.get('href', '')
            txt = link.get_text(strip=True)
            if '/schools/' in href:
                name = txt
            elif '/boxscores/' in href and txt.isdigit():
                score = int(txt)
        return {"seed": seed, "name": name, "score": score, "winner": is_winner}
    def extract_games(bracket_div, rmap):
        result = []
        round_divs = bracket_div.find_all('div', class_='round', recursive=False)
        for ridx, rdiv in enumerate(round_divs):
            for gc in rdiv.find_all('div', recursive=False):
                team_divs = gc.find_all('div', recursive=False)
                if len(team_divs) >= 2:
                    t1, t2 = parse_team_div(team_divs[0]), parse_team_div(team_divs[1])
                    if t1["name"] and t2["name"] and (t1["score"] or t2["score"]):
                        w = t1 if t1["winner"] else t2
                        l = t2 if t1["winner"] else t1
                        rnd = rmap.get(ridx, f"R{ridx}")
                        result.append({
                            "year": year, "round": rnd,
                            "w_team": NAME_MAP.get(w["name"], w["name"]),
                            "w_seed": w["seed"], "w_score": w["score"],
                            "l_team": NAME_MAP.get(l["name"], l["name"]),
                            "l_seed": l["seed"], "l_score": l["score"],
                        })
        return result
    for bd in soup.find_all('div', class_='team16'):
        games.extend(extract_games(bd, rounds_16))
    for fd in soup.find_all('div', class_='team4'):
        games.extend(extract_games(fd, rounds_4))
    return games


def download_tournament_history():
    all_games = []
    cache_path = os.path.join(DATA_DIR, "tournament_history.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    print("Downloading tournament history from sports-reference...")
    for year in YEARS:
        url = f"https://www.sports-reference.com/cbb/postseason/men/{year}-ncaa.html"
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code == 200:
                year_games = parse_sref_bracket(r.text, year)
                all_games.extend(year_games)
                ncg = [g for g in year_games if g['round'] == 'NCG']
                champ = ncg[0]['w_team'] if ncg else "?"
                print(f"  {year}: {len(year_games)} games, champion: {champ}")
            time.sleep(1.5)
        except Exception as e:
            print(f"  {year}: FAILED ({e})")
    with open(cache_path, "w") as f:
        json.dump(all_games, f)
    return all_games


def build_training_data(games, bt_data_by_year):
    rows = []
    for g in games:
        yr = g['year']
        bt = bt_data_by_year.get(yr, {})
        w_stats = bt.get(g['w_team'])
        l_stats = bt.get(g['l_team'])
        if not w_stats or not l_stats:
            continue
        w_seed = g.get('w_seed', 8) or 8
        l_seed = g.get('l_seed', 8) or 8
        features = {
            "year": yr,
            "seed_diff": w_seed - l_seed,
            "oe_diff": w_stats["adj_oe"] - l_stats["adj_oe"],
            "de_diff": w_stats["adj_de"] - l_stats["adj_de"],
            "eff_margin_diff": w_stats["eff_margin"] - l_stats["eff_margin"],
            "barthag_diff": w_stats["barthag"] - l_stats["barthag"],
            "rank_diff": w_stats["rank"] - l_stats["rank"],
            "win_pct_diff": w_stats["win_pct"] - l_stats["win_pct"],
            "tempo_diff": w_stats["adj_tempo"] - l_stats["adj_tempo"],
            "oe_rank_diff": w_stats["adj_oe_rank"] - l_stats["adj_oe_rank"],
            "de_rank_diff": w_stats["adj_de_rank"] - l_stats["adj_de_rank"],
            "w_seed": w_seed,
            "l_seed": l_seed,
            "w_eff_margin": w_stats["eff_margin"],
            "l_eff_margin": l_stats["eff_margin"],
            "result": 1,
        }
        rows.append(features)
        flipped = {
            "year": yr,
            "seed_diff": -features["seed_diff"],
            "oe_diff": -features["oe_diff"],
            "de_diff": -features["de_diff"],
            "eff_margin_diff": -features["eff_margin_diff"],
            "barthag_diff": -features["barthag_diff"],
            "rank_diff": -features["rank_diff"],
            "win_pct_diff": -features["win_pct_diff"],
            "tempo_diff": -features["tempo_diff"],
            "oe_rank_diff": -features["oe_rank_diff"],
            "de_rank_diff": -features["de_rank_diff"],
            "w_seed": l_seed,
            "l_seed": w_seed,
            "w_eff_margin": l_stats["eff_margin"],
            "l_eff_margin": w_stats["eff_margin"],
            "result": 0,
        }
        rows.append(flipped)
    return pd.DataFrame(rows)


FEATURE_COLS = [
    "seed_diff", "oe_diff", "de_diff", "eff_margin_diff",
    "barthag_diff", "rank_diff", "win_pct_diff", "tempo_diff",
    "oe_rank_diff", "de_rank_diff",
]


def train_model(df):
    print("\n=== TRAINING MODEL ===")
    print(f"Total samples: {len(df)} ({len(df)//2} games, flipped)")
    X = df[FEATURE_COLS].values
    y = df["result"].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lr = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
    lr.fit(X_scaled, y)
    lr_pred = lr.predict_proba(X_scaled)[:, 1]
    lr_acc = accuracy_score(y, (lr_pred > 0.5).astype(int))
    lr_ll = log_loss(y, lr_pred)
    print(f"Logistic Regression  -> Accuracy: {lr_acc:.4f}, Log Loss: {lr_ll:.4f}")
    lgb_model = None
    if HAS_LGB:
        lgb_model = lgb.LGBMClassifier(
            n_estimators=150, learning_rate=0.02, max_depth=3,
            num_leaves=8, min_child_samples=50, subsample=0.7,
            colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=2.0,
            verbose=-1, random_state=42,
        )
        lgb_model.fit(X_scaled, y)
        lgb_pred = lgb_model.predict_proba(X_scaled)[:, 1]
        lgb_acc = accuracy_score(y, (lgb_pred > 0.5).astype(int))
        lgb_ll = log_loss(y, lgb_pred)
        print(f"LightGBM             -> Accuracy: {lgb_acc:.4f}, Log Loss: {lgb_ll:.4f}")
    print("\n=== ROLLING FORWARD CROSS-VALIDATION ===")
    accs = []
    for test_year in YEARS:
        if test_year < 2014:
            continue
        train_mask = df["year"] < test_year
        test_mask = df["year"] == test_year
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        X_tr = scaler.fit_transform(df.loc[train_mask, FEATURE_COLS].values)
        y_tr = df.loc[train_mask, "result"].values
        X_te = scaler.transform(df.loc[test_mask, FEATURE_COLS].values)
        y_te = df.loc[test_mask, "result"].values
        lr_cv = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
        lr_cv.fit(X_tr, y_tr)
        pred = lr_cv.predict_proba(X_te)[:, 1]
        acc = accuracy_score(y_te, (pred > 0.5).astype(int))
        original_games = df.loc[test_mask & (df["result"] == 1)]
        n_games = len(original_games)
        accs.append(acc)
        print(f"  {test_year}: Accuracy = {acc:.4f} ({n_games} games)")
    mean_acc = np.mean(accs)
    print(f"\n  Mean CV Accuracy: {mean_acc:.4f}")
    scaler_final = StandardScaler()
    X_final = scaler_final.fit_transform(df[FEATURE_COLS].values)
    y_final = df["result"].values
    lr_final = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
    lr_final.fit(X_final, y_final)
    lgb_final = None
    if HAS_LGB:
        lgb_final = lgb.LGBMClassifier(
            n_estimators=150, learning_rate=0.02, max_depth=3,
            num_leaves=8, min_child_samples=50, subsample=0.7,
            colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=2.0,
            verbose=-1, random_state=42,
        )
        lgb_final.fit(X_final, y_final)
    return lr_final, lgb_final, scaler_final


def predict_matchup(team_a_stats, team_a_seed, team_b_stats, team_b_seed, lr_model, lgb_model, scaler):
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
        prob = 0.65 * lr_prob + 0.35 * lgb_prob
    else:
        prob = lr_prob
    return prob


def apply_gear(prob, gear):
    if gear == 0:
        return prob
    logit = np.log(np.clip(prob, 1e-6, 1-1e-6) / (1 - np.clip(prob, 1e-6, 1-1e-6)))
    temperature = 1.0 + gear * 0.3
    adjusted = logit / temperature
    return 1.0 / (1.0 + np.exp(-adjusted))


def simulate_bracket(bracket, bt_data, lr_model, lgb_model, scaler, gear=0, n_sims=10000):
    print(f"\n{'='*70}")
    print(f"  MARCH MADNESS 2026 BRACKET PREDICTION")
    print(f"  Confidence Gear: {gear} ({'Chalk' if gear < 0 else 'Upset-friendly' if gear > 0 else 'Balanced'})")
    print(f"  Simulations: {n_sims:,}")
    print(f"{'='*70}")
    region_winners = {}
    for region_name, teams in bracket.items():
        print(f"\n{'='*50}")
        print(f"  {region_name.upper()} REGION")
        print(f"{'='*50}")
        matchups = []
        for i in range(0, len(teams), 2):
            matchups.append((teams[i], teams[i+1]))
        round_names = ["Round of 64", "Round of 32", "Sweet 16", "Elite Eight"]
        current = list(matchups)
        round_num = 0
        while len(current) > 0:
            rname = round_names[round_num] if round_num < len(round_names) else f"Round {round_num}"
            print(f"\n  --- {rname} ---")
            next_round = []
            for match in current:
                (seed_a, name_a), (seed_b, name_b) = match[0], match[1]
                stats_a = bt_data.get(name_a)
                stats_b = bt_data.get(name_b)
                if stats_a and stats_b:
                    raw_prob = predict_matchup(stats_a, seed_a, stats_b, seed_b, lr_model, lgb_model, scaler)
                    prob = apply_gear(raw_prob, gear)
                else:
                    prob = 0.5 + (seed_b - seed_a) * 0.03
                    prob = np.clip(prob, 0.05, 0.95)
                a_wins = 0
                for _ in range(n_sims):
                    if np.random.random() < prob:
                        a_wins += 1
                a_pct = a_wins / n_sims
                if a_pct >= 0.5:
                    winner = (seed_a, name_a)
                    print(f"  ({seed_a:>2}) {name_a:<22} {a_pct*100:5.1f}%  >  ({seed_b:>2}) {name_b:<22} {(1-a_pct)*100:5.1f}%")
                else:
                    winner = (seed_b, name_b)
                    print(f"  ({seed_b:>2}) {name_b:<22} {(1-a_pct)*100:5.1f}%  >  ({seed_a:>2}) {name_a:<22} {a_pct*100:5.1f}%")
                next_round.append(winner)
            if len(next_round) == 1:
                region_winners[region_name] = next_round[0]
                print(f"\n  >>> {region_name.upper()} CHAMPION: ({next_round[0][0]}) {next_round[0][1]} <<<")
                break
            paired = []
            for i in range(0, len(next_round), 2):
                paired.append((next_round[i], next_round[i+1]))
            current = paired
            round_num += 1
    print(f"\n{'='*70}")
    print(f"  FINAL FOUR")
    print(f"{'='*70}")
    ff_matchups = [
        (region_winners["East"], region_winners["South"]),
        (region_winners["West"], region_winners["Midwest"]),
    ]
    ff_winners = []
    print(f"\n  --- Semifinals ---")
    for (seed_a, name_a), (seed_b, name_b) in ff_matchups:
        stats_a = bt_data.get(name_a)
        stats_b = bt_data.get(name_b)
        if stats_a and stats_b:
            raw_prob = predict_matchup(stats_a, seed_a, stats_b, seed_b, lr_model, lgb_model, scaler)
            prob = apply_gear(raw_prob, gear)
        else:
            prob = 0.5
        a_wins = sum(1 for _ in range(n_sims) if np.random.random() < prob)
        a_pct = a_wins / n_sims
        if a_pct >= 0.5:
            winner = (seed_a, name_a)
            print(f"  ({seed_a:>2}) {name_a:<22} {a_pct*100:5.1f}%  >  ({seed_b:>2}) {name_b:<22} {(1-a_pct)*100:5.1f}%")
        else:
            winner = (seed_b, name_b)
            print(f"  ({seed_b:>2}) {name_b:<22} {(1-a_pct)*100:5.1f}%  >  ({seed_a:>2}) {name_a:<22} {a_pct*100:5.1f}%")
        ff_winners.append(winner)
    print(f"\n  --- National Championship ---")
    (seed_a, name_a), (seed_b, name_b) = ff_winners[0], ff_winners[1]
    stats_a = bt_data.get(name_a)
    stats_b = bt_data.get(name_b)
    if stats_a and stats_b:
        raw_prob = predict_matchup(stats_a, seed_a, stats_b, seed_b, lr_model, lgb_model, scaler)
        prob = apply_gear(raw_prob, gear)
    else:
        prob = 0.5
    a_wins = sum(1 for _ in range(n_sims) if np.random.random() < prob)
    a_pct = a_wins / n_sims
    if a_pct >= 0.5:
        champion = (seed_a, name_a)
        print(f"  ({seed_a:>2}) {name_a:<22} {a_pct*100:5.1f}%  >  ({seed_b:>2}) {name_b:<22} {(1-a_pct)*100:5.1f}%")
    else:
        champion = (seed_b, name_b)
        print(f"  ({seed_b:>2}) {name_b:<22} {(1-a_pct)*100:5.1f}%  >  ({seed_a:>2}) {name_a:<22} {a_pct*100:5.1f}%")
    print(f"\n{'='*70}")
    print(f"  2026 NATIONAL CHAMPION: ({champion[0]}) {champion[1]}")
    print(f"{'='*70}")
    return champion


def run_historical_backtest(games, bt_data_by_year, lr_model, lgb_model, scaler, test_years=None):
    if test_years is None:
        test_years = [2022, 2023, 2024, 2025]
    print(f"\n{'='*70}")
    print(f"  HISTORICAL BACKTEST")
    print(f"{'='*70}")
    for yr in test_years:
        yr_games = [g for g in games if g['year'] == yr]
        bt = bt_data_by_year.get(yr, {})
        if not bt:
            print(f"\n  {yr}: No Barttorvik data available")
            continue
        correct = 0
        total = 0
        for g in yr_games:
            w_stats = bt.get(g['w_team'])
            l_stats = bt.get(g['l_team'])
            if not w_stats or not l_stats:
                continue
            w_seed = g.get('w_seed', 8) or 8
            l_seed = g.get('l_seed', 8) or 8
            prob = predict_matchup(w_stats, w_seed, l_stats, l_seed, lr_model, lgb_model, scaler)
            if prob > 0.5:
                correct += 1
            total += 1
        if total > 0:
            print(f"  {yr}: {correct}/{total} correct ({correct/total*100:.1f}%)")


def main():
    ensure_data_dir()
    print("=" * 70)
    print("  MARCH MADNESS 2026 BRACKET PREDICTOR")
    print("  Powered by Barttorvik T-Rank + LightGBM/LogReg Ensemble")
    print("=" * 70)
    print("\n[1/4] Downloading Barttorvik data...")
    bt_data_by_year = {}
    for year in YEARS + [CURRENT_YEAR]:
        success = download_barttorvik(year)
        if success:
            bt_data_by_year[year] = load_barttorvik(year)
            print(f"  {year}: {len(bt_data_by_year[year])} teams loaded")
    print("\n[2/4] Loading tournament history...")
    games = download_tournament_history()
    print(f"  Total historical games: {len(games)}")
    print("\n[3/4] Building features and training...")
    df = build_training_data(games, bt_data_by_year)
    lr_model, lgb_model, scaler = train_model(df)
    run_historical_backtest(games, bt_data_by_year, lr_model, lgb_model, scaler)
    print("\n[4/4] Predicting 2026 bracket...")
    bt_2026 = bt_data_by_year.get(CURRENT_YEAR, {})
    if not bt_2026:
        print("ERROR: Could not load 2026 data")
        return
    gear = 0
    if len(sys.argv) > 1:
        try:
            gear = int(sys.argv[1])
        except:
            pass
    print(f"\n  Available gears:")
    print(f"    -2  = Heavy chalk (favor higher seeds strongly)")
    print(f"    -1  = Mild chalk")
    print(f"     0  = Balanced (default)")
    print(f"    +1  = Mild upset mode")
    print(f"    +2  = Chaos mode (favor upsets)")
    if len(sys.argv) <= 1:
        try:
            gear_input = input(f"\n  Select gear [-2 to +2] (default 0): ").strip()
            if gear_input:
                gear = int(gear_input)
        except:
            gear = 0
    gear = max(-2, min(2, gear))
    simulate_bracket(BRACKET_2026, bt_2026, lr_model, lgb_model, scaler, gear=gear, n_sims=10000)
    model_path = os.path.join(DATA_DIR, "trained_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"lr": lr_model, "lgb": lgb_model, "scaler": scaler}, f)
    print(f"\n  Model saved to {model_path}")


if __name__ == "__main__":
    main()
