import json
import os
import pickle
import warnings
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
warnings.filterwarnings('ignore')
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mm_data")
os.makedirs(DATA_DIR, exist_ok=True)
YEARS = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]
CURRENT_YEAR = 2026

FEATURE_COLS = [
    "seed_diff",
    "oe_diff",
    "de_diff",
    "eff_margin_diff",
    "barthag_diff",
    "rank_diff",
    "win_pct_diff",
    "tempo_diff",
    "oe_rank_diff",
    "de_rank_diff",
    "sos_diff",
    "luck_diff",
    "experience_diff",
    "away_oe_diff",
    "away_de_diff",
    "conf_rank_diff",
    "seed_eff_margin_interact",
    "tempo_mismatch",
    "consistency_diff",
]


def parse_bt_team(t):
    rec = t[3] if len(t) > 3 else "0-0"
    wins, losses = 0, 0
    if isinstance(rec, str) and '-' in rec:
        parts = rec.split('-')
        wins, losses = int(parts[0]), int(parts[1])
    total = wins + losses if (wins + losses) > 0 else 1
    adj_oe = t[4] if len(t) > 4 else 100.0
    adj_de = t[6] if len(t) > 6 else 100.0
    away_oe = t[29] if len(t) > 29 else adj_oe
    away_de = t[30] if len(t) > 30 else adj_de
    home_oe = t[27] if len(t) > 27 else adj_oe
    home_de = t[28] if len(t) > 28 else adj_de
    oe_consistency = abs(away_oe - home_oe) if away_oe and home_oe else 0
    de_consistency = abs(away_de - home_de) if away_de and home_de else 0
    return {
        "rank": t[0],
        "conf": t[2],
        "record": rec,
        "wins": wins,
        "losses": losses,
        "win_pct": wins / total,
        "adj_oe": adj_oe,
        "adj_oe_rank": t[5] if len(t) > 5 else 180,
        "adj_de": adj_de,
        "adj_de_rank": t[7] if len(t) > 7 else 180,
        "barthag": t[8] if len(t) > 8 else 0.5,
        "barthag_rank": t[9] if len(t) > 9 else 180,
        "adj_tempo": t[44] if len(t) > 44 else 67.0,
        "eff_margin": (adj_oe - adj_de),
        "sos": t[15] if len(t) > 15 else 0.5,
        "luck": t[31] if len(t) > 31 else 0.5,
        "experience": t[41] if len(t) > 41 else 1.0,
        "away_oe": away_oe,
        "away_de": away_de,
        "conf_rank": t[13] if len(t) > 13 else 5.0,
        "consistency": oe_consistency + de_consistency,
    }


def load_bt_year(year):
    import requests
    path = os.path.join(DATA_DIR, f"barttorvik_{year}.json")
    if not os.path.exists(path):
        url = f"https://barttorvik.com/{year}_team_results.json"
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code == 200:
                with open(path, "w") as f:
                    f.write(r.text)
            else:
                return {}
        except Exception:
            return {}
    with open(path) as f:
        data = json.load(f)
    return {t[1]: parse_bt_team(t) for t in data}


def build_features(a_stats, a_seed, b_stats, b_seed):
    oe_diff = a_stats["adj_oe"] - b_stats["adj_oe"]
    de_diff = a_stats["adj_de"] - b_stats["adj_de"]
    eff_margin_diff = a_stats["eff_margin"] - b_stats["eff_margin"]
    seed_diff = a_seed - b_seed
    tempo_a = a_stats["adj_tempo"]
    tempo_b = b_stats["adj_tempo"]
    return {
        "seed_diff": seed_diff,
        "oe_diff": oe_diff,
        "de_diff": de_diff,
        "eff_margin_diff": eff_margin_diff,
        "barthag_diff": a_stats["barthag"] - b_stats["barthag"],
        "rank_diff": a_stats["rank"] - b_stats["rank"],
        "win_pct_diff": a_stats["win_pct"] - b_stats["win_pct"],
        "tempo_diff": tempo_a - tempo_b,
        "oe_rank_diff": a_stats["adj_oe_rank"] - b_stats["adj_oe_rank"],
        "de_rank_diff": a_stats["adj_de_rank"] - b_stats["adj_de_rank"],
        "sos_diff": a_stats["sos"] - b_stats["sos"],
        "luck_diff": a_stats["luck"] - b_stats["luck"],
        "experience_diff": a_stats["experience"] - b_stats["experience"],
        "away_oe_diff": a_stats["away_oe"] - b_stats["away_oe"],
        "away_de_diff": a_stats["away_de"] - b_stats["away_de"],
        "conf_rank_diff": a_stats["conf_rank"] - b_stats["conf_rank"],
        "seed_eff_margin_interact": seed_diff * eff_margin_diff,
        "tempo_mismatch": abs(tempo_a - tempo_b),
        "consistency_diff": a_stats["consistency"] - b_stats["consistency"],
    }


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
        feat = build_features(w_stats, w_seed, l_stats, l_seed)
        feat["year"] = yr
        feat["result"] = 1
        rows.append(feat)
        flipped = build_features(l_stats, l_seed, w_stats, w_seed)
        flipped["year"] = yr
        flipped["result"] = 0
        rows.append(flipped)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    hist_path = os.path.join(DATA_DIR, "tournament_history.json")
    with open(hist_path) as f:
        all_games = json.load(f)
    print(f"Loaded {len(all_games)} tournament games")
    bt_data_by_year = {}
    for yr in YEARS:
        bt = load_bt_year(yr)
        bt_data_by_year[yr] = bt
        print(f"  {yr}: {len(bt)} teams")
    bt_data_by_year[CURRENT_YEAR] = load_bt_year(CURRENT_YEAR)
    print(f"  {CURRENT_YEAR}: {len(bt_data_by_year[CURRENT_YEAR])} teams")
    df = build_training_data(all_games, bt_data_by_year)
    print(f"\nTraining set: {len(df)} rows ({len(df) // 2} games x2)")
    print(f"Features: {len(FEATURE_COLS)}")
    print("\nFeature correlations with result:")
    for col in FEATURE_COLS:
        corr = df[col].corr(df["result"])
        print(f"  {col:30s}: {corr:+.4f}")
    print("\n--- Rolling Forward CV ---")
    cv_results = []
    for test_year in YEARS:
        if test_year < 2014:
            continue
        train_mask = df["year"] < test_year
        test_mask = df["year"] == test_year
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        s = StandardScaler()
        X_tr = pd.DataFrame(s.fit_transform(df.loc[train_mask, FEATURE_COLS]), columns=FEATURE_COLS)
        y_tr = df.loc[train_mask, "result"].values
        X_te = pd.DataFrame(s.transform(df.loc[test_mask, FEATURE_COLS]), columns=FEATURE_COLS)
        y_te = df.loc[test_mask, "result"].values
        lr_cv = LogisticRegression(C=0.5, max_iter=1000, solver='lbfgs')
        lr_cv.fit(X_tr, y_tr)
        lr_pred = lr_cv.predict_proba(X_te)[:, 1]
        if HAS_LGB:
            lgb_cv = lgb.LGBMClassifier(
                n_estimators=200, learning_rate=0.015, max_depth=4,
                num_leaves=12, min_child_samples=30, subsample=0.8,
                colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=1.5,
                verbose=-1, random_state=42,
            )
            lgb_cv.fit(X_tr, y_tr)
            lgb_pred = lgb_cv.predict_proba(X_te)[:, 1]
            pred = 0.55 * lr_pred + 0.45 * lgb_pred
        else:
            pred = lr_pred
        acc = accuracy_score(y_te, (pred > 0.5).astype(int))
        ll = log_loss(y_te, pred)
        n_games = len(df.loc[test_mask & (df["result"] == 1)])
        cv_results.append({"year": test_year, "accuracy": acc, "log_loss": ll, "games": n_games})
        print(f"  {test_year}: Accuracy = {acc:.4f}, Log Loss = {ll:.4f} ({n_games} games)")
    cv_df = pd.DataFrame(cv_results)
    print(f"\nMean CV Accuracy: {cv_df['accuracy'].mean():.4f}")
    print(f"Mean CV Log Loss: {cv_df['log_loss'].mean():.4f}")
    print("\n--- Training Final Model on All Data ---")
    X_all = df[FEATURE_COLS]
    y_all = df["result"].values
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_all), columns=FEATURE_COLS)
    lr_model = LogisticRegression(C=0.5, max_iter=1000, solver='lbfgs')
    lr_model.fit(X_scaled, y_all)
    lr_pred_all = lr_model.predict_proba(X_scaled)[:, 1]
    print(f"LogReg Training Acc: {accuracy_score(y_all, (lr_pred_all > 0.5).astype(int)):.4f}")
    lgb_model = None
    if HAS_LGB:
        lgb_model = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.015, max_depth=4,
            num_leaves=12, min_child_samples=30, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=1.5,
            verbose=-1, random_state=42,
        )
        lgb_model.fit(X_scaled, y_all)
        lgb_pred_all = lgb_model.predict_proba(X_scaled)[:, 1]
        print(f"LightGBM Training Acc: {accuracy_score(y_all, (lgb_pred_all > 0.5).astype(int)):.4f}")
    model_path = os.path.join(DATA_DIR, "trained_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"lr": lr_model, "lgb": lgb_model, "scaler": scaler}, f)
    print(f"\nModel saved to {model_path}")
    print(f"Features: {len(FEATURE_COLS)}")
    print("Ensemble: 55% LogReg + 45% LightGBM")
    if HAS_LGB:
        importances = pd.Series(lgb_model.feature_importances_, index=FEATURE_COLS)
        print("\nLightGBM Feature Importance:")
        for feat, imp in importances.sort_values(ascending=False).items():
            print(f"  {feat:30s}: {imp}")
