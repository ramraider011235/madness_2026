
import argparse
import datetime as dt
import gzip
import io
import json
import math
import os
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error
from sklearn.preprocessing import StandardScaler

USER_AGENT = "Mozilla/5.0"
BRACKET_2026 = {
    "East": [(1, "Duke"), (16, "Siena"), (8, "Ohio St."), (9, "TCU"), (5, "St. John's"), (12, "Northern Iowa"), (4, "Kansas"), (13, "Cal Baptist"), (6, "Louisville"), (11, "South Florida"), (3, "Michigan St."), (14, "North Dakota St."), (7, "UCLA"), (10, "UCF"), (2, "Connecticut"), (15, "Furman")],
    "South": [(1, "Florida"), (16, "Lehigh"), (8, "Clemson"), (9, "Iowa"), (5, "Vanderbilt"), (12, "McNeese St."), (4, "Nebraska"), (13, "Troy"), (6, "North Carolina"), (11, "VCU"), (3, "Illinois"), (14, "Penn"), (7, "Saint Mary's"), (10, "Texas A&M"), (2, "Houston"), (15, "Idaho")],
    "West": [(1, "Arizona"), (16, "Long Island"), (8, "Villanova"), (9, "Utah St."), (5, "Wisconsin"), (12, "High Point"), (4, "Arkansas"), (13, "Hawaii"), (6, "BYU"), (11, "Texas"), (3, "Gonzaga"), (14, "Kennesaw St."), (7, "Miami FL"), (10, "Missouri"), (2, "Purdue"), (15, "Queens")],
    "Midwest": [(1, "Michigan"), (16, "Howard"), (8, "Georgia"), (9, "Saint Louis"), (5, "Texas Tech"), (12, "Akron"), (4, "Alabama"), (13, "Hofstra"), (6, "Tennessee"), (11, "SMU"), (3, "Virginia"), (14, "Wright St."), (7, "Kentucky"), (10, "Santa Clara"), (2, "Iowa St."), (15, "Tennessee St.")]
}
TEAM_ALIASES = {
    "uconn": "Connecticut",
    "u conn": "Connecticut",
    "unc": "North Carolina",
    "ole miss": "Mississippi",
    "miami (fl)": "Miami FL",
    "st john's": "St. John's",
    "st johns": "St. John's",
    "saint marys": "Saint Mary's",
    "saint mary's": "Saint Mary's",
    "saint louis": "Saint Louis",
    "ohio state": "Ohio St.",
    "michigan state": "Michigan St.",
    "iowa state": "Iowa St.",
    "utah state": "Utah St.",
    "north dakota state": "North Dakota St.",
    "wright state": "Wright St.",
    "mcneese": "McNeese St.",
    "connecticut": "Connecticut",
    "miami ohio": "Miami (OH)",
    "miami (oh)": "Miami (OH)",
    "brigham young": "BYU",
    "southern methodist": "SMU"
}
PLAYER_ADV_COLUMNS = ["player", "pos", "exp", "team", "conf", "g", "min", "porpag", "dporpag", "ortg", "adj_oe", "drtg", "adj_de", "stops", "obpm", "dbpm", "bpm", "oreb", "dreb", "ast", "to", "blk", "stl", "ftr", "pfr", "rec", "pick", "year", "id"]
BASE_FEATURES = [
    "seed_diff", "seed_sum", "rank_diff", "rank_sum", "barthag_diff", "barthag_sum", "adj_oe_diff", "adj_de_diff",
    "eff_margin_diff", "adj_tempo_diff", "momentum_7_diff", "momentum_14_diff", "momentum_30_diff",
    "program_1y_diff", "program_3y_diff", "win_pct_diff", "roster_top1_porpag_diff", "roster_top3_porpag_diff",
    "roster_top7_min_share_diff", "roster_exp_diff", "conf_strength_diff", "abs_seed_diff", "abs_rank_diff",
    "abs_barthag_diff", "adj_oe_sum", "adj_de_sum", "eff_margin_sum", "adj_tempo_sum"
]
SCORE_FEATURES = BASE_FEATURES + ["barthag_product", "tempo_product"]
ROUND_POINTS = {"R64": 1, "R32": 2, "S16": 4, "E8": 8, "F4": 16, "NCG": 32}

def clean_team_name(name):
    text = str(name).strip()
    text = text.replace("&", "and")
    text = text.replace("'", "")
    text = re.sub(r"\(.*?\)", "", text)
    text = text.replace(".", " ")
    text = text.replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip().lower()
    if text in TEAM_ALIASES:
        return TEAM_ALIASES[text]
    return " ".join(word.capitalize() if word not in {"st", "and"} else ("St." if word == "st" else "and") for word in text.split())

def selection_sunday(year):
    d = dt.date(year, 3, 1)
    sundays = []
    while d.month == 3:
        if d.weekday() == 6:
            sundays.append(d)
        d += dt.timedelta(days=1)
    if year == 2021:
        return dt.date(2021, 3, 14)
    return sundays[1]

def _fetch_bytes(url, cache_path=None, timeout=45):
    if cache_path and Path(cache_path).exists():
        return Path(cache_path).read_bytes()
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    response.raise_for_status()
    content = response.content
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_path).write_bytes(content)
    return content

def _fetch_text(url, cache_path=None, timeout=45):
    return _fetch_bytes(url, cache_path=cache_path, timeout=timeout).decode("utf-8", errors="ignore")

def _parse_bart_json_rows(rows):
    records = []
    for row in rows:
        record = {
            "team": row[1],
            "conf": row[2] if len(row) > 2 else None,
            "record": row[3] if len(row) > 3 else None,
            "adj_oe": float(row[4]) if len(row) > 4 and row[4] not in ("", None) else np.nan,
            "adj_oe_rank": float(row[5]) if len(row) > 5 and row[5] not in ("", None) else np.nan,
            "adj_de": float(row[6]) if len(row) > 6 and row[6] not in ("", None) else np.nan,
            "adj_de_rank": float(row[7]) if len(row) > 7 and row[7] not in ("", None) else np.nan,
            "barthag": float(row[8]) if len(row) > 8 and row[8] not in ("", None) else np.nan,
            "rank": float(row[9]) if len(row) > 9 and row[9] not in ("", None) else np.nan,
            "adj_tempo": float(row[44]) if len(row) > 44 and row[44] not in ("", None) else np.nan
        }
        wins = np.nan
        losses = np.nan
        if isinstance(record["record"], str) and "-" in record["record"]:
            parts = record["record"].split("-")
            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                wins = int(parts[0])
                losses = int(parts[1])
        record["wins"] = wins
        record["losses"] = losses
        total = wins + losses if pd.notna(wins) and pd.notna(losses) else np.nan
        record["win_pct"] = wins / total if pd.notna(total) and total > 0 else np.nan
        record["eff_margin"] = record["adj_oe"] - record["adj_de"] if pd.notna(record["adj_oe"]) and pd.notna(record["adj_de"]) else np.nan
        records.append(record)
    frame = pd.DataFrame(records)
    if len(frame) == 0:
        return frame
    frame["team_norm"] = frame["team"].map(clean_team_name)
    frame = frame.sort_values(["rank", "team"]).drop_duplicates("team_norm", keep="first").reset_index(drop=True)
    return frame

def _decode_bart_snapshot_bytes(content):
    content = bytes(content)
    if content[:2] == b"\x1f\x8b":
        decoded = gzip.decompress(content).decode("utf-8")
    else:
        decoded = content.decode("utf-8", errors="ignore")
    rows = json.loads(decoded)
    if not isinstance(rows, list):
        raise ValueError("Bart snapshot payload was not a JSON list")
    return rows

def load_bart_team_snapshot(year, snapshot_date=None, cache_dir="cache"):
    cache_dir = Path(cache_dir)
    if snapshot_date is None:
        url = f"https://barttorvik.com/{year}_team_results.json"
        cache_path = cache_dir / "bart" / f"{year}_team_results.json"
        text = _fetch_text(url, cache_path=cache_path)
        rows = json.loads(text)
        return _parse_bart_json_rows(rows)
    if isinstance(snapshot_date, str):
        snap = dt.datetime.strptime(snapshot_date, "%Y-%m-%d").date()
    else:
        snap = snapshot_date
    stamp = snap.strftime("%Y%m%d")
    url = f"https://barttorvik.com/timemachine/team_results/{stamp}_team_results.json.gz"
    cache_path = cache_dir / "bart" / "timemachine" / f"{stamp}_team_results.json.gz"
    content = _fetch_bytes(url, cache_path=cache_path)
    try:
        rows = _decode_bart_snapshot_bytes(content)
    except Exception:
        fallback_url = f"https://barttorvik.com/timemachine/team_results/{stamp}_team_results.json"
        fallback_cache_path = cache_dir / "bart" / "timemachine" / f"{stamp}_team_results.json"
        fallback_content = _fetch_bytes(fallback_url, cache_path=fallback_cache_path)
        rows = _decode_bart_snapshot_bytes(fallback_content)
    return _parse_bart_json_rows(rows)

def load_current_player_advanced(year, cache_dir="cache"):
    cache_path = Path(cache_dir) / "bart" / f"{year}_player_advanced.csv"
    url = f"https://barttorvik.com/getadvstats.php?year={year}&csv=1"
    text = _fetch_text(url, cache_path=cache_path)
    frame = pd.read_csv(io.StringIO(text), header=None)
    if frame.shape[1] >= len(PLAYER_ADV_COLUMNS):
        frame = frame.iloc[:, :len(PLAYER_ADV_COLUMNS)]
        frame.columns = PLAYER_ADV_COLUMNS
    else:
        cols = PLAYER_ADV_COLUMNS[:frame.shape[1]]
        frame.columns = cols
    if "team" in frame.columns:
        frame["team_norm"] = frame["team"].map(clean_team_name)
    for column in ["g", "min", "porpag", "dporpag", "ortg", "adj_oe", "drtg", "adj_de", "stops", "obpm", "dbpm", "bpm", "oreb", "dreb", "ast", "to", "blk", "stl", "ftr", "pfr", "rec", "pick", "year", "id"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame

def build_roster_features(player_frame):
    if player_frame is None or len(player_frame) == 0 or "team_norm" not in player_frame.columns:
        return pd.DataFrame(columns=["team_norm", "roster_top1_porpag", "roster_top3_porpag", "roster_top7_min_share", "roster_exp"])
    exp_map = {"Fr": 1, "So": 2, "Jr": 3, "Sr": 4, "Gr": 5}
    work = player_frame.copy()
    work["min"] = pd.to_numeric(work.get("min"), errors="coerce").fillna(0)
    work["porpag"] = pd.to_numeric(work.get("porpag"), errors="coerce").fillna(0)
    work["exp_num"] = work.get("exp", pd.Series("", index=work.index)).map(exp_map).fillna(0)
    pieces = []
    for team, group in work.groupby("team_norm"):
        group = group.sort_values(["min", "porpag"], ascending=False)
        total_min = group["min"].sum()
        pieces.append({
            "team_norm": team,
            "roster_top1_porpag": group["porpag"].head(1).sum(),
            "roster_top3_porpag": group["porpag"].head(3).sum(),
            "roster_top7_min_share": group["min"].head(7).sum() / total_min if total_min > 0 else np.nan,
            "roster_exp": np.average(group["exp_num"].head(7), weights=np.maximum(group["min"].head(7), 1)) if len(group) else np.nan
        })
    return pd.DataFrame(pieces)

def build_conf_strength(snapshot):
    if snapshot is None or len(snapshot) == 0:
        return pd.DataFrame(columns=["conf", "conf_strength"])
    conf = snapshot.groupby("conf", dropna=False)["barthag"].mean().reset_index().rename(columns={"barthag": "conf_strength"})
    return conf

def build_program_history(team_names, current_year, cache_dir="cache", seasons_back=3):
    history_rows = []
    for offset in range(1, seasons_back + 1):
        year = current_year - offset
        try:
            snapshot = load_bart_team_snapshot(year, snapshot_date=None, cache_dir=cache_dir)
        except Exception:
            continue
        keep = snapshot[snapshot["team_norm"].isin(team_names)][["team_norm", "barthag", "rank"]].copy()
        keep["season_offset"] = offset
        history_rows.append(keep)
    if not history_rows:
        return pd.DataFrame(columns=["team_norm", "program_1y", "program_3y"])
    hist = pd.concat(history_rows, ignore_index=True)
    out = []
    for team, group in hist.groupby("team_norm"):
        one = group.loc[group["season_offset"] == 1, "barthag"]
        weights = 1 / group["season_offset"]
        out.append({
            "team_norm": team,
            "program_1y": float(one.iloc[0]) if len(one) else np.nan,
            "program_3y": np.average(group["barthag"], weights=weights) if len(group) else np.nan
        })
    return pd.DataFrame(out)

def build_team_feature_table(year=2026, bracket=None, cache_dir="cache", current_date=None):
    bracket = bracket or BRACKET_2026
    current_date = current_date or selection_sunday(year)
    team_list = []
    for region, teams in bracket.items():
        for seed, team in teams:
            team_list.append({"team": team, "team_norm": clean_team_name(team), "seed": seed, "region": region})
    teams = pd.DataFrame(team_list)
    base = load_bart_team_snapshot(year, snapshot_date=current_date, cache_dir=cache_dir).drop(columns=["team"], errors="ignore")
    try:
        snapshot_7 = load_bart_team_snapshot(year, snapshot_date=current_date - dt.timedelta(days=7), cache_dir=cache_dir)[["team_norm", "barthag"]].rename(columns={"barthag": "barthag_7"})
    except Exception:
        snapshot_7 = pd.DataFrame(columns=["team_norm", "barthag_7"])
    try:
        snapshot_14 = load_bart_team_snapshot(year, snapshot_date=current_date - dt.timedelta(days=14), cache_dir=cache_dir)[["team_norm", "barthag"]].rename(columns={"barthag": "barthag_14"})
    except Exception:
        snapshot_14 = pd.DataFrame(columns=["team_norm", "barthag_14"])
    try:
        snapshot_30 = load_bart_team_snapshot(year, snapshot_date=current_date - dt.timedelta(days=30), cache_dir=cache_dir)[["team_norm", "barthag"]].rename(columns={"barthag": "barthag_30"})
    except Exception:
        snapshot_30 = pd.DataFrame(columns=["team_norm", "barthag_30"])
    conf_strength = build_conf_strength(base)
    try:
        roster = build_roster_features(load_current_player_advanced(year, cache_dir=cache_dir))
    except Exception:
        roster = pd.DataFrame(columns=["team_norm", "roster_top1_porpag", "roster_top3_porpag", "roster_top7_min_share", "roster_exp"])
    history = build_program_history(set(teams["team_norm"]), year, cache_dir=cache_dir)
    feature_table = teams.merge(base, on="team_norm", how="left").merge(snapshot_7, on="team_norm", how="left").merge(snapshot_14, on="team_norm", how="left").merge(snapshot_30, on="team_norm", how="left").merge(conf_strength, on="conf", how="left").merge(roster, on="team_norm", how="left").merge(history, on="team_norm", how="left")
    if "team" not in feature_table.columns:
        if "team_x" in feature_table.columns:
            feature_table["team"] = feature_table["team_x"]
        elif "team_y" in feature_table.columns:
            feature_table["team"] = feature_table["team_y"]
    feature_table["momentum_7"] = feature_table["barthag"] - feature_table["barthag_7"]
    feature_table["momentum_14"] = feature_table["barthag"] - feature_table["barthag_14"]
    feature_table["momentum_30"] = feature_table["barthag"] - feature_table["barthag_30"]
    feature_table = feature_table.drop(columns=[c for c in ["barthag_7", "barthag_14", "barthag_30", "team_x", "team_y"] if c in feature_table.columns])
    cols = ["team", "team_norm", "seed", "region"] + [c for c in feature_table.columns if c not in {"team", "team_norm", "seed", "region"}]
    return feature_table[cols]

def load_kaggle_inputs(kaggle_dir):
    if kaggle_dir is None:
        return {}
    kaggle_dir = Path(kaggle_dir)
    required = {
        "teams": kaggle_dir / "MTeams.csv",
        "seeds": kaggle_dir / "MNCAATourneySeeds.csv",
        "compact": kaggle_dir / "MNCAATourneyCompactResults.csv"
    }
    if not all(path.exists() for path in required.values()):
        return {}
    data = {name: pd.read_csv(path) for name, path in required.items()}
    detailed_path = kaggle_dir / "MNCAATourneyDetailedResults.csv"
    if detailed_path.exists():
        data["detailed"] = pd.read_csv(detailed_path)
    regular_path = kaggle_dir / "MRegularSeasonDetailedResults.csv"
    if regular_path.exists():
        data["regular"] = pd.read_csv(regular_path)
    teams = data["teams"].copy()
    teams["team_norm"] = teams["TeamName"].map(clean_team_name)
    data["teams"] = teams
    return data

def parse_seed(seed):
    digits = re.findall(r"\d+", str(seed))
    return int(digits[0]) if digits else np.nan

def build_seed_lookup(data):
    if not data:
        return pd.DataFrame(columns=["Season", "TeamID", "SeedNum"])
    seeds = data["seeds"].copy()
    seeds["SeedNum"] = seeds["Seed"].map(parse_seed)
    return seeds[["Season", "TeamID", "SeedNum"]]

def _round_from_daynum(daynum):
    if daynum <= 136:
        return "R64"
    if daynum <= 138:
        return "R32"
    if daynum <= 144:
        return "S16"
    if daynum <= 146:
        return "E8"
    if daynum <= 152:
        return "F4"
    return "NCG"

def build_historical_team_features(seasons, kaggle_data=None, cache_dir="cache"):
    frames = []
    if not seasons:
        return {}
    team_name_map = None
    if kaggle_data:
        team_name_map = kaggle_data["teams"][["TeamID", "TeamName", "team_norm"]].drop_duplicates()
    for season in seasons:
        try:
            snapshot = load_bart_team_snapshot(season, snapshot_date=selection_sunday(season), cache_dir=cache_dir)
        except Exception:
            try:
                snapshot = load_bart_team_snapshot(season, snapshot_date=None, cache_dir=cache_dir)
            except Exception:
                continue
        try:
            snap_7 = load_bart_team_snapshot(season, snapshot_date=selection_sunday(season) - dt.timedelta(days=7), cache_dir=cache_dir)[["team_norm", "barthag"]].rename(columns={"barthag": "barthag_7"})
        except Exception:
            snap_7 = pd.DataFrame(columns=["team_norm", "barthag_7"])
        try:
            snap_14 = load_bart_team_snapshot(season, snapshot_date=selection_sunday(season) - dt.timedelta(days=14), cache_dir=cache_dir)[["team_norm", "barthag"]].rename(columns={"barthag": "barthag_14"})
        except Exception:
            snap_14 = pd.DataFrame(columns=["team_norm", "barthag_14"])
        try:
            snap_30 = load_bart_team_snapshot(season, snapshot_date=selection_sunday(season) - dt.timedelta(days=30), cache_dir=cache_dir)[["team_norm", "barthag"]].rename(columns={"barthag": "barthag_30"})
        except Exception:
            snap_30 = pd.DataFrame(columns=["team_norm", "barthag_30"])
        conf_strength = build_conf_strength(snapshot)
        history = build_program_history(set(snapshot["team_norm"]), season, cache_dir=cache_dir)
        frame = snapshot.merge(snap_7, on="team_norm", how="left").merge(snap_14, on="team_norm", how="left").merge(snap_30, on="team_norm", how="left").merge(conf_strength, on="conf", how="left").merge(history, on="team_norm", how="left")
        frame["momentum_7"] = frame["barthag"] - frame["barthag_7"]
        frame["momentum_14"] = frame["barthag"] - frame["barthag_14"]
        frame["momentum_30"] = frame["barthag"] - frame["barthag_30"]
        if kaggle_data:
            seeds = build_seed_lookup(kaggle_data)
            frame = team_name_map.merge(frame, on="team_norm", how="left").merge(seeds[seeds["Season"] == season][["TeamID", "SeedNum"]], on="TeamID", how="left")
            frame["seed"] = frame["SeedNum"]
            frame["team"] = frame["TeamName"]
        frame["season"] = season
        frames.append(frame)
    if not frames:
        return {}
    combined = pd.concat(frames, ignore_index=True)
    output = {}
    for season, group in combined.groupby("season"):
        output[season] = group.copy()
    return output

def build_training_dataset(kaggle_dir=None, cache_dir="cache", seasons=range(2010, 2026)):
    kaggle_data = load_kaggle_inputs(kaggle_dir)
    if not kaggle_data:
        return pd.DataFrame()
    features_by_season = build_historical_team_features(list(seasons), kaggle_data=kaggle_data, cache_dir=cache_dir)
    if not features_by_season:
        return pd.DataFrame()
    compact = kaggle_data["compact"].copy()
    compact = compact[compact["Season"].isin(seasons)].copy()
    compact["round"] = compact["DayNum"].map(_round_from_daynum)
    rows = []
    for game in compact.itertuples():
        season_frame = features_by_season.get(game.Season)
        if season_frame is None or len(season_frame) == 0:
            continue
        team_index = season_frame.set_index("TeamID")
        if game.WTeamID not in team_index.index or game.LTeamID not in team_index.index:
            continue
        w = team_index.loc[game.WTeamID]
        l = team_index.loc[game.LTeamID]
        rows.append(make_training_row(game.Season, game.round, w, l, 1, game.WScore, game.LScore))
        rows.append(make_training_row(game.Season, game.round, l, w, 0, game.LScore, game.WScore))
    train = pd.DataFrame(rows)
    train = train.replace([np.inf, -np.inf], np.nan)
    return train

def make_training_row(season, round_name, team1, team2, label, team1_score=None, team2_score=None):
    row = {
        "season": season,
        "round": round_name,
        "team1": team1["team"],
        "team2": team2["team"],
        "team1_win": label,
        "team1_score": team1_score,
        "team2_score": team2_score
    }
    row.update(make_feature_vector(team1, team2))
    if team1_score is not None and team2_score is not None:
        row["margin"] = team1_score - team2_score
        row["total"] = team1_score + team2_score
    return row

def make_feature_vector(team1, team2, recent_form_weight=1.0, program_weight=1.0):
    def val(obj, key):
        value = obj.get(key, np.nan)
        return float(value) if pd.notna(value) else np.nan
    features = {
        "seed_diff": val(team1, "seed") - val(team2, "seed"),
        "seed_sum": val(team1, "seed") + val(team2, "seed"),
        "rank_diff": val(team1, "rank") - val(team2, "rank"),
        "rank_sum": val(team1, "rank") + val(team2, "rank"),
        "barthag_diff": val(team1, "barthag") - val(team2, "barthag"),
        "barthag_sum": val(team1, "barthag") + val(team2, "barthag"),
        "adj_oe_diff": val(team1, "adj_oe") - val(team2, "adj_oe"),
        "adj_de_diff": val(team1, "adj_de") - val(team2, "adj_de"),
        "eff_margin_diff": val(team1, "eff_margin") - val(team2, "eff_margin"),
        "adj_tempo_diff": val(team1, "adj_tempo") - val(team2, "adj_tempo"),
        "momentum_7_diff": recent_form_weight * (val(team1, "momentum_7") - val(team2, "momentum_7")),
        "momentum_14_diff": recent_form_weight * (val(team1, "momentum_14") - val(team2, "momentum_14")),
        "momentum_30_diff": recent_form_weight * (val(team1, "momentum_30") - val(team2, "momentum_30")),
        "program_1y_diff": program_weight * (val(team1, "program_1y") - val(team2, "program_1y")),
        "program_3y_diff": program_weight * (val(team1, "program_3y") - val(team2, "program_3y")),
        "win_pct_diff": val(team1, "win_pct") - val(team2, "win_pct"),
        "roster_top1_porpag_diff": val(team1, "roster_top1_porpag") - val(team2, "roster_top1_porpag"),
        "roster_top3_porpag_diff": val(team1, "roster_top3_porpag") - val(team2, "roster_top3_porpag"),
        "roster_top7_min_share_diff": val(team1, "roster_top7_min_share") - val(team2, "roster_top7_min_share"),
        "roster_exp_diff": val(team1, "roster_exp") - val(team2, "roster_exp"),
        "conf_strength_diff": val(team1, "conf_strength") - val(team2, "conf_strength"),
        "abs_seed_diff": abs(val(team1, "seed") - val(team2, "seed")),
        "abs_rank_diff": abs(val(team1, "rank") - val(team2, "rank")),
        "abs_barthag_diff": abs(val(team1, "barthag") - val(team2, "barthag")),
        "adj_oe_sum": val(team1, "adj_oe") + val(team2, "adj_oe"),
        "adj_de_sum": val(team1, "adj_de") + val(team2, "adj_de"),
        "eff_margin_sum": val(team1, "eff_margin") + val(team2, "eff_margin"),
        "adj_tempo_sum": val(team1, "adj_tempo") + val(team2, "adj_tempo"),
        "barthag_product": val(team1, "barthag") * val(team2, "barthag"),
        "tempo_product": val(team1, "adj_tempo") * val(team2, "adj_tempo")
    }
    return features

def _season_splits(train):
    seasons = sorted(train["season"].dropna().unique())
    splits = []
    for holdout in seasons[3:]:
        tr = train[train["season"] < holdout]
        va = train[train["season"] == holdout]
        if len(tr) and len(va):
            splits.append((tr.index.to_numpy(), va.index.to_numpy()))
    if not splits and len(seasons) >= 2:
        holdout = seasons[-1]
        splits.append((train[train["season"] < holdout].index.to_numpy(), train[train["season"] == holdout].index.to_numpy()))
    return splits

def train_matchup_model(train):
    if len(train) == 0:
        return {"mode": "heuristic"}
    X = train[BASE_FEATURES].copy()
    y = train["team1_win"].astype(int).to_numpy()
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)
    logit = LogisticRegression(max_iter=5000, C=0.5, solver="lbfgs")
    tree = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.04, max_iter=350, l2_regularization=0.2, min_samples_leaf=20)
    splits = _season_splits(train)
    oof_logit = np.full(len(train), np.nan)
    oof_tree = np.full(len(train), np.nan)
    for train_idx, valid_idx in splits:
        imp = SimpleImputer(strategy="median")
        Xtr = imp.fit_transform(X.iloc[train_idx])
        Xva = imp.transform(X.iloc[valid_idx])
        scl = StandardScaler()
        Xtrs = scl.fit_transform(Xtr)
        Xvas = scl.transform(Xva)
        m1 = LogisticRegression(max_iter=5000, C=0.5, solver="lbfgs")
        m2 = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.04, max_iter=350, l2_regularization=0.2, min_samples_leaf=20)
        m1.fit(Xtrs, y[train_idx])
        m2.fit(Xtr, y[train_idx])
        oof_logit[valid_idx] = m1.predict_proba(Xvas)[:, 1]
        oof_tree[valid_idx] = m2.predict_proba(Xva)[:, 1]
    valid_mask = np.isfinite(oof_logit) & np.isfinite(oof_tree)
    blend_weight = 0.65
    calibrator = None
    if valid_mask.any():
        best_loss = float("inf")
        best_weight = blend_weight
        for weight in np.linspace(0.0, 1.0, 21):
            blend = weight * oof_logit[valid_mask] + (1 - weight) * oof_tree[valid_mask]
            loss = log_loss(y[valid_mask], np.clip(blend, 1e-6, 1 - 1e-6))
            if loss < best_loss:
                best_loss = loss
                best_weight = weight
        blend_weight = float(best_weight)
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(blend_weight * oof_logit[valid_mask] + (1 - blend_weight) * oof_tree[valid_mask], y[valid_mask])
    logit.fit(X_scaled, y)
    tree.fit(X_imp, y)
    pred = predict_matchup_proba_from_matrix({"mode": "trained", "feature_cols": BASE_FEATURES, "imputer": imputer, "scaler": scaler, "logit": logit, "tree": tree, "blend_weight": blend_weight, "calibrator": calibrator}, X)
    metrics = {
        "log_loss": float(log_loss(y, np.clip(pred, 1e-6, 1 - 1e-6))),
        "accuracy": float(accuracy_score(y, pred >= 0.5)),
        "n_games": int(len(train))
    }
    return {"mode": "trained", "feature_cols": BASE_FEATURES, "imputer": imputer, "scaler": scaler, "logit": logit, "tree": tree, "blend_weight": blend_weight, "calibrator": calibrator, "metrics": metrics}

def predict_matchup_proba_from_matrix(model, X):
    if model.get("mode") == "heuristic":
        frame = pd.DataFrame(X).copy()
        return np.asarray(heuristic_probability(frame))
    if isinstance(X, pd.DataFrame):
        frame = X[model["feature_cols"]].copy()
    else:
        frame = pd.DataFrame(X, columns=model["feature_cols"])
    X_imp = model["imputer"].transform(frame)
    X_scaled = model["scaler"].transform(X_imp)
    p1 = model["logit"].predict_proba(X_scaled)[:, 1]
    p2 = model["tree"].predict_proba(X_imp)[:, 1]
    blend = model["blend_weight"] * p1 + (1 - model["blend_weight"]) * p2
    if model.get("calibrator") is not None:
        blend = model["calibrator"].transform(blend)
    return np.asarray(np.clip(blend, 1e-6, 1 - 1e-6))

def heuristic_probability(frame):
    f = frame.copy()
    score = (
        -0.08 * f.get("seed_diff", 0).fillna(0)
        -0.012 * f.get("rank_diff", 0).fillna(0)
        +6.0 * f.get("barthag_diff", 0).fillna(0)
        +0.03 * f.get("eff_margin_diff", 0).fillna(0)
        +0.012 * f.get("adj_oe_diff", 0).fillna(0)
        -0.012 * f.get("adj_de_diff", 0).fillna(0)
        +1.8 * f.get("momentum_14_diff", 0).fillna(0)
        +0.9 * f.get("program_3y_diff", 0).fillna(0)
        +0.6 * f.get("roster_top3_porpag_diff", 0).fillna(0)
    )
    return 1 / (1 + np.exp(-score))

def train_score_models(train):
    if len(train) == 0 or "margin" not in train.columns:
        return {"mode": "formula", "margin_sd": 10.5, "total_sd": 13.5}
    X = train[SCORE_FEATURES].copy()
    margin = train["margin"].astype(float).to_numpy()
    total = train["total"].astype(float).to_numpy()
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)
    margin_linear = ElasticNet(alpha=0.02, l1_ratio=0.2, max_iter=5000)
    margin_tree = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.04, max_iter=300, min_samples_leaf=20)
    total_linear = ElasticNet(alpha=0.02, l1_ratio=0.2, max_iter=5000)
    total_tree = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.04, max_iter=300, min_samples_leaf=20)
    margin_linear.fit(X_scaled, margin)
    margin_tree.fit(X_imp, margin)
    total_linear.fit(X_scaled, total)
    total_tree.fit(X_imp, total)
    margin_pred = 0.5 * margin_linear.predict(X_scaled) + 0.5 * margin_tree.predict(X_imp)
    total_pred = 0.5 * total_linear.predict(X_scaled) + 0.5 * total_tree.predict(X_imp)
    return {
        "mode": "trained",
        "feature_cols": SCORE_FEATURES,
        "imputer": imputer,
        "scaler": scaler,
        "margin_linear": margin_linear,
        "margin_tree": margin_tree,
        "total_linear": total_linear,
        "total_tree": total_tree,
        "margin_sd": float(max(6.0, np.std(margin - margin_pred))),
        "total_sd": float(max(8.0, np.std(total - total_pred))),
        "mae_margin": float(mean_absolute_error(margin, margin_pred)),
        "mae_total": float(mean_absolute_error(total, total_pred))
    }

def predict_scores(score_model, feature_row, team1, team2, n_sims=50000, seed=7):
    if score_model.get("mode") == "formula":
        poss = np.nanmean([team1.get("adj_tempo"), team2.get("adj_tempo")])
        poss = 67.5 if pd.isna(poss) else poss
        avg_eff = 107.0
        t1_ppp = (team1.get("adj_oe", avg_eff) / avg_eff) * (team2.get("adj_de", avg_eff) / avg_eff)
        t2_ppp = (team2.get("adj_oe", avg_eff) / avg_eff) * (team1.get("adj_de", avg_eff) / avg_eff)
        team1_mean = max(45, poss * t1_ppp)
        team2_mean = max(45, poss * t2_ppp)
        margin_mean = team1_mean - team2_mean
        total_mean = team1_mean + team2_mean
        margin_sd = score_model.get("margin_sd", 10.5)
        total_sd = score_model.get("total_sd", 13.5)
    else:
        X = pd.DataFrame([feature_row])[score_model["feature_cols"]]
        X_imp = score_model["imputer"].transform(X)
        X_scaled = score_model["scaler"].transform(X_imp)
        margin_mean = 0.5 * score_model["margin_linear"].predict(X_scaled)[0] + 0.5 * score_model["margin_tree"].predict(X_imp)[0]
        total_mean = 0.5 * score_model["total_linear"].predict(X_scaled)[0] + 0.5 * score_model["total_tree"].predict(X_imp)[0]
        margin_sd = score_model["margin_sd"]
        total_sd = score_model["total_sd"]
    rng = np.random.default_rng(seed)
    margins = rng.normal(margin_mean, margin_sd, n_sims)
    totals = np.maximum(rng.normal(total_mean, total_sd, n_sims), 90)
    t1_scores = np.maximum((totals + margins) / 2, 35)
    t2_scores = np.maximum((totals - margins) / 2, 35)
    return {
        "team1_score_mean": float(np.mean(t1_scores)),
        "team2_score_mean": float(np.mean(t2_scores)),
        "team1_score_p05": float(np.percentile(t1_scores, 5)),
        "team1_score_p95": float(np.percentile(t1_scores, 95)),
        "team2_score_p05": float(np.percentile(t2_scores, 5)),
        "team2_score_p95": float(np.percentile(t2_scores, 95)),
        "margin_mean": float(np.mean(t1_scores - t2_scores)),
        "margin_sd": float(np.std(t1_scores - t2_scores)),
        "total_mean": float(np.mean(t1_scores + t2_scores)),
        "team1_win_pct": float(np.mean(t1_scores > t2_scores)),
        "close_game_pct": float(np.mean(np.abs(t1_scores - t2_scores) <= 3)),
        "blowout_pct": float(np.mean(np.abs(t1_scores - t2_scores) >= 10)),
        "pace_proxy": float(np.nanmean([team1.get("adj_tempo"), team2.get("adj_tempo")])),
        "team1_scores": t1_scores,
        "team2_scores": t2_scores
    }

def save_pickle(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(obj, handle)

def load_pickle(path):
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "rb") as handle:
        return pickle.load(handle)

def fit_or_load_models(model_dir="artifacts/models", kaggle_dir=None, cache_dir="cache", force=False):
    model_dir = Path(model_dir)
    matchup_path = model_dir / "matchup_model.pkl"
    score_path = model_dir / "score_model.pkl"
    matchup_model = None if force else load_pickle(matchup_path)
    score_model = None if force else load_pickle(score_path)
    if matchup_model is not None and score_model is not None:
        return matchup_model, score_model
    train = build_training_dataset(kaggle_dir=kaggle_dir, cache_dir=cache_dir)
    matchup_model = train_matchup_model(train)
    score_model = train_score_models(train)
    save_pickle(matchup_model, matchup_path)
    save_pickle(score_model, score_path)
    return matchup_model, score_model

def get_current_team_table(year=2026, cache_dir="cache"):
    return build_team_feature_table(year=year, cache_dir=cache_dir)

def team_lookup(feature_table):
    name_col = "team"
    if name_col not in feature_table.columns:
        for candidate in ["team_x", "team_y", "TeamName"]:
            if candidate in feature_table.columns:
                name_col = candidate
                break
        else:
            raise KeyError(f"No team name column found. Columns: {list(feature_table.columns)}")
    out = {}
    for row in feature_table.to_dict("records"):
        out[row[name_col]] = row
    return out

def adjust_probability(prob, gear=0, upset_factor=0.0):
    prob = np.clip(prob, 1e-6, 1 - 1e-6)
    logit = math.log(prob / (1 - prob))
    temperature = 1.0 + 0.25 * gear
    adj = 1 / (1 + math.exp(-(logit / temperature)))
    if upset_factor != 0:
        adj = 0.5 + (adj - 0.5) * (1 - 0.25 * upset_factor)
    return float(np.clip(adj, 1e-6, 1 - 1e-6))

def predict_two_teams(team_a, team_b, feature_table, matchup_model, score_model, recent_form_weight=1.0, program_weight=1.0, roster_weight=0.1, gear=0, upset_factor=0.0):
    lookup = team_lookup(feature_table)
    a = lookup[team_a]
    b = lookup[team_b]
    features = make_feature_vector(a, b, recent_form_weight=recent_form_weight, program_weight=program_weight)
    prob = predict_matchup_proba_from_matrix(matchup_model, pd.DataFrame([features]))[0]
    roster_edge = np.tanh(0.08 * features.get("roster_top3_porpag_diff", 0) + 0.5 * features.get("roster_exp_diff", 0))
    prob = (1 - roster_weight) * prob + roster_weight * (0.5 + 0.5 * roster_edge)
    prob = adjust_probability(prob, gear=gear, upset_factor=upset_factor)
    score = predict_scores(score_model, features, a, b)
    contributions = explain_matchup_contributions(matchup_model, features)
    return {"team_a": team_a, "team_b": team_b, "prob_team_a": prob, "prob_team_b": 1 - prob, "features": features, "score": score, "contributions": contributions}

def explain_matchup_contributions(model, feature_row, top_n=6):
    if model.get("mode") != "trained":
        ranked = sorted(feature_row.items(), key=lambda kv: abs(0 if pd.isna(kv[1]) else kv[1]), reverse=True)
        return [{"feature": k, "value": float(v) if pd.notna(v) else np.nan} for k, v in ranked[:top_n]]
    series = pd.Series(feature_row)[model["feature_cols"]].astype(float)
    x_imp = model["imputer"].transform(pd.DataFrame([series]))
    z = model["scaler"].transform(x_imp)[0]
    coef = model["logit"].coef_[0]
    contrib = z * coef
    pairs = sorted(zip(model["feature_cols"], contrib, series.values), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    return [{"feature": f, "contribution": float(c), "raw_value": float(v) if pd.notna(v) else np.nan} for f, c, v in pairs]

def simulate_full_bracket(feature_table, matchup_model, n_sims=20000, gear=0, recent_form_weight=1.0, program_weight=1.0, upset_factor=0.0, seed=7):
    rng = np.random.default_rng(seed)
    lookup = team_lookup(feature_table)
    advancement = {team: {"R32": 0, "S16": 0, "E8": 0, "F4": 0, "NCG": 0, "Champ": 0} for team in lookup}
    deterministic = {}
    for _ in range(n_sims):
        region_winners = {}
        semifinalists = []
        for region, teams in BRACKET_2026.items():
            current = teams[:]
            round_tags = ["R32", "S16", "E8", "F4"]
            for round_tag in round_tags:
                next_round = []
                for i in range(0, len(current), 2):
                    seed_a, team_a = current[i]
                    seed_b, team_b = current[i + 1]
                    row = make_feature_vector(lookup[team_a], lookup[team_b], recent_form_weight=recent_form_weight, program_weight=program_weight)
                    prob = predict_matchup_proba_from_matrix(matchup_model, pd.DataFrame([row]))[0]
                    prob = adjust_probability(prob, gear=gear, upset_factor=upset_factor)
                    winner = (seed_a, team_a) if rng.random() < prob else (seed_b, team_b)
                    next_round.append(winner)
                    advancement[winner[1]][round_tag] += 1
                current = next_round
            region_winners[region] = current[0]
            semifinalists.append(current[0])
        ff_a = region_winners["East"]
        ff_b = region_winners["South"]
        ff_c = region_winners["West"]
        ff_d = region_winners["Midwest"]
        semi1 = _play_game(ff_a, ff_b, lookup, matchup_model, gear, recent_form_weight, program_weight, upset_factor, rng)
        semi2 = _play_game(ff_c, ff_d, lookup, matchup_model, gear, recent_form_weight, program_weight, upset_factor, rng)
        advancement[semi1[1]]["NCG"] += 1
        advancement[semi2[1]]["NCG"] += 1
        champ = _play_game(semi1, semi2, lookup, matchup_model, gear, recent_form_weight, program_weight, upset_factor, rng)
        advancement[semi1[1]]["F4"] += 1
        advancement[semi2[1]]["F4"] += 1
        advancement[champ[1]]["Champ"] += 1
    for team in advancement:
        for round_name in advancement[team]:
            advancement[team][round_name] /= n_sims
    deterministic = build_expected_value_bracket(feature_table, matchup_model, gear=gear, recent_form_weight=recent_form_weight, program_weight=program_weight, upset_factor=upset_factor)
    return advancement, deterministic

def _play_game(team_a, team_b, lookup, matchup_model, gear, recent_form_weight, program_weight, upset_factor, rng):
    seed_a, name_a = team_a
    seed_b, name_b = team_b
    row = make_feature_vector(lookup[name_a], lookup[name_b], recent_form_weight=recent_form_weight, program_weight=program_weight)
    prob = predict_matchup_proba_from_matrix(matchup_model, pd.DataFrame([row]))[0]
    prob = adjust_probability(prob, gear=gear, upset_factor=upset_factor)
    return (seed_a, name_a) if rng.random() < prob else (seed_b, name_b)

def build_expected_value_bracket(feature_table, matchup_model, gear=0, recent_form_weight=1.0, program_weight=1.0, upset_factor=0.0):
    lookup = team_lookup(feature_table)
    picks = {"regions": {}, "final_four": [], "champion": None}
    region_champs = {}
    for region, teams in BRACKET_2026.items():
        current = teams[:]
        region_rounds = []
        while len(current) > 1:
            next_round = []
            games = []
            for i in range(0, len(current), 2):
                seed_a, team_a = current[i]
                seed_b, team_b = current[i + 1]
                row = make_feature_vector(lookup[team_a], lookup[team_b], recent_form_weight=recent_form_weight, program_weight=program_weight)
                prob = predict_matchup_proba_from_matrix(matchup_model, pd.DataFrame([row]))[0]
                prob = adjust_probability(prob, gear=gear, upset_factor=upset_factor)
                if prob >= 0.5:
                    winner = (seed_a, team_a)
                    win_prob = prob
                    loser = (seed_b, team_b)
                else:
                    winner = (seed_b, team_b)
                    win_prob = 1 - prob
                    loser = (seed_a, team_a)
                next_round.append(winner)
                games.append({"team_a": team_a, "team_b": team_b, "winner": winner[1], "winner_prob": float(win_prob), "loser": loser[1]})
            region_rounds.append(games)
            current = next_round
        picks["regions"][region] = region_rounds
        region_champs[region] = current[0]
    semi1 = region_champs["East"], region_champs["South"]
    semi2 = region_champs["West"], region_champs["Midwest"]
    ff_winner1, ff_game1 = _deterministic_game(semi1[0], semi1[1], lookup, matchup_model, gear, recent_form_weight, program_weight, upset_factor)
    ff_winner2, ff_game2 = _deterministic_game(semi2[0], semi2[1], lookup, matchup_model, gear, recent_form_weight, program_weight, upset_factor)
    champ, title_game = _deterministic_game(ff_winner1, ff_winner2, lookup, matchup_model, gear, recent_form_weight, program_weight, upset_factor)
    picks["final_four"] = [ff_game1, ff_game2, title_game]
    picks["champion"] = champ[1]
    return picks

def _deterministic_game(team_a, team_b, lookup, matchup_model, gear, recent_form_weight, program_weight, upset_factor):
    seed_a, name_a = team_a
    seed_b, name_b = team_b
    row = make_feature_vector(lookup[name_a], lookup[name_b], recent_form_weight=recent_form_weight, program_weight=program_weight)
    prob = predict_matchup_proba_from_matrix(matchup_model, pd.DataFrame([row]))[0]
    prob = adjust_probability(prob, gear=gear, upset_factor=upset_factor)
    if prob >= 0.5:
        return (seed_a, name_a), {"team_a": name_a, "team_b": name_b, "winner": name_a, "winner_prob": float(prob)}
    return (seed_b, name_b), {"team_a": name_a, "team_b": name_b, "winner": name_b, "winner_prob": float(1 - prob)}

def bracket_probability_table(advancement):
    rows = []
    for team, probs in advancement.items():
        row = {"team": team}
        row.update(probs)
        rows.append(row)
    frame = pd.DataFrame(rows)
    return frame.sort_values(["Champ", "NCG", "F4", "E8", "S16", "R32"], ascending=False).reset_index(drop=True)

def export_bracket_outputs(advancement, deterministic, output_dir="artifacts/bracket"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prob_table = bracket_probability_table(advancement)
    prob_table.to_csv(output_dir / "advance_probabilities.csv", index=False)
    with open(output_dir / "deterministic_bracket.json", "w", encoding="utf-8") as handle:
        json.dump(deterministic, handle, indent=2)
    return output_dir / "advance_probabilities.csv", output_dir / "deterministic_bracket.json"

def export_head_to_head_output(result, output_dir="artifacts/head_to_head"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_a = re.sub(r"[^A-Za-z0-9]+", "_", result["team_a"]).strip("_")
    safe_b = re.sub(r"[^A-Za-z0-9]+", "_", result["team_b"]).strip("_")
    path = output_dir / f"{safe_a}_vs_{safe_b}.json"
    serializable = {
        "team_a": result["team_a"],
        "team_b": result["team_b"],
        "prob_team_a": result["prob_team_a"],
        "prob_team_b": result["prob_team_b"],
        "features": result["features"],
        "score": {k: v for k, v in result["score"].items() if not isinstance(v, np.ndarray)},
        "contributions": result["contributions"]
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2)
    return path

def current_team_names():
    teams = []
    for region in BRACKET_2026.values():
        teams.extend([team for _, team in region])
    return sorted(teams)

def summarize_result_text(result):
    team_a = result["team_a"]
    team_b = result["team_b"]
    score = result["score"]
    winner = team_a if result["prob_team_a"] >= 0.5 else team_b
    lines = [
        f"{team_a} win probability: {result['prob_team_a']:.3f}",
        f"{team_b} win probability: {result['prob_team_b']:.3f}",
        f"Projected score: {team_a} {score['team1_score_mean']:.1f} - {team_b} {score['team2_score_mean']:.1f}",
        f"Expected winner: {winner}",
        "Top drivers:"
    ]
    for item in result["contributions"]:
        if "contribution" in item:
            lines.append(f"{item['feature']}: contribution={item['contribution']:.4f}, raw={item['raw_value']:.4f}")
        else:
            lines.append(f"{item['feature']}: value={item['value']:.4f}")
    return "\n".join(lines)
