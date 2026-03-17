import os
import json
import pickle
import requests
import streamlit as st

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mm_data")
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


def get_all_tournament_teams():
    teams = {}
    for region, team_list in BRACKET_2026.items():
        for seed, name in team_list:
            teams[name] = {"seed": seed, "region": region}
    return teams


@st.cache_data(ttl=3600)
def load_barttorvik(year):
    path = os.path.join(DATA_DIR, f"barttorvik_{year}.json")
    if not os.path.exists(path):
        url = f"https://barttorvik.com/{year}_team_results.json"
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code == 200:
                os.makedirs(DATA_DIR, exist_ok=True)
                with open(path, "w") as f:
                    f.write(r.text)
        except Exception as e:
            print(e)
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
            "rank": t[0], "conf": t[2], "record": rec,
            "wins": wins, "losses": losses, "win_pct": wins / total,
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


@st.cache_resource
def load_trained_model():
    model_path = os.path.join(DATA_DIR, "trained_model.pkl")
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            m = pickle.load(f)
        return m["lr"], m["lgb"], m["scaler"]
    return None, None, None
