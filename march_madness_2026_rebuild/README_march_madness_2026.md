
# March Madness 2026 rebuild

## Files
- `march_madness_2026_core.py`
- `01_bracket_model.py`
- `02_head_to_head_model.py`
- `march_madness_2026_app.py`
- `01_bracket_model.ipynb`
- `02_head_to_head_model.ipynb`

## What changed
- Replaced the fixed 10-feature-only approach with a wider feature table that can use seed, current Bart ratings, 7/14/30 day momentum, 1-year and 3-year program trajectory, conference strength, and optional roster aggregation from Bart player-season data
- Added rolling-season training with a calibrated LogisticRegression + HistGradientBoosting blend
- Added a separate score model for margin and total
- Added full bracket simulation and deterministic bracket export
- Added a command-line matchup runner and a Streamlit app

## Data sources expected by the code
- Public Bart Torvik team snapshots and time-machine snapshots
- Public Bart player-season export
- Optional Kaggle March Machine Learning Mania CSVs for historical tournament results and seeds

## Optional Kaggle files
Place these in a folder and pass `--kaggle-dir`:
- `MTeams.csv`
- `MNCAATourneySeeds.csv`
- `MNCAATourneyCompactResults.csv`
- `MNCAATourneyDetailedResults.csv` if available

## Examples
```bash
python 01_bracket_model.py --kaggle-dir path/to/kaggle --sims 20000
python 02_head_to_head_model.py --kaggle-dir path/to/kaggle --team-a Duke --team-b Houston
streamlit run march_madness_2026_app.py
```
