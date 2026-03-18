
import argparse
import json
from pathlib import Path

from march_madness_2026_core import BRACKET_2026, export_bracket_outputs, fit_or_load_models, get_current_team_table, simulate_full_bracket

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle-dir", default=None)
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--model-dir", default="artifacts/models")
    parser.add_argument("--output-dir", default="artifacts/bracket")
    parser.add_argument("--sims", type=int, default=20000)
    parser.add_argument("--gear", type=int, default=0)
    parser.add_argument("--recent-form-weight", type=float, default=1.0)
    parser.add_argument("--program-weight", type=float, default=1.0)
    parser.add_argument("--upset-factor", type=float, default=0.0)
    parser.add_argument("--force-train", action="store_true")
    args = parser.parse_args()
    matchup_model, score_model = fit_or_load_models(model_dir=args.model_dir, kaggle_dir=args.kaggle_dir, cache_dir=args.cache_dir, force=args.force_train)
    team_table = get_current_team_table(year=2026, cache_dir=args.cache_dir)
    advancement, deterministic = simulate_full_bracket(team_table, matchup_model, n_sims=args.sims, gear=args.gear, recent_form_weight=args.recent_form_weight, program_weight=args.program_weight, upset_factor=args.upset_factor)
    prob_path, bracket_path = export_bracket_outputs(advancement, deterministic, output_dir=args.output_dir)
    summary = {
        "model_mode": matchup_model.get("mode"),
        "matchup_metrics": matchup_model.get("metrics", {}),
        "score_metrics": {k: v for k, v in score_model.items() if isinstance(v, (int, float, str))},
        "champion_pick": deterministic["champion"],
        "advance_probabilities_csv": str(prob_path),
        "deterministic_bracket_json": str(bracket_path)
    }
    summary_path = Path(args.output_dir) / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
