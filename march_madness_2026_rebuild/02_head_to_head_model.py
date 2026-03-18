
import argparse

from march_madness_2026_core import current_team_names, export_head_to_head_output, fit_or_load_models, get_current_team_table, predict_two_teams, summarize_result_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--team-a", required=True)
    parser.add_argument("--team-b", required=True)
    parser.add_argument("--kaggle-dir", default=None)
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--model-dir", default="artifacts/models")
    parser.add_argument("--output-dir", default="artifacts/head_to_head")
    parser.add_argument("--recent-form-weight", type=float, default=1.0)
    parser.add_argument("--program-weight", type=float, default=1.0)
    parser.add_argument("--roster-weight", type=float, default=0.1)
    parser.add_argument("--gear", type=int, default=0)
    parser.add_argument("--upset-factor", type=float, default=0.0)
    parser.add_argument("--force-train", action="store_true")
    args = parser.parse_args()
    if args.team_a == args.team_b:
        raise ValueError("Team A and Team B must be different.")
    matchup_model, score_model = fit_or_load_models(model_dir=args.model_dir, kaggle_dir=args.kaggle_dir, cache_dir=args.cache_dir, force=args.force_train)
    team_table = get_current_team_table(year=2026, cache_dir=args.cache_dir)
    team_names = set(current_team_names())
    if args.team_a not in team_names or args.team_b not in team_names:
        raise ValueError("Both teams must be in the 2026 tournament field.")
    result = predict_two_teams(args.team_a, args.team_b, team_table, matchup_model, score_model, recent_form_weight=args.recent_form_weight, program_weight=args.program_weight, roster_weight=args.roster_weight, gear=args.gear, upset_factor=args.upset_factor)
    output_path = export_head_to_head_output(result, output_dir=args.output_dir)
    print(summarize_result_text(result))
    print(f"\nSaved to {output_path}")

if __name__ == "__main__":
    main()
