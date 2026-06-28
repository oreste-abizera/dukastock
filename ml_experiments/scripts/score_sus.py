"""
Score System Usability Scale (SUS) responses.

Implements the standard SUS scoring formula (Brooke, 1986):
  - Odd-numbered items (1,3,5,7,9 -- positively worded): contribution = response - 1
  - Even-numbered items (2,4,6,8,10 -- negatively worded): contribution = 5 - response
  - Sum the 10 contributions (range 0-40), multiply by 2.5 -> final score 0-100.

Per-participant scores are NOT averaged from raw question responses
directly (that would be a different, invalid computation) -- each
participant's 0-100 score is computed first, and only THEN averaged
across participants. This script does it in the correct order.

See docs/SUS_QUESTIONNAIRE.md for the questionnaire itself, administration
protocol, and expected input CSV format.

Usage:
    python ml_experiments/scripts/score_sus.py --input sus_responses.csv

Optionally --output to also write per-participant and per-channel summary
CSVs, and --plot to render a quick bar chart of channel comparison
(matplotlib only — for a fuller, multi-panel write-up, see
ml_experiments/notebooks/04_results_dashboard.ipynb, which also has a
usability section if SUS data is present).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

ODD_ITEMS = [1, 3, 5, 7, 9]
EVEN_ITEMS = [2, 4, 6, 8, 10]

# Sauro & Lewis (2011) percentile-to-grade mapping, widely cited in SUS
# practitioner literature; used here purely as a descriptive label, not a
# pass/fail threshold.
GRADE_BANDS = [
    (84.1, "A+ (top ~10%)"),
    (80.8, "A (top ~15%)"),
    (78.9, "A-"),
    (77.2, "B+"),
    (74.1, "B"),
    (72.6, "B-"),
    (71.1, "C+"),
    (65.0, "C"),
    (62.7, "C-"),
    (51.7, "D"),
    (0.0, "F (below average)"),
]


def score_single_response(row: pd.Series) -> float:
    """Compute one participant's 0-100 SUS score from their 10 raw 1-5
    answers (columns q1..q10)."""
    total = 0
    for item in ODD_ITEMS:
        response = row[f"q{item}"]
        _validate_response(item, response)
        total += response - 1
    for item in EVEN_ITEMS:
        response = row[f"q{item}"]
        _validate_response(item, response)
        total += 5 - response
    return total * 2.5


def _validate_response(item: int, response) -> None:
    if pd.isna(response):
        raise ValueError(f"Missing response for q{item} — SUS requires all 10 items answered.")
    if response not in (1, 2, 3, 4, 5):
        raise ValueError(f"q{item} response must be an integer 1-5, got {response!r}.")


def grade_for_score(score: float) -> str:
    for threshold, label in GRADE_BANDS:
        if score >= threshold:
            return label
    return "F (below average)"


def score_file(input_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    required_cols = {"participant_id", "channel"} | {f"q{i}" for i in range(1, 11)}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required column(s): {sorted(missing)}")

    df["sus_score"] = df.apply(score_single_response, axis=1)
    df["grade"] = df["sus_score"].apply(grade_for_score)
    return df


def summarize_by_channel(scored_df: pd.DataFrame) -> pd.DataFrame:
    summary = scored_df.groupby("channel").agg(
        n_participants=("participant_id", "count"),
        mean_sus_score=("sus_score", "mean"),
        min_sus_score=("sus_score", "min"),
        max_sus_score=("sus_score", "max"),
        std_sus_score=("sus_score", "std"),
    ).reset_index()
    summary["mean_grade"] = summary["mean_sus_score"].apply(grade_for_score)
    return summary


def main(input_path: str, output_dir: Optional[str], make_plot: bool):
    scored = score_file(input_path)

    print("Per-participant scores:")
    print(scored[["participant_id", "channel", "sus_score", "grade"]].to_string(index=False))
    print()

    summary = summarize_by_channel(scored)
    print("Per-channel summary:")
    print(summary.to_string(index=False))
    print()
    print("Published benchmark: average SUS score across hundreds of studies is 68/100 "
          "(Sauro, n.d.). Scores above 68 are above-average usability; above ~80 is top 10-15%.")

    if len(scored) < 8:
        print()
        print(f"NOTE: n={len(scored)} participants. SUS remains directionally useful at this "
              "sample size for formative/exploratory feedback (consistent with this project's "
              "minimum-3-participants scope), but avoid presenting these numbers as "
              "statistically generalizable to the broader Duka operator population.")

    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        scored.to_csv(out_dir / "sus_scores_per_participant.csv", index=False)
        summary.to_csv(out_dir / "sus_scores_by_channel.csv", index=False)
        print(f"\nWrote per-participant and per-channel CSVs to {out_dir}")

        if make_plot:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(summary["channel"], summary["mean_sus_score"], color="#1a6e3c")
            ax.axhline(68, color="gray", linestyle="--", label="Published benchmark (68)")
            ax.set_ylim(0, 100)
            ax.set_ylabel("Mean SUS score")
            ax.set_title("DukaStock SUS score by channel")
            ax.legend()
            plot_path = out_dir / "sus_by_channel.png"
            plt.tight_layout()
            plt.savefig(plot_path, dpi=200, bbox_inches="tight")
            print(f"Saved chart to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV of raw SUS responses (see docs/SUS_QUESTIONNAIRE.md)")
    parser.add_argument("--output-dir", default=None, help="If set, write summary CSVs (and optionally a chart) here")
    parser.add_argument("--plot", action="store_true", help="Also save a bar chart comparing channels (requires --output-dir)")
    args = parser.parse_args()
    main(args.input, args.output_dir, args.plot)
