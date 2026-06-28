import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "ml_experiments" / "scripts"))

from score_sus import (  # noqa: E402
    grade_for_score,
    score_file,
    score_single_response,
    summarize_by_channel,
)


def test_all_max_positive_responses_gives_perfect_score():
    # Best possible SUS outcome: strongly agree (5) on positive items,
    # strongly disagree (1) on negative items.
    row = pd.Series({"q1": 5, "q2": 1, "q3": 5, "q4": 1, "q5": 5,
                      "q6": 1, "q7": 5, "q8": 1, "q9": 5, "q10": 1})
    assert score_single_response(row) == 100.0


def test_all_worst_responses_gives_zero_score():
    row = pd.Series({"q1": 1, "q2": 5, "q3": 1, "q4": 5, "q5": 1,
                      "q6": 5, "q7": 1, "q8": 5, "q9": 1, "q10": 5})
    assert score_single_response(row) == 0.0


def test_all_neutral_responses_gives_fifty():
    row = pd.Series({f"q{i}": 3 for i in range(1, 11)})
    assert score_single_response(row) == 50.0


def test_known_worked_example():
    # Hand-verified: odd items [5,5,5,5,5] -> contributions [4]*5 = 20
    # even items [2,1,1,2,1] -> contributions [3,4,4,3,4] = 18
    # total = 38 -> score = 38 * 2.5 = 95.0
    row = pd.Series({"q1": 5, "q2": 2, "q3": 5, "q4": 1, "q5": 5,
                      "q6": 1, "q7": 5, "q8": 2, "q9": 5, "q10": 1})
    assert score_single_response(row) == 95.0


def test_missing_response_raises_clear_error():
    row = pd.Series({"q1": None, "q2": 2, "q3": 5, "q4": 1, "q5": 5,
                      "q6": 1, "q7": 5, "q8": 2, "q9": 5, "q10": 1})
    with pytest.raises(ValueError, match="Missing response for q1"):
        score_single_response(row)


def test_out_of_range_response_raises_clear_error():
    row = pd.Series({"q1": 7, "q2": 2, "q3": 5, "q4": 1, "q5": 5,
                      "q6": 1, "q7": 5, "q8": 2, "q9": 5, "q10": 1})
    with pytest.raises(ValueError, match="must be an integer 1-5"):
        score_single_response(row)


def test_score_file_requires_all_columns(tmp_path):
    csv_path = tmp_path / "missing_col.csv"
    csv_path.write_text("participant_id,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10\nP1,5,2,5,1,5,1,5,2,5,1\n")
    with pytest.raises(ValueError, match="missing required column"):
        score_file(str(csv_path))


def test_score_file_computes_correct_scores(tmp_path):
    csv_path = tmp_path / "valid.csv"
    csv_path.write_text(
        "participant_id,channel,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10\n"
        "P1,whatsapp,5,2,5,1,5,1,5,2,5,1\n"
        "P2,ussd,3,3,3,3,3,3,3,3,3,3\n"
    )
    scored = score_file(str(csv_path))
    assert len(scored) == 2
    assert scored.loc[scored["participant_id"] == "P1", "sus_score"].iloc[0] == 95.0
    assert scored.loc[scored["participant_id"] == "P2", "sus_score"].iloc[0] == 50.0


def test_grade_bands_are_monotonic_and_cover_full_range():
    assert grade_for_score(100) == "A+ (top ~10%)"
    assert grade_for_score(68) != "F (below average)"  # published average should not be graded F
    assert grade_for_score(0) == "F (below average)"
    assert grade_for_score(-5) == "F (below average)"  # defensive: scores are 0-100 by construction, but shouldn't crash


def test_summarize_by_channel_groups_correctly(tmp_path):
    csv_path = tmp_path / "multi.csv"
    csv_path.write_text(
        "participant_id,channel,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10\n"
        "P1,whatsapp,5,2,5,1,5,1,5,2,5,1\n"
        "P2,whatsapp,3,3,3,3,3,3,3,3,3,3\n"
        "P3,ussd,1,5,1,5,1,5,1,5,1,5\n"
    )
    scored = score_file(str(csv_path))
    summary = summarize_by_channel(scored)
    whatsapp_row = summary[summary["channel"] == "whatsapp"].iloc[0]
    ussd_row = summary[summary["channel"] == "ussd"].iloc[0]
    assert whatsapp_row["n_participants"] == 2
    assert whatsapp_row["mean_sus_score"] == pytest.approx((95.0 + 50.0) / 2)
    assert ussd_row["n_participants"] == 1
    assert ussd_row["mean_sus_score"] == 0.0
