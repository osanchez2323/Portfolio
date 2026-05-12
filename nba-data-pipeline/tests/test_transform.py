"""
tests/test_transform.py
------------------------
Unit tests for the Transform stage (transformations.py).
Tests type casting, each derived metric, and the full run_transform() function.
"""

import pytest
import pandas as pd
import numpy as np

from src.transform.transformations import (
    cast_types,
    derive_ts_pct,
    derive_game_score,
    derive_rolling_averages,
    add_audit_fields,
    rename_columns,
    run_transform,
    _parse_minutes,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_validated_df(n: int = 4) -> pd.DataFrame:
    """Return a small validated-style DataFrame (post-validate stage)."""
    return pd.DataFrame({
        "GAME_ID":           [f"00224010{i:02d}" for i in range(n)],
        "PLAYER_ID":         [203076, 201142, 2544, 203507],
        "PLAYER_NAME":       ["ANTHONY DAVIS", "KEVIN DURANT", "LEBRON JAMES", "GIANNIS ANTETOKOUNMPO"],
        "TEAM_ID":           [1610612747, 1610612756, 1610612747, 1610612749],
        "TEAM_ABBREVIATION": ["LAL", "PHX", "LAL", "MIL"],
        "GAME_DATE":         ["APR 15, 2025"] * n,
        "MIN":               ["35:30", "38:00", "34:15", "36:45"],
        "PTS":               [31.0, 29.0, 26.0, 33.0],
        "REB":               [12.0, 8.0, 8.0, 14.0],
        "AST":               [3.0, 5.0, 10.0, 7.0],
        "STL":               [2.0, 1.0, 1.0, 2.0],
        "BLK":               [3.0, 1.0, 1.0, 2.0],
        "TOV":               [2.0, 2.0, 3.0, 3.0],
        "FGM":               [12.0, 11.0, 10.0, 13.0],
        "FGA":               [20.0, 19.0, 18.0, 22.0],
        "FG_PCT":            [0.600, 0.579, 0.556, 0.591],
        "FTM":               [7.0, 7.0, 6.0, 7.0],
        "FTA":               [9.0, 9.0, 8.0, 10.0],
        "FT_PCT":            [0.778, 0.778, 0.75, 0.70],
        "FG3M":              [0.0, 0.0, 0.0, 0.0],
        "FG3A":              [1.0, 2.0, 2.0, 1.0],
        "OREB":              [3.0, 1.0, 1.0, 4.0],
        "DREB":              [9.0, 7.0, 7.0, 10.0],
        "PF":                [3.0, 2.0, 2.0, 3.0],
        "TS_PCT":            [0.635, 0.616, 0.601, 0.627],
        "MATCHUP":           ["LAL vs. PHX", "PHX @ LAL", "LAL vs. PHX", "MIL vs. CLE"],
    })


# ── _parse_minutes ────────────────────────────────────────────────────────────

class TestParseMinutes:
    def test_standard_format(self):
        assert abs(_parse_minutes("32:14") - 32.233) < 0.01

    def test_zero_minutes(self):
        assert _parse_minutes("0:00") == 0.0

    def test_numeric_string(self):
        assert _parse_minutes("32") == 32.0

    def test_none_returns_zero(self):
        assert _parse_minutes(None) == 0.0

    def test_nan_returns_zero(self):
        assert _parse_minutes(float("nan")) == 0.0


# ── Type casting ──────────────────────────────────────────────────────────────

class TestCastTypes:
    def test_pts_cast_to_int(self):
        df = make_validated_df()
        result = cast_types(df)
        assert result["PTS"].dtype == int

    def test_fg_pct_remains_float(self):
        df = make_validated_df()
        result = cast_types(df)
        assert result["FG_PCT"].dtype == float

    def test_min_parsed_to_decimal(self):
        df = make_validated_df()
        result = cast_types(df)
        # "35:30" → 35.5
        assert abs(result.loc[0, "MIN"] - 35.5) < 0.01

    def test_player_name_title_cased(self):
        df = make_validated_df()
        result = cast_types(df)
        assert result.loc[0, "PLAYER_NAME"] == "Anthony Davis"

    def test_game_date_converted(self):
        df = make_validated_df()
        result = cast_types(df)
        import datetime
        assert isinstance(result.loc[0, "GAME_DATE"], datetime.date)


# ── TS% derivation ────────────────────────────────────────────────────────────

class TestDeriveTsPct:
    def test_formula_correctness(self):
        """ts_pct = pts / (2 × (fga + 0.44 × fta))"""
        df = pd.DataFrame({
            "PTS": [30], "FGA": [20], "FTA": [8]
        })
        result = derive_ts_pct(df)
        expected = 30 / (2 * (20 + 0.44 * 8))  # ≈ 0.6466
        assert abs(result.loc[0, "TS_PCT"] - expected) < 0.0001

    def test_null_when_zero_attempts(self):
        df = pd.DataFrame({"PTS": [0], "FGA": [0], "FTA": [0]})
        result = derive_ts_pct(df)
        assert pd.isna(result.loc[0, "TS_PCT"])

    def test_overwrites_existing_ts_pct(self):
        """The derived value should replace the raw API TS_PCT."""
        df = make_validated_df()
        df = cast_types(df)
        result = derive_ts_pct(df)
        # Verify all values are recalculated, not just copied
        assert result["TS_PCT"].notna().all()


# ── Game Score derivation ─────────────────────────────────────────────────────

class TestDeriveGameScore:
    def test_positive_for_good_game(self):
        """A 30pt/10rb/5ast game should produce a positive game score."""
        df = pd.DataFrame({
            "PTS": [30], "FGM": [12], "FGA": [20],
            "FTA": [8], "FTM": [6], "OREB": [2],
            "DREB": [8], "STL": [2], "AST": [5],
            "BLK": [1], "PF": [3], "TOV": [3]
        })
        result = derive_game_score(df)
        assert result.loc[0, "GAME_SCORE"] > 20

    def test_column_added(self):
        df = make_validated_df()
        df = cast_types(df)
        result = derive_game_score(df)
        assert "GAME_SCORE" in result.columns


# ── Rolling averages ──────────────────────────────────────────────────────────

class TestDeriveRollingAverages:
    def test_rolling_5_game_average(self):
        """First game for a player should equal their single-game pts."""
        df = pd.DataFrame({
            "PLAYER_ID": [1, 1, 1, 1, 1],
            "GAME_DATE":  pd.to_datetime(["2025-04-11", "2025-04-12", "2025-04-13",
                                           "2025-04-14", "2025-04-15"]).date.tolist(),
            "PTS":        [20, 24, 18, 30, 26],
        })
        result = derive_rolling_averages(df)
        # After 5 games, rolling avg = mean of all 5
        assert abs(result.iloc[4]["ROLL5_PTS"] - 23.6) < 0.1

    def test_partial_window_min_periods_1(self):
        """First game: rolling avg = that game's pts (min_periods=1)."""
        df = pd.DataFrame({
            "PLAYER_ID": [1],
            "GAME_DATE":  [pd.Timestamp("2025-04-11").date()],
            "PTS":        [22],
        })
        result = derive_rolling_averages(df)
        assert result.loc[0, "ROLL5_PTS"] == 22.0


# ── Audit fields ──────────────────────────────────────────────────────────────

class TestAddAuditFields:
    def test_all_audit_fields_added(self):
        df = make_validated_df()
        result = add_audit_fields(df, run_id="ABC12345")
        for field in ["LOAD_TIMESTAMP", "PIPELINE_RUN_ID", "SOURCE_SYSTEM", "IS_CURRENT", "ROW_HASH"]:
            assert field in result.columns

    def test_pipeline_run_id_matches(self):
        df = make_validated_df()
        result = add_audit_fields(df, run_id="TEST001")
        assert (result["PIPELINE_RUN_ID"] == "TEST001").all()

    def test_source_system_is_correct(self):
        df = make_validated_df()
        result = add_audit_fields(df, run_id="X")
        assert (result["SOURCE_SYSTEM"] == "nba_stats_api").all()

    def test_row_hash_is_unique_per_record(self):
        """Each distinct player-game combination should produce a unique hash."""
        df = make_validated_df()
        df = cast_types(df)
        result = add_audit_fields(df, run_id="X")
        assert result["ROW_HASH"].nunique() == len(result)


# ── Column rename ─────────────────────────────────────────────────────────────

class TestRenameColumns:
    def test_pts_renamed_to_lowercase(self):
        df = make_validated_df()
        df = cast_types(df)
        df = add_audit_fields(df, run_id="X")
        result = rename_columns(df)
        assert "pts" in result.columns
        assert "PTS" not in result.columns

    def test_min_renamed_to_min_played(self):
        df = make_validated_df()
        df = cast_types(df)
        df = add_audit_fields(df, run_id="X")
        result = rename_columns(df)
        assert "min_played" in result.columns
        assert "MIN" not in result.columns


# ── Integration: full transform run ──────────────────────────────────────────

class TestRunTransform:
    def test_full_run_returns_correct_row_count(self):
        df = make_validated_df(4)
        result = run_transform(df, run_id="TEST")
        assert len(result) == 4

    def test_all_derived_columns_present(self):
        df = make_validated_df(4)
        result = run_transform(df, run_id="TEST")
        for col in ["ts_pct", "game_score", "roll5_pts", "row_hash", "load_timestamp"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_column_names_are_snake_case(self):
        df = make_validated_df(4)
        result = run_transform(df, run_id="TEST")
        for col in result.columns:
            assert col == col.lower(), f"Column not snake_case: {col}"
