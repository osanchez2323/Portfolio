"""
tests/test_validate.py
-----------------------
Unit tests for the Validate stage (quality_checks.py).
Tests each individual check and the overall run_validate() function.
"""

import pytest
import pandas as pd
import numpy as np

from src.validate.quality_checks import (
    check_schema,
    check_inactive_players,
    check_null_ts_pct,
    check_fg_pct_range,
    check_pts_range,
    check_drop_rate,
    run_validate,
    ValidationReport,
    QAResult,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_clean_df(n: int = 5) -> pd.DataFrame:
    """Return a small clean DataFrame with all expected columns."""
    return pd.DataFrame({
        "GAME_ID":           [f"00224010{i:02d}" for i in range(n)],
        "PLAYER_ID":         list(range(1, n + 1)),
        "PLAYER_NAME":       [f"PLAYER {i}" for i in range(n)],
        "TEAM_ID":           [1610612747] * n,
        "TEAM_ABBREVIATION": ["LAL"] * n,
        "GAME_DATE":         ["APR 15, 2025"] * n,
        "MIN":               ["32:14", "28:00", "35:30", "20:00", "40:00"],
        "PTS":               [26, 18, 31, 12, 24],
        "REB":               [8, 5, 10, 3, 7],
        "AST":               [10, 4, 8, 2, 6],
        "STL":               [1, 2, 1, 0, 2],
        "BLK":               [1, 0, 3, 1, 0],
        "TOV":               [2, 1, 3, 1, 2],
        "FGM":               [10, 7, 12, 5, 9],
        "FGA":               [20, 14, 22, 10, 17],
        "FG_PCT":            [0.50, 0.50, 0.545, 0.50, 0.529],
        "FTM":               [6, 4, 7, 2, 6],
        "FTA":               [8, 5, 9, 3, 8],
        "FT_PCT":            [0.75, 0.80, 0.778, 0.667, 0.75],
        "FG3M":              [0, 0, 0, 0, 0],
        "FG3A":              [2, 1, 1, 1, 2],
        "OREB":              [1, 2, 3, 1, 1],
        "DREB":              [7, 3, 7, 2, 6],
        "PF":                [2, 3, 2, 4, 2],
        "TS_PCT":            [0.601, 0.562, 0.608, 0.522, 0.618],
    })


# ── Schema checks ─────────────────────────────────────────────────────────────

class TestCheckSchema:
    def test_passes_with_all_columns(self):
        df = make_clean_df()
        report = ValidationReport()
        result = check_schema(df, report)
        assert len(result) == 5
        assert report.checks[0].passed is True

    def test_raises_on_missing_column(self):
        df = make_clean_df().drop(columns=["PTS"])
        report = ValidationReport()
        with pytest.raises(ValueError, match="Missing columns"):
            check_schema(df, report)


# ── Inactive player checks ────────────────────────────────────────────────────

class TestCheckInactivePlayers:
    def test_drops_zero_minute_rows(self):
        df = make_clean_df()
        df.loc[0, "MIN"] = "0:00"   # Inactive player
        df.loc[1, "MIN"] = "0"      # Another inactive format
        report = ValidationReport()
        clean = check_inactive_players(df, report)
        assert len(clean) == 3
        assert report.checks[0].rows_affected == 2
        assert report.total_rows_dropped == 2

    def test_keeps_all_active_players(self):
        df = make_clean_df()
        report = ValidationReport()
        clean = check_inactive_players(df, report)
        assert len(clean) == 5
        assert report.total_rows_dropped == 0


# ── Null TS% check ────────────────────────────────────────────────────────────

class TestCheckNullTsPct:
    def test_warns_when_null_rate_exceeds_threshold(self):
        df = make_clean_df(20)
        df.loc[0:5, "TS_PCT"] = np.nan  # 30% null rate — above 2% threshold
        report = ValidationReport()
        result = check_null_ts_pct(df, report)
        assert report.checks[0].passed is False
        assert report.checks[0].rule_type == "SOFT"
        assert len(result) == 20  # Rows are NOT dropped for soft warning

    def test_passes_when_null_rate_acceptable(self):
        df = make_clean_df(100)
        df.loc[0, "TS_PCT"] = np.nan   # 1% null rate — below 2% threshold
        report = ValidationReport()
        check_null_ts_pct(df, report)
        assert report.checks[0].passed is True


# ── FG_PCT range check ────────────────────────────────────────────────────────

class TestCheckFgPctRange:
    def test_quarantines_out_of_range_rows(self):
        df = make_clean_df()
        df.loc[0, "FG_PCT"] = 1.5   # Invalid — above 1.0
        df.loc[1, "FG_PCT"] = -0.1  # Invalid — below 0.0
        report = ValidationReport()
        clean, quarantine = check_fg_pct_range(df, report)
        assert len(clean) == 3
        assert len(quarantine) == 2
        assert report.total_rows_quarantined == 2

    def test_allows_null_fg_pct(self):
        """FG_PCT can be null when a player had 0 attempts."""
        df = make_clean_df()
        df.loc[0, "FG_PCT"] = np.nan
        report = ValidationReport()
        clean, quarantine = check_fg_pct_range(df, report)
        assert len(clean) == 5
        assert len(quarantine) == 0


# ── PTS range check ───────────────────────────────────────────────────────────

class TestCheckPtsRange:
    def test_quarantines_out_of_range_pts(self):
        df = make_clean_df()
        df.loc[0, "PTS"] = 150  # Physically impossible
        report = ValidationReport()
        clean, quarantine = check_pts_range(df, report)
        assert len(clean) == 4
        assert len(quarantine) == 1

    def test_passes_valid_pts(self):
        df = make_clean_df()
        report = ValidationReport()
        clean, quarantine = check_pts_range(df, report)
        assert len(clean) == 5
        assert len(quarantine) == 0


# ── Drop rate gate ────────────────────────────────────────────────────────────

class TestCheckDropRate:
    def test_aborts_when_drop_rate_too_high(self):
        report = ValidationReport(total_rows_in=100, total_rows_dropped=10)
        with pytest.raises(RuntimeError, match="Pipeline aborted"):
            check_drop_rate(report)

    def test_passes_when_drop_rate_acceptable(self):
        report = ValidationReport(total_rows_in=100, total_rows_dropped=3)
        # Should not raise
        check_drop_rate(report)


# ── Integration: full validate run ───────────────────────────────────────────

class TestRunValidate:
    def test_full_run_on_clean_data(self):
        df = make_clean_df(50)
        clean, report = run_validate(df)
        assert len(clean) == 50
        assert report.passed_overall is True
        assert report.total_rows_dropped == 0

    def test_full_run_drops_inactive_players(self):
        df = make_clean_df(10)
        df.loc[0, "MIN"] = "0:00"
        df.loc[1, "MIN"] = "0:00"
        clean, report = run_validate(df)
        assert len(clean) == 8
        assert report.total_rows_dropped == 2

    def test_full_run_aborts_on_high_drop_rate(self):
        df = make_clean_df(10)
        # Set 8 rows as inactive — 80% drop rate, exceeds threshold
        for i in range(8):
            df.loc[i, "MIN"] = "0:00"
        with pytest.raises(RuntimeError, match="Pipeline aborted"):
            run_validate(df)
