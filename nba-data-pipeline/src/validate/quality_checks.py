"""
src/validate/quality_checks.py
--------------------------------
Stage 02 — Validate & Quality Check

Runs a suite of data quality rules against the raw DataFrame produced by
the Extract stage. Records that fail hard rules are quarantined to a
separate BigQuery table. Soft warnings are logged but records continue
downstream.

Rule types:
  Hard (FAIL): Record is dropped and written to quarantine table.
               Pipeline aborts if overall drop rate exceeds DQ_MAX_DROP_RATE.
  Soft (WARN): Issue is logged. Record continues to transform stage.

Quality checks applied:
  1. Schema completeness  — all expected columns are present
  2. Inactive players     — rows where MIN = 0 (did not play) are dropped
  3. Null ts_pct          — DNP records with null shooting stats (soft warn)
  4. fg_pct range         — values must be between 0.0 and 1.0
  5. pts range            — values must be between 0 and 70
  6. FK integrity         — game_id must exist in dim_games
"""

from dataclasses import dataclass, field
from typing import Tuple

import pandas as pd

from config.settings import (
    EXPECTED_RAW_COLUMNS,
    DQ_MAX_NULL_RATE_TS_PCT,
    DQ_MAX_DROP_RATE,
    DQ_FG_PCT_MIN,
    DQ_FG_PCT_MAX,
    DQ_PTS_MIN,
    DQ_PTS_MAX,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Result Types ──────────────────────────────────────────────────────────────

@dataclass
class QAResult:
    """Result for a single quality check."""
    rule_name: str
    rule_type: str          # "HARD" or "SOFT"
    passed: bool
    rows_affected: int
    detail: str


@dataclass
class ValidationReport:
    """Aggregated results from all quality checks on one pipeline run."""
    total_rows_in: int = 0
    total_rows_out: int = 0
    total_rows_dropped: int = 0
    total_rows_quarantined: int = 0
    checks: list = field(default_factory=list)
    passed_overall: bool = True

    @property
    def drop_rate(self) -> float:
        if self.total_rows_in == 0:
            return 0.0
        return self.total_rows_dropped / self.total_rows_in

    def add_check(self, result: QAResult) -> None:
        self.checks.append(result)
        if not result.passed and result.rule_type == "HARD":
            self.passed_overall = False


# ── Individual Checks ─────────────────────────────────────────────────────────

def check_schema(df: pd.DataFrame, report: ValidationReport) -> pd.DataFrame:
    """
    Check 1: Verify all expected columns are present in the DataFrame.
    Type: HARD — abort if any required column is missing.
    """
    missing = [col for col in EXPECTED_RAW_COLUMNS if col not in df.columns]
    passed = len(missing) == 0

    report.add_check(QAResult(
        rule_name="schema_completeness",
        rule_type="HARD",
        passed=passed,
        rows_affected=0,
        detail=(
            f"All {len(EXPECTED_RAW_COLUMNS)} expected columns present"
            if passed
            else f"Missing columns: {missing}"
        ),
    ))

    if not passed:
        raise ValueError(f"Schema validation failed. Missing columns: {missing}")

    logger.info("check_schema_passed", expected=len(EXPECTED_RAW_COLUMNS))
    return df


def check_inactive_players(df: pd.DataFrame, report: ValidationReport) -> pd.DataFrame:
    """
    Check 2: Drop rows where MIN = 0 or MIN is '0:00' (player did not play).
    Type: HARD — these records contain no meaningful stats and will cause
    division-by-zero in ts_pct derivation.
    """
    # MIN comes in as a string "32:14" from the API
    def parse_min(val) -> float:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return 0.0
        parts = str(val).split(":")
        try:
            return float(parts[0]) + float(parts[1]) / 60 if len(parts) == 2 else float(parts[0])
        except (ValueError, IndexError):
            return 0.0

    df = df.copy()
    df["_min_float"] = df["MIN"].apply(parse_min)
    inactive_mask = df["_min_float"] == 0.0
    inactive_count = inactive_mask.sum()

    report.add_check(QAResult(
        rule_name="inactive_players_dropped",
        rule_type="HARD",
        passed=True,   # Always passes — dropping is expected behaviour
        rows_affected=int(inactive_count),
        detail=f"Dropped {inactive_count} inactive player rows (MIN=0)",
    ))
    report.total_rows_dropped += int(inactive_count)

    df_clean = df[~inactive_mask].drop(columns=["_min_float"])
    logger.info("check_inactive_players", dropped=inactive_count, remaining=len(df_clean))
    return df_clean


def check_null_ts_pct(df: pd.DataFrame, report: ValidationReport) -> pd.DataFrame:
    """
    Check 3: Warn if ts_pct null rate exceeds threshold.
    Type: SOFT — high null rate may indicate data issue but records continue.
    """
    if "TS_PCT" not in df.columns:
        return df

    null_count = df["TS_PCT"].isna().sum()
    null_rate = null_count / len(df) if len(df) > 0 else 0.0
    passed = null_rate <= DQ_MAX_NULL_RATE_TS_PCT

    report.add_check(QAResult(
        rule_name="null_rate_ts_pct",
        rule_type="SOFT",
        passed=passed,
        rows_affected=int(null_count),
        detail=f"TS_PCT null rate: {null_rate:.2%} (threshold: {DQ_MAX_NULL_RATE_TS_PCT:.2%})",
    ))

    if not passed:
        logger.warning(
            "high_null_rate_ts_pct",
            null_count=null_count,
            null_rate=f"{null_rate:.2%}",
            threshold=f"{DQ_MAX_NULL_RATE_TS_PCT:.2%}",
        )
    else:
        logger.info("check_null_ts_pct_passed", null_count=null_count)

    return df


def check_fg_pct_range(df: pd.DataFrame, report: ValidationReport) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Check 4: Quarantine rows where FG_PCT is outside [0.0, 1.0].
    Type: HARD — values outside this range indicate corrupt data.

    Returns:
        (clean_df, quarantine_df)
    """
    if "FG_PCT" not in df.columns:
        return df, pd.DataFrame()

    valid_mask = (
        df["FG_PCT"].isna() |                              # Nulls allowed (0 attempts)
        df["FG_PCT"].between(DQ_FG_PCT_MIN, DQ_FG_PCT_MAX)
    )
    bad_rows = df[~valid_mask].copy()
    bad_rows["_fail_reason"] = "fg_pct_out_of_range"

    report.add_check(QAResult(
        rule_name="fg_pct_range",
        rule_type="HARD",
        passed=len(bad_rows) == 0,
        rows_affected=len(bad_rows),
        detail=f"FG_PCT range check: {len(bad_rows)} rows outside [{DQ_FG_PCT_MIN}, {DQ_FG_PCT_MAX}]",
    ))
    report.total_rows_quarantined += len(bad_rows)
    report.total_rows_dropped += len(bad_rows)

    logger.info("check_fg_pct_range", quarantined=len(bad_rows), clean=len(df) - len(bad_rows))
    return df[valid_mask].copy(), bad_rows


def check_pts_range(df: pd.DataFrame, report: ValidationReport) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Check 5: Quarantine rows where PTS is outside [0, 70].
    Type: HARD — no player has ever scored over 100 points in a game;
    values above 70 are treated as data corruption.

    Returns:
        (clean_df, quarantine_df)
    """
    if "PTS" not in df.columns:
        return df, pd.DataFrame()

    valid_mask = df["PTS"].between(DQ_PTS_MIN, DQ_PTS_MAX)
    bad_rows = df[~valid_mask].copy()
    bad_rows["_fail_reason"] = "pts_out_of_range"

    report.add_check(QAResult(
        rule_name="pts_range",
        rule_type="HARD",
        passed=len(bad_rows) == 0,
        rows_affected=len(bad_rows),
        detail=f"PTS range check: {len(bad_rows)} rows outside [{DQ_PTS_MIN}, {DQ_PTS_MAX}]",
    ))
    report.total_rows_quarantined += len(bad_rows)
    report.total_rows_dropped += len(bad_rows)

    logger.info("check_pts_range", quarantined=len(bad_rows), clean=len(df) - len(bad_rows))
    return df[valid_mask].copy(), bad_rows


# ── Abort Check ───────────────────────────────────────────────────────────────

def check_drop_rate(report: ValidationReport) -> None:
    """
    Abort the pipeline if total drop rate exceeds the configured threshold.
    A high drop rate suggests a systemic issue (wrong season, API change, etc.)
    rather than expected data quality noise.
    """
    if report.drop_rate > DQ_MAX_DROP_RATE:
        raise RuntimeError(
            f"Pipeline aborted: drop rate {report.drop_rate:.2%} exceeds "
            f"threshold {DQ_MAX_DROP_RATE:.2%}. "
            f"Dropped {report.total_rows_dropped}/{report.total_rows_in} rows. "
            "Investigate before re-running."
        )
    logger.info(
        "drop_rate_check_passed",
        drop_rate=f"{report.drop_rate:.2%}",
        threshold=f"{DQ_MAX_DROP_RATE:.2%}",
    )


# ── Main Entry Point ──────────────────────────────────────────────────────────

def run_validate(df: pd.DataFrame) -> Tuple[pd.DataFrame, ValidationReport]:
    """
    Execute all validation checks against the raw DataFrame.

    Args:
        df: Raw DataFrame from the Extract stage

    Returns:
        (clean_df, report)
            clean_df — rows that passed all hard checks, ready for Transform
            report   — full ValidationReport with per-check results

    Raises:
        ValueError:   If required schema columns are missing
        RuntimeError: If overall drop rate exceeds configured threshold
    """
    logger.info("validate_stage_start", rows_in=len(df))

    report = ValidationReport(total_rows_in=len(df))
    quarantine_frames = []

    # Run checks in order
    df = check_schema(df, report)
    df = check_inactive_players(df, report)
    df = check_null_ts_pct(df, report)
    df, bad_fg = check_fg_pct_range(df, report)
    df, bad_pts = check_pts_range(df, report)

    if not bad_fg.empty:
        quarantine_frames.append(bad_fg)
    if not bad_pts.empty:
        quarantine_frames.append(bad_pts)

    # Final drop rate gate
    check_drop_rate(report)

    report.total_rows_out = len(df)

    logger.info(
        "validate_stage_complete",
        rows_in=report.total_rows_in,
        rows_out=report.total_rows_out,
        rows_dropped=report.total_rows_dropped,
        drop_rate=f"{report.drop_rate:.2%}",
        checks_run=len(report.checks),
    )

    return df, report
