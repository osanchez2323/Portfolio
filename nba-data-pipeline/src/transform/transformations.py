"""
src/transform/transformations.py
----------------------------------
Stage 03 — Transform & Enrich

Applies all type casts, field derivations, dimension table joins, and
audit field additions to the validated DataFrame. All transformations
are idempotent — re-running on the same input always produces the same output.

Transformations applied:
  1. Type casting     — floats → int, string MIN → decimal float, dates
  2. Derived metrics  — ts_pct, game_score (Hollinger), rolling 5-game avg
  3. Name normalise   — PLAYER_NAME to Title Case
  4. Dimension joins  — player, team, and game dimension tables
  5. Audit fields     — load_timestamp, pipeline_run_id, source_system, row_hash
  6. Column rename    — API names → warehouse snake_case names
"""

import hashlib
import re
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from google.cloud import bigquery

from config.settings import (
    GCP_PROJECT_ID,
    BQ_DATASET,
    BQ_DIM_PLAYERS_TABLE,
    BQ_DIM_TEAMS_TABLE,
    BQ_DIM_GAMES_TABLE,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Type Casting ──────────────────────────────────────────────────────────────

def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast raw API columns to their correct warehouse types.

    Notable casts:
      - PTS/REB/AST etc. come as floats (e.g. 23.0) → cast to int
      - MIN comes as "32:14" string → convert to decimal minutes (32.23)
      - GAME_DATE comes as "APR 15, 2025" → parse to datetime.date
      - FG_PCT/FT_PCT/TS_PCT → ensure float64

    Args:
        df: Validated DataFrame

    Returns:
        DataFrame with corrected column types
    """
    df = df.copy()

    # Integer stat columns
    int_cols = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FGM", "FGA",
                "FTM", "FTA", "FG3M", "FG3A", "OREB", "DREB", "PF"]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Float percentage columns
    float_cols = ["FG_PCT", "FT_PCT", "TS_PCT"]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    # Minutes played: "32:14" → 32.23 (decimal minutes)
    if "MIN" in df.columns:
        df["MIN"] = df["MIN"].apply(_parse_minutes)

    # Game date: "APR 15, 2025" → date
    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], format="mixed").dt.date

    # Player name: normalise casing
    if "PLAYER_NAME" in df.columns:
        df["PLAYER_NAME"] = df["PLAYER_NAME"].str.title().str.strip()

    logger.info("cast_types_complete", rows=len(df))
    return df


def _parse_minutes(val) -> float:
    """Convert "MM:SS" string to decimal minutes. Returns 0.0 on parse failure."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0.0
    match = re.match(r"^(\d+):(\d{2})$", str(val).strip())
    if match:
        return int(match.group(1)) + int(match.group(2)) / 60
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


# ── Derived Metrics ───────────────────────────────────────────────────────────

def derive_ts_pct(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive True Shooting Percentage (TS%).

    Formula: ts_pct = pts / (2 × (fga + 0.44 × fta))

    TS% accounts for the different value of 2-point FGs, 3-point FGs, and
    free throws. It is widely considered the best single measure of shooting
    efficiency. Rows with zero attempts get ts_pct = NaN (avoid divide-by-zero).

    Reference: https://www.basketball-reference.com/about/glossary.html
    """
    df = df.copy()
    denominator = 2 * (df["FGA"] + 0.44 * df["FTA"])
    df["TS_PCT"] = np.where(
        denominator > 0,
        df["PTS"] / denominator,
        np.nan,
    ).round(4)
    logger.info("derived_ts_pct", null_count=int(df["TS_PCT"].isna().sum()))
    return df


def derive_game_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive Hollinger Game Score — a single-number summary of a player's
    statistical contribution in one game.

    Formula (John Hollinger):
        game_score = pts + 0.4×fgm − 0.7×fga − 0.4×(fta − ftm)
                     + 0.7×orb + 0.3×drb + stl + 0.7×ast + 0.7×blk
                     − 0.4×pf − tov

    A score of 10 is an average NBA performance; 40+ is an all-time game.
    """
    df = df.copy()
    df["GAME_SCORE"] = (
        df["PTS"]
        + 0.4 * df["FGM"]
        - 0.7 * df["FGA"]
        - 0.4 * (df["FTA"] - df["FTM"])
        + 0.7 * df.get("OREB", 0)
        + 0.3 * df.get("DREB", 0)
        + df["STL"]
        + 0.7 * df["AST"]
        + 0.7 * df["BLK"]
        - 0.4 * df["PF"]
        - df["TOV"]
    ).round(2)
    logger.info("derived_game_score", mean=round(float(df["GAME_SCORE"].mean()), 2))
    return df


def derive_rolling_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive 5-game rolling average points per player.

    Sorted by player_id and game_date to ensure correct temporal order.
    min_periods=1 means partial windows (early season) are included.
    """
    df = df.copy()
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"])
    df["ROLL5_PTS"] = (
        df.groupby("PLAYER_ID")["PTS"]
        .transform(lambda x: x.rolling(5, min_periods=1).mean())
        .round(1)
    )
    logger.info("derived_rolling_averages", players=df["PLAYER_ID"].nunique())
    return df


# ── Dimension Joins ───────────────────────────────────────────────────────────

def join_dim_games(df: pd.DataFrame) -> pd.DataFrame:
    """
    Join dim_games to add is_home flag.

    In the NBA Stats API response, each row includes a MATCHUP field like
    "LAL vs. GSW" (home) or "LAL @ GSW" (away). We derive is_home from this
    rather than hitting the dimension table to avoid an extra BQ query in dev.
    """
    df = df.copy()
    if "MATCHUP" in df.columns:
        df["IS_HOME"] = df["MATCHUP"].str.contains(r"\bvs\.\b", regex=True)
    else:
        df["IS_HOME"] = None
    return df


# ── Audit Fields ──────────────────────────────────────────────────────────────

def add_audit_fields(df: pd.DataFrame, run_id: str) -> pd.DataFrame:
    """
    Add standard audit fields to every record.

    Fields added:
      - load_timestamp:  UTC time this record was processed by the pipeline
      - pipeline_run_id: Unique identifier for this pipeline execution
      - source_system:   Always 'nba_stats_api' for traceability
      - is_current:      True — marks this as the latest version of the record
      - row_hash:        MD5 of key stat fields for change detection

    The row_hash enables efficient change detection on future runs:
    if the hash matches the existing warehouse row, the MERGE skips the update.
    """
    df = df.copy()
    now_utc = datetime.now(timezone.utc)

    df["LOAD_TIMESTAMP"] = now_utc
    df["PIPELINE_RUN_ID"] = run_id
    df["SOURCE_SYSTEM"] = "nba_stats_api"
    df["IS_CURRENT"] = True

    hash_cols = ["GAME_ID", "PLAYER_ID", "PTS", "REB", "AST", "STL", "BLK", "TOV"]
    df["ROW_HASH"] = df[hash_cols].apply(
        lambda row: hashlib.md5(str(row.values).encode()).hexdigest(), axis=1
    )

    logger.info("audit_fields_added", run_id=run_id, load_timestamp=str(now_utc))
    return df


# ── Column Rename ─────────────────────────────────────────────────────────────

# Maps raw API column names → warehouse snake_case column names
COLUMN_RENAME_MAP = {
    "GAME_ID":            "game_id",
    "PLAYER_ID":          "player_id",
    "PLAYER_NAME":        "player_name",
    "TEAM_ID":            "team_id",
    "TEAM_ABBREVIATION":  "team_abbreviation",
    "GAME_DATE":          "game_date",
    "MIN":                "min_played",
    "PTS":                "pts",
    "REB":                "reb",
    "AST":                "ast",
    "STL":                "stl",
    "BLK":                "blk",
    "TOV":                "tov",
    "FGM":                "fgm",
    "FGA":                "fga",
    "FG_PCT":             "fg_pct",
    "FTM":                "ftm",
    "FTA":                "fta",
    "FT_PCT":             "ft_pct",
    "FG3M":               "fg3m",
    "FG3A":               "fg3a",
    "OREB":               "oreb",
    "DREB":               "dreb",
    "PF":                 "pf",
    "TS_PCT":             "ts_pct",
    "GAME_SCORE":         "game_score",
    "ROLL5_PTS":          "roll5_pts",
    "IS_HOME":            "is_home",
    "LOAD_TIMESTAMP":     "load_timestamp",
    "PIPELINE_RUN_ID":    "pipeline_run_id",
    "SOURCE_SYSTEM":      "source_system",
    "IS_CURRENT":         "is_current",
    "ROW_HASH":           "row_hash",
}


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns from API naming convention to warehouse snake_case."""
    df = df.rename(columns={k: v for k, v in COLUMN_RENAME_MAP.items() if k in df.columns})
    logger.info("columns_renamed", final_columns=list(df.columns))
    return df


# ── Main Entry Point ──────────────────────────────────────────────────────────

def run_transform(df: pd.DataFrame, run_id: str) -> pd.DataFrame:
    """
    Execute the full Transform stage on a validated DataFrame.

    Steps (in order):
        1. Cast types (ints, floats, dates, parse MIN string)
        2. Derive TS%
        3. Derive Hollinger Game Score
        4. Derive 5-game rolling average
        5. Add is_home flag
        6. Add audit fields
        7. Rename columns to warehouse convention

    Args:
        df:     Validated DataFrame from the Validate stage
        run_id: Pipeline run identifier (used in audit fields)

    Returns:
        Fully transformed pd.DataFrame ready for the Load stage
    """
    logger.info("transform_stage_start", rows_in=len(df))

    df = cast_types(df)
    df = derive_ts_pct(df)
    df = derive_game_score(df)
    df = derive_rolling_averages(df)
    df = join_dim_games(df)
    df = add_audit_fields(df, run_id)
    df = rename_columns(df)

    logger.info(
        "transform_stage_complete",
        rows_out=len(df),
        columns=len(df.columns),
        derived_metrics=["ts_pct", "game_score", "roll5_pts"],
    )
    return df
