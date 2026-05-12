"""
config/settings.py
------------------
Centralised configuration for the NBA Data Pipeline.
All settings are loaded from environment variables (via .env).
Import this module wherever config values are needed.
"""

import os
from dotenv import load_dotenv

load_dotenv()


# ── Google Cloud ──────────────────────────────────────────────────────────────
GCP_PROJECT_ID: str = os.getenv("GCP_PROJECT_ID", "")
GCP_LOCATION: str = os.getenv("GCP_LOCATION", "US")

# ── BigQuery ──────────────────────────────────────────────────────────────────
BQ_DATASET: str = os.getenv("BQ_DATASET", "nba_dw")
BQ_FACT_TABLE: str = os.getenv("BQ_FACT_TABLE", "fact_box_scores")
BQ_DIM_PLAYERS_TABLE: str = os.getenv("BQ_DIM_PLAYERS_TABLE", "dim_players")
BQ_DIM_TEAMS_TABLE: str = os.getenv("BQ_DIM_TEAMS_TABLE", "dim_teams")
BQ_DIM_GAMES_TABLE: str = os.getenv("BQ_DIM_GAMES_TABLE", "dim_games")
BQ_BAD_RECORDS_TABLE: str = os.getenv("BQ_BAD_RECORDS_TABLE", "quarantine_bad_records")

# Fully-qualified table references
BQ_FACT_TABLE_FQN: str = f"{GCP_PROJECT_ID}.{BQ_DATASET}.{BQ_FACT_TABLE}"
BQ_BAD_RECORDS_FQN: str = f"{GCP_PROJECT_ID}.{BQ_DATASET}.{BQ_BAD_RECORDS_TABLE}"

# ── Google Cloud Storage ──────────────────────────────────────────────────────
GCS_BUCKET: str = os.getenv("GCS_BUCKET", "")
GCS_RAW_PREFIX: str = os.getenv("GCS_RAW_PREFIX", "nba/raw/box_scores")

# ── NBA Stats API ─────────────────────────────────────────────────────────────
NBA_API_BASE_URL: str = os.getenv("NBA_API_BASE_URL", "https://stats.nba.com/stats")
NBA_SEASON: str = os.getenv("NBA_SEASON", "2024-25")
NBA_SEASON_TYPE: str = os.getenv("NBA_SEASON_TYPE", "Regular Season")
NBA_API_TIMEOUT: int = int(os.getenv("NBA_API_TIMEOUT", "30"))
NBA_API_MAX_RETRIES: int = int(os.getenv("NBA_API_MAX_RETRIES", "3"))

# Headers required to avoid 403 from stats.nba.com
NBA_API_HEADERS: dict = {
    "Host": "stats.nba.com",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
    "Referer": "https://www.nba.com/",
    "Connection": "keep-alive",
}

# ── Pipeline settings ─────────────────────────────────────────────────────────
PIPELINE_LOOKBACK_DAYS: int = int(os.getenv("PIPELINE_LOOKBACK_DAYS", "1"))
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# Expected columns in raw API response (used for schema validation)
EXPECTED_RAW_COLUMNS: list = [
    "GAME_ID",
    "PLAYER_ID",
    "PLAYER_NAME",
    "TEAM_ID",
    "TEAM_ABBREVIATION",
    "GAME_DATE",
    "MIN",
    "PTS",
    "REB",
    "AST",
    "STL",
    "BLK",
    "TOV",
    "FGM",
    "FGA",
    "FG_PCT",
    "FTM",
    "FTA",
    "FT_PCT",
    "FG3M",
    "FG3A",
    "OREB",
    "DREB",
    "PF",
]

# Data quality thresholds
DQ_MAX_NULL_RATE_TS_PCT: float = 0.02        # 2% — soft warning
DQ_MAX_DROP_RATE: float = 0.05               # 5% — abort if more than 5% dropped
DQ_FG_PCT_MIN: float = 0.0
DQ_FG_PCT_MAX: float = 1.0
DQ_PTS_MIN: int = 0
DQ_PTS_MAX: int = 70
