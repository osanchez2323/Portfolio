"""
src/extract/api_client.py
--------------------------
Stage 01 — Extract & Ingest

Pulls NBA box score data from the NBA Stats API for a given game date,
parses the JSON response into a pandas DataFrame, and writes the raw
records to Google Cloud Storage as a date-partitioned Parquet file.

Key engineering decisions:
  - Browser headers are required; stats.nba.com returns 403 without them.
  - Exponential backoff retry handles rate limiting and transient errors.
  - Raw JSON is always preserved in GCS before any transformation —
    immutable source-of-truth that allows full re-processing from scratch.
  - Parquet chosen over CSV: ~60% smaller, preserves dtypes, faster BQ load.
"""

import json
import time
from datetime import date, datetime
from typing import Optional

import pandas as pd
import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from config.settings import (
    NBA_API_BASE_URL,
    NBA_API_HEADERS,
    NBA_API_MAX_RETRIES,
    NBA_API_TIMEOUT,
    NBA_SEASON,
    NBA_SEASON_TYPE,
    GCS_BUCKET,
    GCS_RAW_PREFIX,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── API Client ────────────────────────────────────────────────────────────────

class NBAApiClient:
    """
    Client for the NBA Stats REST API.

    Wraps the /boxscores/v2 endpoint with retry logic, header management,
    and response parsing.
    """

    BASE_URL = NBA_API_BASE_URL

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(NBA_API_HEADERS)

    @retry(
        retry=retry_if_exception_type((requests.HTTPError, requests.ConnectionError)),
        stop=stop_after_attempt(NBA_API_MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def _get(self, endpoint: str, params: dict) -> dict:
        """
        Make a GET request to the NBA Stats API with retry on failure.

        Args:
            endpoint: API path (e.g. '/leaguegamelog')
            params:   Query parameters dict

        Returns:
            Parsed JSON response as dict

        Raises:
            requests.HTTPError: On 4xx/5xx after all retries exhausted
        """
        url = f"{self.BASE_URL}{endpoint}"
        logger.info("api_request", url=url, params=params)

        response = self.session.get(url, params=params, timeout=NBA_API_TIMEOUT)
        response.raise_for_status()

        # Respect rate limiting — brief pause between requests
        time.sleep(0.6)
        return response.json()

    def fetch_box_scores(self, game_date: date) -> dict:
        """
        Fetch all player box score records for a given game date.

        Args:
            game_date: The date to pull (pulls all games played that day)

        Returns:
            Raw JSON response dict from the API
        """
        params = {
            "LeagueID": "00",
            "Season": NBA_SEASON,
            "SeasonType": NBA_SEASON_TYPE,
            "DateFrom": game_date.strftime("%m/%d/%Y"),
            "DateTo": game_date.strftime("%m/%d/%Y"),
        }

        logger.info("fetching_box_scores", game_date=str(game_date), season=NBA_SEASON)
        raw = self._get("/leaguegamelog", params)

        result_set = raw.get("resultSets", [{}])[0]
        headers = result_set.get("headers", [])
        rows = result_set.get("rowSet", [])

        logger.info(
            "api_response_received",
            game_date=str(game_date),
            row_count=len(rows),
            column_count=len(headers),
        )

        return {"headers": headers, "rows": rows, "game_date": str(game_date)}


# ── DataFrame Builder ─────────────────────────────────────────────────────────

def parse_response_to_dataframe(raw_response: dict) -> pd.DataFrame:
    """
    Flatten a raw NBA Stats API response into a pandas DataFrame.

    The API returns headers and rows separately:
        {"headers": ["GAME_ID", "PLAYER_ID", ...], "rows": [[...], [...]]}

    Args:
        raw_response: Dict with 'headers' and 'rows' keys

    Returns:
        pd.DataFrame with one row per player-game record
    """
    headers = raw_response["headers"]
    rows = raw_response["rows"]

    if not rows:
        logger.warning("empty_response", message="API returned 0 rows for this date")
        return pd.DataFrame(columns=headers)

    df = pd.DataFrame(rows, columns=headers)
    df.columns = [col.upper() for col in df.columns]  # Normalise to uppercase

    logger.info(
        "dataframe_built",
        rows=len(df),
        columns=list(df.columns),
    )
    return df


# ── GCS Writer ────────────────────────────────────────────────────────────────

def write_raw_to_gcs(df: pd.DataFrame, game_date: date, run_id: str) -> str:
    """
    Write the raw DataFrame to GCS as a date-partitioned Parquet file.

    Path pattern:
        gs://{bucket}/{prefix}/date={game_date}/run_{run_id}.parquet

    Preserving the raw file before any transformation means we always have
    a source-of-truth and can fully re-process the data without re-hitting
    the API.

    Args:
        df:        Raw DataFrame from the API
        game_date: The game date (used for partition path)
        run_id:    Unique pipeline run identifier for the filename

    Returns:
        GCS URI of the written file
    """
    from google.cloud import storage

    gcs_path = (
        f"{GCS_RAW_PREFIX}/date={game_date.strftime('%Y-%m-%d')}/run_{run_id}.parquet"
    )
    gcs_uri = f"gs://{GCS_BUCKET}/{gcs_path}"

    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(gcs_path)

    parquet_bytes = df.to_parquet(index=False)
    blob.upload_from_string(parquet_bytes, content_type="application/octet-stream")

    logger.info(
        "raw_written_to_gcs",
        uri=gcs_uri,
        rows=len(df),
        size_bytes=len(parquet_bytes),
    )
    return gcs_uri


# ── Main Entry Point ──────────────────────────────────────────────────────────

def run_extract(game_date: date, run_id: str) -> pd.DataFrame:
    """
    Execute the full Extract stage for a given game date.

    Steps:
        1. Hit the NBA Stats API for the given date
        2. Parse JSON response into a DataFrame
        3. Write raw Parquet to GCS as immutable source-of-truth
        4. Return the DataFrame for downstream validation

    Args:
        game_date: Date to extract (pulls all games played that day)
        run_id:    Unique run identifier for file naming and audit trail

    Returns:
        Raw pd.DataFrame ready for the Validate stage
    """
    logger.info("extract_stage_start", game_date=str(game_date), run_id=run_id)

    client = NBAApiClient()
    raw_response = client.fetch_box_scores(game_date)
    df = parse_response_to_dataframe(raw_response)

    if not df.empty:
        write_raw_to_gcs(df, game_date, run_id)

    logger.info(
        "extract_stage_complete",
        game_date=str(game_date),
        rows_extracted=len(df),
    )
    return df
