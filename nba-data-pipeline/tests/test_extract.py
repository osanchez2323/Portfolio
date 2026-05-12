"""
tests/test_extract.py
----------------------
Unit tests for the Extract stage (api_client.py).
Uses the `responses` library to mock HTTP calls — no real API calls are made.
"""

import pytest
import json
from datetime import date
from unittest.mock import patch, MagicMock

import pandas as pd
import responses as rsps

from src.extract.api_client import (
    NBAApiClient,
    parse_response_to_dataframe,
)

from config.settings import NBA_API_BASE_URL


# ── Fixtures ──────────────────────────────────────────────────────────────────

MOCK_HEADERS = [
    "GAME_ID", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION",
    "GAME_DATE", "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV",
    "FGM", "FGA", "FG_PCT", "FTM", "FTA", "FT_PCT",
    "FG3M", "FG3A", "OREB", "DREB", "PF", "TS_PCT",
]

MOCK_ROWS = [
    ["0022401001", 2544, "LEBRON JAMES", 1610612747, "LAL",
     "APR 15, 2025", "34:15", 26.0, 8.0, 10.0, 1.0, 1.0, 3.0,
     10.0, 18.0, 0.556, 6.0, 8.0, 0.750, 0.0, 2.0, 1.0, 7.0, 2.0, 0.601],
    ["0022401001", 203507, "GIANNIS ANTETOKOUNMPO", 1610612749, "MIL",
     "APR 15, 2025", "36:45", 33.0, 14.0, 7.0, 2.0, 2.0, 3.0,
     13.0, 22.0, 0.591, 7.0, 10.0, 0.700, 0.0, 1.0, 4.0, 10.0, 3.0, 0.627],
]

MOCK_API_RESPONSE = {
    "resultSets": [
        {"headers": MOCK_HEADERS, "rowSet": MOCK_ROWS}
    ]
}


# ── parse_response_to_dataframe ───────────────────────────────────────────────

class TestParseResponseToDataframe:
    def test_returns_correct_row_count(self):
        raw = {"headers": MOCK_HEADERS, "rows": MOCK_ROWS}
        df = parse_response_to_dataframe(raw)
        assert len(df) == 2

    def test_returns_correct_columns(self):
        raw = {"headers": MOCK_HEADERS, "rows": MOCK_ROWS}
        df = parse_response_to_dataframe(raw)
        assert "GAME_ID" in df.columns
        assert "PTS" in df.columns

    def test_columns_normalised_to_uppercase(self):
        headers = [h.lower() for h in MOCK_HEADERS]  # Simulate lowercase from API
        raw = {"headers": headers, "rows": MOCK_ROWS}
        df = parse_response_to_dataframe(raw)
        assert all(col == col.upper() for col in df.columns)

    def test_returns_empty_df_on_no_rows(self):
        raw = {"headers": MOCK_HEADERS, "rows": []}
        df = parse_response_to_dataframe(raw)
        assert len(df) == 0
        assert list(df.columns) == MOCK_HEADERS

    def test_player_names_preserved(self):
        raw = {"headers": MOCK_HEADERS, "rows": MOCK_ROWS}
        df = parse_response_to_dataframe(raw)
        assert "LEBRON JAMES" in df["PLAYER_NAME"].values


# ── NBAApiClient ──────────────────────────────────────────────────────────────

class TestNBAApiClient:
    @rsps.activate
    def test_fetch_box_scores_parses_response(self):
        """Mock the API endpoint and verify fetch_box_scores returns expected rows."""
        rsps.add(
            rsps.GET,
            f"{NBA_API_BASE_URL}/leaguegamelog",
            json=MOCK_API_RESPONSE,
            status=200,
        )

        client = NBAApiClient()
        result = client.fetch_box_scores(date(2025, 4, 15))

        assert result["rows"] == MOCK_ROWS
        assert result["headers"] == MOCK_HEADERS
        assert result["game_date"] == "2025-04-15"

    @rsps.activate
    def test_raises_on_http_error(self):
        """Client should raise after max retries on 403."""
        rsps.add(
            rsps.GET,
            f"{NBA_API_BASE_URL}/leaguegamelog",
            status=403,
        )

        client = NBAApiClient()
        with pytest.raises(Exception):
            client.fetch_box_scores(date(2025, 4, 15))

    @rsps.activate
    def test_retries_on_connection_error(self):
        """Client should retry on ConnectionError."""
        rsps.add(rsps.GET, f"{NBA_API_BASE_URL}/leaguegamelog", body=ConnectionError("timeout"))
        rsps.add(rsps.GET, f"{NBA_API_BASE_URL}/leaguegamelog", body=ConnectionError("timeout"))
        rsps.add(rsps.GET, f"{NBA_API_BASE_URL}/leaguegamelog", json=MOCK_API_RESPONSE, status=200)

        client = NBAApiClient()
        result = client.fetch_box_scores(date(2025, 4, 15))
        assert result["rows"] == MOCK_ROWS
