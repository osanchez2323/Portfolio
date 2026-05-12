-- sql/create_fact_box_scores.sql
-- ============================================================
-- DDL for the main fact table: nba_dw.fact_box_scores
--
-- Design decisions:
--   - Partitioned by game_date (DATE): limits bytes scanned on
--     date-range queries and reduces query costs by ~70% vs.
--     unpartitioned tables for typical access patterns.
--   - Clustered by (team_id, player_id): the two most common
--     filter columns in downstream analytical queries.
--   - MERGE key (game_id, player_id): natural composite PK for
--     box scores — each player has exactly one record per game.
--   - row_hash: MD5 of key stat fields enables efficient change
--     detection in the MERGE; unchanged records are skipped.
-- ============================================================

CREATE TABLE IF NOT EXISTS `{project}.nba_dw.fact_box_scores`
(
    -- ── Primary key components ──────────────────────────────
    game_id             STRING    NOT NULL  OPTIONS(description="NBA game identifier (e.g. 0022401012)"),
    player_id           INT64     NOT NULL  OPTIONS(description="NBA player identifier"),

    -- ── Partition column ────────────────────────────────────
    game_date           DATE      NOT NULL  OPTIONS(description="Date the game was played — partition key"),

    -- ── Player / team dimensions ────────────────────────────
    player_name         STRING              OPTIONS(description="Full player name, title-cased"),
    team_id             INT64               OPTIONS(description="NBA team identifier"),
    team_abbreviation   STRING              OPTIONS(description="3-letter team code, e.g. LAL"),

    -- ── Playing time ────────────────────────────────────────
    min_played          FLOAT64             OPTIONS(description="Minutes played (decimal, e.g. 32.23)"),

    -- ── Box score stats ─────────────────────────────────────
    pts                 INT64               OPTIONS(description="Points scored"),
    reb                 INT64               OPTIONS(description="Total rebounds"),
    ast                 INT64               OPTIONS(description="Assists"),
    stl                 INT64               OPTIONS(description="Steals"),
    blk                 INT64               OPTIONS(description="Blocks"),
    tov                 INT64               OPTIONS(description="Turnovers"),

    fgm                 INT64               OPTIONS(description="Field goals made"),
    fga                 INT64               OPTIONS(description="Field goals attempted"),
    fg_pct              FLOAT64             OPTIONS(description="Field goal percentage (0.0–1.0)"),

    ftm                 INT64               OPTIONS(description="Free throws made"),
    fta                 INT64               OPTIONS(description="Free throws attempted"),
    ft_pct              FLOAT64             OPTIONS(description="Free throw percentage (0.0–1.0)"),

    fg3m                INT64               OPTIONS(description="3-point field goals made"),
    fg3a                INT64               OPTIONS(description="3-point field goals attempted"),

    oreb                INT64               OPTIONS(description="Offensive rebounds"),
    dreb                INT64               OPTIONS(description="Defensive rebounds"),
    pf                  INT64               OPTIONS(description="Personal fouls"),

    -- ── Derived metrics (computed in Transform stage) ───────
    ts_pct              FLOAT64             OPTIONS(description="True Shooting %: pts / (2 × (fga + 0.44 × fta))"),
    game_score          FLOAT64             OPTIONS(description="Hollinger Game Score — single-number performance summary"),
    roll5_pts           FLOAT64             OPTIONS(description="5-game rolling average points per player"),

    -- ── Context ─────────────────────────────────────────────
    is_home             BOOL                OPTIONS(description="True if player's team was the home team"),

    -- ── Audit fields (added in Transform stage) ─────────────
    load_timestamp      TIMESTAMP           OPTIONS(description="UTC timestamp when this record was loaded by the pipeline"),
    pipeline_run_id     STRING              OPTIONS(description="Unique identifier for the pipeline run that wrote this record"),
    source_system       STRING              OPTIONS(description="Source system identifier — always 'nba_stats_api'"),
    is_current          BOOL                OPTIONS(description="True — marks this as the active version of the record"),
    row_hash            STRING              OPTIONS(description="MD5 hash of key stat fields for change detection in MERGE")
)
PARTITION BY game_date
CLUSTER BY team_id, player_id
OPTIONS (
    description = "NBA box score fact table — one row per player per game. Partitioned by game_date, clustered by team_id and player_id.",
    require_partition_filter = FALSE
);
