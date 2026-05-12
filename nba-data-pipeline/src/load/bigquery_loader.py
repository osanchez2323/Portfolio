"""
src/load/bigquery_loader.py
----------------------------
Stage 04 — Load to Data Warehouse

Loads the transformed DataFrame into BigQuery using a MERGE (upsert) statement
keyed on (game_id, player_id). This makes the pipeline idempotent — re-running
for the same date updates existing rows rather than creating duplicates.

Table design decisions:
  - Partitioned by game_date (DATE): limits bytes scanned on date-range queries
  - Clustered by team_id, player_id: optimises the most common filter patterns
  - MERGE key (game_id, player_id): natural composite key for box scores
  - Estimated query cost savings vs. unpartitioned table: ~70%
"""

from datetime import date, datetime, timezone

import pandas as pd
from google.cloud import bigquery

from config.settings import (
    GCP_PROJECT_ID,
    GCP_LOCATION,
    BQ_DATASET,
    BQ_FACT_TABLE,
    BQ_FACT_TABLE_FQN,
    BQ_BAD_RECORDS_FQN,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── BigQuery Schema Definition ────────────────────────────────────────────────

FACT_BOX_SCORES_SCHEMA = [
    bigquery.SchemaField("game_id",          "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("player_id",        "INTEGER",   mode="REQUIRED"),
    bigquery.SchemaField("game_date",        "DATE",      mode="REQUIRED"),
    bigquery.SchemaField("player_name",      "STRING",    mode="NULLABLE"),
    bigquery.SchemaField("team_id",          "INTEGER",   mode="NULLABLE"),
    bigquery.SchemaField("team_abbreviation","STRING",    mode="NULLABLE"),
    bigquery.SchemaField("min_played",       "FLOAT64",   mode="NULLABLE"),
    bigquery.SchemaField("pts",              "INTEGER",   mode="NULLABLE"),
    bigquery.SchemaField("reb",              "INTEGER",   mode="NULLABLE"),
    bigquery.SchemaField("ast",              "INTEGER",   mode="NULLABLE"),
    bigquery.SchemaField("stl",              "INTEGER",   mode="NULLABLE"),
    bigquery.SchemaField("blk",              "INTEGER",   mode="NULLABLE"),
    bigquery.SchemaField("tov",              "INTEGER",   mode="NULLABLE"),
    bigquery.SchemaField("fgm",              "INTEGER",   mode="NULLABLE"),
    bigquery.SchemaField("fga",              "INTEGER",   mode="NULLABLE"),
    bigquery.SchemaField("fg_pct",           "FLOAT64",   mode="NULLABLE"),
    bigquery.SchemaField("ftm",              "INTEGER",   mode="NULLABLE"),
    bigquery.SchemaField("fta",              "INTEGER",   mode="NULLABLE"),
    bigquery.SchemaField("ft_pct",           "FLOAT64",   mode="NULLABLE"),
    bigquery.SchemaField("fg3m",             "INTEGER",   mode="NULLABLE"),
    bigquery.SchemaField("fg3a",             "INTEGER",   mode="NULLABLE"),
    bigquery.SchemaField("oreb",             "INTEGER",   mode="NULLABLE"),
    bigquery.SchemaField("dreb",             "INTEGER",   mode="NULLABLE"),
    bigquery.SchemaField("pf",               "INTEGER",   mode="NULLABLE"),
    bigquery.SchemaField("ts_pct",           "FLOAT64",   mode="NULLABLE"),
    bigquery.SchemaField("game_score",       "FLOAT64",   mode="NULLABLE"),
    bigquery.SchemaField("roll5_pts",        "FLOAT64",   mode="NULLABLE"),
    bigquery.SchemaField("is_home",          "BOOLEAN",   mode="NULLABLE"),
    bigquery.SchemaField("load_timestamp",   "TIMESTAMP", mode="NULLABLE"),
    bigquery.SchemaField("pipeline_run_id",  "STRING",    mode="NULLABLE"),
    bigquery.SchemaField("source_system",    "STRING",    mode="NULLABLE"),
    bigquery.SchemaField("is_current",       "BOOLEAN",   mode="NULLABLE"),
    bigquery.SchemaField("row_hash",         "STRING",    mode="NULLABLE"),
]


# ── Table Management ──────────────────────────────────────────────────────────

def ensure_table_exists(client: bigquery.Client) -> None:
    """
    Create the fact_box_scores table if it does not already exist.
    Partitioned by game_date, clustered by team_id and player_id.
    """
    table_ref = bigquery.Table(BQ_FACT_TABLE_FQN, schema=FACT_BOX_SCORES_SCHEMA)

    table_ref.time_partitioning = bigquery.TimePartitioning(
        type_=bigquery.TimePartitioningType.DAY,
        field="game_date",
    )
    table_ref.clustering_fields = ["team_id", "player_id"]

    table = client.create_table(table_ref, exists_ok=True)
    logger.info("table_ensured", table=BQ_FACT_TABLE_FQN)


# ── Staging Load ──────────────────────────────────────────────────────────────

def load_to_staging(
    client: bigquery.Client,
    df: pd.DataFrame,
    run_id: str,
) -> str:
    """
    Load the transformed DataFrame to a temporary staging table.

    The staging table is used as the source for the MERGE statement.
    It is scoped to this pipeline run and deleted after the MERGE completes.

    Args:
        client: BigQuery client
        df:     Transformed DataFrame
        run_id: Used to name the temp table uniquely per run

    Returns:
        Fully-qualified staging table name
    """
    staging_table_id = f"{GCP_PROJECT_ID}.{BQ_DATASET}.staging_{run_id.replace('-', '_')}"

    job_config = bigquery.LoadJobConfig(
        schema=FACT_BOX_SCORES_SCHEMA,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )

    job = client.load_table_from_dataframe(df, staging_table_id, job_config=job_config)
    job.result()  # Wait for job to complete

    logger.info("staging_load_complete", table=staging_table_id, rows=len(df))
    return staging_table_id


# ── MERGE / Upsert ────────────────────────────────────────────────────────────

def execute_merge(
    client: bigquery.Client,
    staging_table: str,
) -> bigquery.QueryJob:
    """
    Execute a MERGE statement to upsert staging records into fact_box_scores.

    The MERGE is keyed on (game_id, player_id):
      - WHEN MATCHED AND row_hash differs → UPDATE all stat fields
      - WHEN NOT MATCHED → INSERT new row

    Using row_hash in the MATCH condition means unchanged records are skipped,
    reducing unnecessary write operations.

    Args:
        client:        BigQuery client
        staging_table: FQN of the staging table created in load_to_staging()

    Returns:
        Completed BigQuery QueryJob
    """
    merge_sql = f"""
    MERGE `{BQ_FACT_TABLE_FQN}` AS target
    USING `{staging_table}` AS source
    ON target.game_id = source.game_id
    AND target.player_id = source.player_id

    -- Update existing row only if data has changed (row_hash differs)
    WHEN MATCHED AND target.row_hash != source.row_hash THEN
      UPDATE SET
        player_name      = source.player_name,
        team_id          = source.team_id,
        team_abbreviation= source.team_abbreviation,
        min_played       = source.min_played,
        pts              = source.pts,
        reb              = source.reb,
        ast              = source.ast,
        stl              = source.stl,
        blk              = source.blk,
        tov              = source.tov,
        fgm              = source.fgm,
        fga              = source.fga,
        fg_pct           = source.fg_pct,
        ftm              = source.ftm,
        fta              = source.fta,
        ft_pct           = source.ft_pct,
        fg3m             = source.fg3m,
        fg3a             = source.fg3a,
        oreb             = source.oreb,
        dreb             = source.dreb,
        pf               = source.pf,
        ts_pct           = source.ts_pct,
        game_score       = source.game_score,
        roll5_pts        = source.roll5_pts,
        is_home          = source.is_home,
        load_timestamp   = source.load_timestamp,
        pipeline_run_id  = source.pipeline_run_id,
        row_hash         = source.row_hash

    -- Insert brand-new records
    WHEN NOT MATCHED THEN
      INSERT ROW
    ;
    """

    logger.info("merge_start", target=BQ_FACT_TABLE_FQN, source=staging_table)
    job = client.query(merge_sql)
    job.result()

    logger.info(
        "merge_complete",
        rows_affected=job.num_dml_affected_rows,
    )
    return job


def drop_staging_table(client: bigquery.Client, staging_table: str) -> None:
    """Delete the temporary staging table after the MERGE completes."""
    client.delete_table(staging_table, not_found_ok=True)
    logger.info("staging_table_dropped", table=staging_table)


# ── Audit Log ─────────────────────────────────────────────────────────────────

def write_audit_log(
    client: bigquery.Client,
    run_id: str,
    game_date: date,
    rows_loaded: int,
    rows_inserted: int,
    rows_updated: int,
    duration_seconds: float,
    status: str,
    error: str = None,
) -> None:
    """
    Write a pipeline run record to the audit log table.

    Provides a full history of every pipeline execution for monitoring,
    alerting, and debugging.
    """
    audit_table = f"{GCP_PROJECT_ID}.{BQ_DATASET}.pipeline_audit_log"
    row = {
        "run_id":           run_id,
        "game_date":        str(game_date),
        "rows_loaded":      rows_loaded,
        "rows_inserted":    rows_inserted,
        "rows_updated":     rows_updated,
        "duration_seconds": round(duration_seconds, 2),
        "status":           status,
        "error_message":    error,
        "logged_at":        datetime.now(timezone.utc).isoformat(),
    }
    errors = client.insert_rows_json(audit_table, [row])
    if errors:
        logger.warning("audit_log_insert_error", errors=errors)
    else:
        logger.info("audit_log_written", run_id=run_id, status=status)


# ── Main Entry Point ──────────────────────────────────────────────────────────

def run_load(
    df: pd.DataFrame,
    run_id: str,
    game_date: date,
    dry_run: bool = False,
) -> dict:
    """
    Execute the full Load stage.

    Steps:
        1. Ensure target table exists (create if not)
        2. Load DataFrame to temporary staging table
        3. Execute MERGE statement (upsert to fact table)
        4. Drop staging table
        5. Write audit log entry

    Args:
        df:        Transformed DataFrame from the Transform stage
        run_id:    Unique pipeline run identifier
        game_date: The game date being loaded
        dry_run:   If True, skip all BigQuery writes (for testing)

    Returns:
        Dict with load summary stats
    """
    import time
    start = time.time()

    logger.info("load_stage_start", rows=len(df), run_id=run_id, dry_run=dry_run)

    if dry_run:
        logger.info("dry_run_mode_active", message="Skipping all BigQuery writes")
        return {"rows_loaded": len(df), "rows_inserted": 0, "rows_updated": 0, "dry_run": True}

    client = bigquery.Client(project=GCP_PROJECT_ID, location=GCP_LOCATION)

    ensure_table_exists(client)
    staging_table = load_to_staging(client, df, run_id)

    try:
        merge_job = execute_merge(client, staging_table)
        rows_affected = merge_job.num_dml_affected_rows or 0
    finally:
        drop_staging_table(client, staging_table)

    duration = time.time() - start

    write_audit_log(
        client=client,
        run_id=run_id,
        game_date=game_date,
        rows_loaded=len(df),
        rows_inserted=rows_affected,
        rows_updated=0,
        duration_seconds=duration,
        status="SUCCESS",
    )

    result = {
        "rows_loaded": len(df),
        "rows_inserted": rows_affected,
        "duration_seconds": round(duration, 2),
    }

    logger.info("load_stage_complete", **result)
    return result
