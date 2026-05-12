"""
src/main.py
-----------
Main entry point for the NBA Data Pipeline.

Orchestrates all four ETL stages in sequence:
  01. Extract  — pull from NBA Stats API, write raw to GCS
  02. Validate — enforce quality rules, quarantine bad records
  03. Transform — type cast, derive metrics, add audit fields
  04. Load     — upsert to BigQuery via MERGE statement

Usage:
    python -m src.main                         # Yesterday's games
    python -m src.main --date 2025-04-15       # Specific date
    python -m src.main --stage extract         # Single stage only
    python -m src.main --dry-run               # No BQ writes
"""

import argparse
import sys
import time
import uuid
from datetime import date, datetime, timedelta

from src.extract.api_client import run_extract
from src.validate.quality_checks import run_validate
from src.transform.transformations import run_transform
from src.load.bigquery_loader import run_load
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NBA Data Pipeline")
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Game date to process in YYYY-MM-DD format (default: yesterday)",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["extract", "validate", "transform", "load", "all"],
        default="all",
        help="Pipeline stage to run (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Skip BigQuery writes (useful for testing transformations)",
    )
    return parser.parse_args()


def resolve_game_date(date_str: str | None) -> date:
    """Parse --date arg or default to yesterday."""
    if date_str:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    return date.today() - timedelta(days=1)


def run_pipeline(game_date: date, stage: str = "all", dry_run: bool = False) -> dict:
    """
    Execute the pipeline for a given game date.

    Args:
        game_date: Date to process
        stage:     Which stage(s) to run ('all' runs the full pipeline)
        dry_run:   If True, skip BigQuery writes

    Returns:
        Summary dict with row counts and timing per stage
    """
    run_id = str(uuid.uuid4())[:8].upper()
    pipeline_start = time.time()

    logger.info(
        "pipeline_start",
        run_id=run_id,
        game_date=str(game_date),
        stage=stage,
        dry_run=dry_run,
    )

    summary = {"run_id": run_id, "game_date": str(game_date), "stages": {}}

    try:
        # ── Stage 01: Extract ──────────────────────────────────────────────
        if stage in ("extract", "all"):
            t0 = time.time()
            raw_df = run_extract(game_date, run_id)
            summary["stages"]["extract"] = {
                "rows": len(raw_df),
                "duration_s": round(time.time() - t0, 2),
                "status": "SUCCESS",
            }
            logger.info("stage_complete", stage="extract", **summary["stages"]["extract"])

            if stage == "extract":
                return summary

        # ── Stage 02: Validate ─────────────────────────────────────────────
        if stage in ("validate", "all"):
            t0 = time.time()
            clean_df, report = run_validate(raw_df)
            summary["stages"]["validate"] = {
                "rows_in":      report.total_rows_in,
                "rows_out":     report.total_rows_out,
                "rows_dropped": report.total_rows_dropped,
                "drop_rate":    f"{report.drop_rate:.2%}",
                "duration_s":   round(time.time() - t0, 2),
                "status":       "SUCCESS" if report.passed_overall else "WARNING",
            }
            logger.info("stage_complete", stage="validate", **summary["stages"]["validate"])

            if stage == "validate":
                return summary

        # ── Stage 03: Transform ────────────────────────────────────────────
        if stage in ("transform", "all"):
            t0 = time.time()
            transformed_df = run_transform(clean_df, run_id)
            summary["stages"]["transform"] = {
                "rows":       len(transformed_df),
                "columns":    len(transformed_df.columns),
                "duration_s": round(time.time() - t0, 2),
                "status":     "SUCCESS",
            }
            logger.info("stage_complete", stage="transform", **summary["stages"]["transform"])

            if stage == "transform":
                return summary

        # ── Stage 04: Load ─────────────────────────────────────────────────
        if stage in ("load", "all"):
            t0 = time.time()
            load_result = run_load(transformed_df, run_id, game_date, dry_run=dry_run)
            summary["stages"]["load"] = {
                **load_result,
                "duration_s": round(time.time() - t0, 2),
                "status":     "SUCCESS",
            }
            logger.info("stage_complete", stage="load", **summary["stages"]["load"])

    except Exception as exc:
        logger.error(
            "pipeline_failed",
            run_id=run_id,
            game_date=str(game_date),
            error=str(exc),
            exc_info=True,
        )
        summary["status"] = "FAILED"
        summary["error"] = str(exc)
        raise

    total_duration = round(time.time() - pipeline_start, 2)
    summary["status"] = "SUCCESS"
    summary["total_duration_s"] = total_duration

    logger.info(
        "pipeline_complete",
        run_id=run_id,
        game_date=str(game_date),
        total_duration_s=total_duration,
        status="SUCCESS",
    )
    return summary


if __name__ == "__main__":
    args = parse_args()
    game_date = resolve_game_date(args.date)

    result = run_pipeline(
        game_date=game_date,
        stage=args.stage,
        dry_run=args.dry_run,
    )

    print("\n── Pipeline Summary ──────────────────────────────────────────")
    for stage_name, stats in result.get("stages", {}).items():
        print(f"  {stage_name.upper():12s}: {stats}")
    print(f"  STATUS      : {result.get('status')}")
    print(f"  TOTAL TIME  : {result.get('total_duration_s', '—')}s")
    print("──────────────────────────────────────────────────────────────\n")

    sys.exit(0 if result.get("status") == "SUCCESS" else 1)
