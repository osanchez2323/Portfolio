"""
airflow/dags/nba_pipeline_dag.py
----------------------------------
Apache Airflow DAG for the NBA Data Pipeline.

Schedule: Daily at 6:00 AM ET (11:00 UTC) — after night games have ended
          and scores are finalised on the NBA Stats API.

DAG tasks (in order):
  1. check_api_health    — Verify the NBA Stats API is reachable
  2. run_extract         — Pull yesterday's box scores from the API
  3. run_validate        — Apply quality checks, quarantine bad records
  4. run_transform       — Type cast, derive metrics, add audit fields
  5. run_load            — MERGE upsert into BigQuery
  6. notify_success      — Post success summary to Slack
  7. notify_failure      — Post failure alert to Slack (on error path)

Retry policy: 3 retries with 5-minute delays on transient errors.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator

from src.extract.api_client import run_extract
from src.validate.quality_checks import run_validate
from src.transform.transformations import run_transform
from src.load.bigquery_loader import run_load


# ── Default args ──────────────────────────────────────────────────────────────

DEFAULT_ARGS = {
    "owner": "oscar.sanchez",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
}


# ── Task functions ────────────────────────────────────────────────────────────

def task_check_api_health(**context) -> None:
    """Verify the NBA Stats API is reachable before starting the pipeline."""
    import requests
    from config.settings import NBA_API_BASE_URL, NBA_API_HEADERS, NBA_API_TIMEOUT

    try:
        response = requests.get(
            f"{NBA_API_BASE_URL}/commonallplayers",
            headers=NBA_API_HEADERS,
            params={"LeagueID": "00", "Season": "2024-25", "IsOnlyCurrentSeason": "1"},
            timeout=NBA_API_TIMEOUT,
        )
        response.raise_for_status()
        context["ti"].xcom_push(key="api_health", value="OK")
    except Exception as e:
        raise RuntimeError(f"NBA Stats API health check failed: {e}")


def task_run_extract(**context) -> dict:
    """Extract stage: pull box scores for the execution date."""
    from datetime import date
    import uuid

    execution_date = context["ds"]                      # e.g. "2025-04-15"
    game_date = date.fromisoformat(execution_date)
    run_id = context["run_id"][:8].upper()

    raw_df = run_extract(game_date, run_id)

    # Pass row count to next tasks via XCom
    context["ti"].xcom_push(key="rows_extracted", value=len(raw_df))

    # Serialise DataFrame for downstream tasks via XCom (small DataFrames only)
    # For larger datasets, write to GCS and pass the URI instead
    return {"rows_extracted": len(raw_df), "run_id": run_id}


def task_run_validate(**context) -> dict:
    """Validate stage: apply quality rules to raw DataFrame."""
    # In production: read from GCS. Here we re-pull for simplicity.
    from datetime import date
    from src.extract.api_client import run_extract

    execution_date = context["ds"]
    game_date = date.fromisoformat(execution_date)
    run_id = context["ti"].xcom_pull(task_ids="extract", key="run_id") or "UNKNOWN"

    raw_df = run_extract(game_date, run_id)
    clean_df, report = run_validate(raw_df)

    context["ti"].xcom_push(key="rows_valid", value=report.total_rows_out)
    context["ti"].xcom_push(key="rows_dropped", value=report.total_rows_dropped)
    context["ti"].xcom_push(key="drop_rate", value=f"{report.drop_rate:.2%}")

    return {
        "rows_in": report.total_rows_in,
        "rows_out": report.total_rows_out,
        "rows_dropped": report.total_rows_dropped,
    }


def task_run_transform(**context) -> dict:
    """Transform stage: type cast, derive metrics, add audit fields."""
    from datetime import date
    from src.extract.api_client import run_extract
    from src.validate.quality_checks import run_validate

    execution_date = context["ds"]
    game_date = date.fromisoformat(execution_date)
    run_id = context["ti"].xcom_pull(task_ids="extract", key="run_id") or "UNKNOWN"

    raw_df = run_extract(game_date, run_id)
    clean_df, _ = run_validate(raw_df)
    transformed_df = run_transform(clean_df, run_id)

    context["ti"].xcom_push(key="rows_transformed", value=len(transformed_df))
    return {"rows_transformed": len(transformed_df)}


def task_run_load(**context) -> dict:
    """Load stage: MERGE upsert into BigQuery."""
    from datetime import date
    from src.extract.api_client import run_extract
    from src.validate.quality_checks import run_validate
    from src.transform.transformations import run_transform

    execution_date = context["ds"]
    game_date = date.fromisoformat(execution_date)
    run_id = context["ti"].xcom_pull(task_ids="extract", key="run_id") or "UNKNOWN"

    raw_df = run_extract(game_date, run_id)
    clean_df, _ = run_validate(raw_df)
    transformed_df = run_transform(clean_df, run_id)
    result = run_load(transformed_df, run_id, game_date)

    context["ti"].xcom_push(key="rows_loaded", value=result["rows_loaded"])
    return result


def build_success_message(**context) -> str:
    ti = context["ti"]
    return (
        f"✅ *NBA Pipeline — SUCCESS*\n"
        f"Date: `{context['ds']}`\n"
        f"Extracted: `{ti.xcom_pull(task_ids='extract', key='rows_extracted')}` rows\n"
        f"Valid:      `{ti.xcom_pull(task_ids='validate', key='rows_valid')}` rows "
        f"({ti.xcom_pull(task_ids='validate', key='rows_dropped')} dropped)\n"
        f"Loaded:     `{ti.xcom_pull(task_ids='load', key='rows_loaded')}` rows to BigQuery\n"
        f"Run ID:     `{ti.xcom_pull(task_ids='extract', key='run_id')}`"
    )


# ── DAG definition ────────────────────────────────────────────────────────────

with DAG(
    dag_id="nba_box_scores_pipeline",
    description="Daily ETL: NBA Stats API → BigQuery (fact_box_scores)",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2024, 10, 22),     # Start of 2024-25 regular season
    schedule_interval="0 11 * * *",        # 6:00 AM ET = 11:00 UTC
    catchup=False,                         # Don't backfill historical dates on first run
    max_active_runs=1,                     # Prevent overlapping daily runs
    tags=["nba", "etl", "bigquery"],
) as dag:

    start = EmptyOperator(task_id="start")

    check_api = PythonOperator(
        task_id="check_api_health",
        python_callable=task_check_api_health,
    )

    extract = PythonOperator(
        task_id="extract",
        python_callable=task_run_extract,
    )

    validate = PythonOperator(
        task_id="validate",
        python_callable=task_run_validate,
    )

    transform = PythonOperator(
        task_id="transform",
        python_callable=task_run_transform,
    )

    load = PythonOperator(
        task_id="load",
        python_callable=task_run_load,
    )

    notify_success = SlackWebhookOperator(
        task_id="notify_success",
        slack_webhook_conn_id="slack_webhook_nba_pipeline",
        message="{{ ti.xcom_pull(task_ids='notify_success_message') }}",
        trigger_rule="all_success",
    )

    notify_failure = SlackWebhookOperator(
        task_id="notify_failure",
        slack_webhook_conn_id="slack_webhook_nba_pipeline",
        message="❌ *NBA Pipeline — FAILED* on `{{ ds }}`. Check Airflow logs.",
        trigger_rule="one_failed",
    )

    end = EmptyOperator(task_id="end", trigger_rule="none_failed_or_skipped")

    # ── Task dependencies ─────────────────────────────────────────────────────
    (
        start
        >> check_api
        >> extract
        >> validate
        >> transform
        >> load
        >> [notify_success, notify_failure]
        >> end
    )
