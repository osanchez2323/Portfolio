# NBA Data Pipeline

A production-grade ETL pipeline that ingests NBA box score data from the NBA Stats API, enforces data quality rules, applies transformation logic, and loads analysis-ready records into a partitioned BigQuery data warehouse.

---

## Project Overview

This pipeline runs daily, pulling the prior day's game logs and upserting them into BigQuery. It demonstrates core data engineering patterns including incremental loading, idempotent upserts, schema validation, derived metric calculation, and orchestration via Apache Airflow.

**Dataset:** 2024–25 NBA season box scores · 82 regular season games · ~25 active players per game · 13 tracked metrics per player-game record

---

## Pipeline Architecture

```
NBA Stats API  →  Extract  →  Validate  →  Transform  →  Load (BigQuery)
                   (01)         (02)          (03)           (04)
```

| Stage | Description | Key Tools |
|-------|-------------|-----------|
| **01 Extract** | HTTP pull from stats.nba.com, JSON parse, raw Parquet landing | `requests`, `pandas` |
| **02 Validate** | Schema enforcement, null checks, range rules, FK integrity | `pandas`, `great_expectations` |
| **03 Transform** | Type casting, derived metrics, dimension joins, audit fields | `pandas`, `numpy`, SQL |
| **04 Load** | BigQuery MERGE upsert, date partitioning, audit log | `google-cloud-bigquery` |
| **Orchestrate** | Daily DAG trigger, retry logic, alerting | `Apache Airflow` |

---

## Repository Structure

```
nba-pipeline/
├── README.md
├── requirements.txt
├── .env.example
├── config/
│   └── settings.py            # All config: API endpoints, BQ project/dataset, table names
├── src/
│   ├── extract/
│   │   └── api_client.py      # NBA Stats API client with retry logic
│   ├── validate/
│   │   └── quality_checks.py  # Schema, null, range, and FK validation
│   ├── transform/
│   │   └── transformations.py # Type casting, derived metrics, enrichment joins
│   ├── load/
│   │   └── bigquery_loader.py # BigQuery MERGE upsert and audit logging
│   └── utils/
│       └── logger.py          # Structured logging utility
├── sql/
│   ├── create_fact_box_scores.sql    # DDL for the target BQ table
│   ├── create_dim_tables.sql         # DDL for dimension tables
│   └── merge_box_scores.sql          # MERGE / upsert statement
├── airflow/
│   └── dags/
│       └── nba_pipeline_dag.py       # Airflow DAG definition
├── tests/
│   ├── test_extract.py
│   ├── test_validate.py
│   ├── test_transform.py
│   └── test_load.py
├── data/
│   └── samples/
│       ├── raw_api_response_sample.json    # Sample raw API response
│       └── transformed_output_sample.csv   # Sample transformed output
└── docs/
    └── data_dictionary.md     # Field definitions and business rules
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- Google Cloud project with BigQuery enabled
- Service account with BigQuery Data Editor and Job User roles
- Apache Airflow 2.x (for orchestration)

### Installation

```bash
# Clone the repository
git clone https://github.com/osanchez2323/Portfolio.git
cd nba-pipeline

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your GCP project and credentials
```

### Configuration

Copy `.env.example` to `.env` and fill in your values:

```
GCP_PROJECT_ID=your-project-id
BQ_DATASET=nba_dw
BQ_TABLE=fact_box_scores
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json
NBA_API_BASE_URL=https://stats.nba.com/stats
NBA_SEASON=2024-25
```

### Running the Pipeline

```bash
# Run the full pipeline for yesterday's games
python -m src.main

# Run for a specific date
python -m src.main --date 2025-04-15

# Run a specific stage only
python -m src.main --stage extract
python -m src.main --stage validate
python -m src.main --stage transform
python -m src.main --stage load

# Run with dry-run (no BQ write)
python -m src.main --dry-run
```

### Running Tests

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=html
```

---

## BigQuery Table Schema

**Table:** `{project}.nba_dw.fact_box_scores`  
**Partition:** `game_date` (DATE)  
**Cluster:** `team_id`, `player_id`

| Column | Type | Description |
|--------|------|-------------|
| `game_id` | STRING | NBA game identifier (PK component) |
| `player_id` | INT64 | NBA player identifier (PK component) |
| `game_date` | DATE | Game date — partition column |
| `player_name` | STRING | Full name, title-cased |
| `team_id` | INT64 | Team identifier |
| `team_abbreviation` | STRING | 3-letter team code |
| `pts` | INT64 | Points scored |
| `reb` | INT64 | Total rebounds |
| `ast` | INT64 | Assists |
| `stl` | INT64 | Steals |
| `blk` | INT64 | Blocks |
| `tov` | INT64 | Turnovers |
| `min_played` | FLOAT64 | Minutes played (decimal) |
| `fg_pct` | FLOAT64 | Field goal percentage |
| `ft_pct` | FLOAT64 | Free throw percentage |
| `ts_pct` | FLOAT64 | True shooting percentage (derived) |
| `game_score` | FLOAT64 | Hollinger game score (derived) |
| `roll5_pts` | FLOAT64 | 5-game rolling average points |
| `is_home` | BOOL | True if player's team was home |
| `load_timestamp` | TIMESTAMP | Pipeline load time (audit) |
| `pipeline_run_id` | STRING | Unique run identifier (audit) |
| `source_system` | STRING | Source identifier (audit) |
| `row_hash` | STRING | MD5 hash of record for change detection |

---

## Data Quality Rules

| Rule | Type | Threshold | Action on Fail |
|------|------|-----------|----------------|
| Schema: all 13 expected columns present | Hard | 100% | Abort pipeline |
| Null rate: `ts_pct` | Soft | < 2% | Log warning, continue |
| Inactive players: `min = 0` | Hard | — | Drop row |
| Range: `fg_pct` between 0.0–1.0 | Hard | 100% | Quarantine row |
| Range: `pts` between 0–70 | Hard | 100% | Quarantine row |
| FK integrity: `game_id` in `dim_games` | Hard | 100% | Quarantine row |

---

## Derived Metrics

**True Shooting % (TS%)**
```
ts_pct = pts / (2 × (fga + 0.44 × fta))
```

**Hollinger Game Score**
```
game_score = pts + 0.4×fgm − 0.7×fga − 0.4×(fta − ftm)
             + 0.7×orb + 0.3×drb + stl + 0.7×ast + 0.7×blk
             − 0.4×pf − tov
```

**5-Game Rolling Average**
```python
df['roll5_pts'] = df.groupby('player_id')['pts'].transform(
    lambda x: x.rolling(5, min_periods=1).mean()
)
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
