# Data Dictionary — NBA Data Pipeline

This document defines every field in the `fact_box_scores` table, including
data type, source, transformation logic, and business rules.

---

## fact_box_scores

One row per player per game. Partitioned by `game_date`, clustered by `team_id` and `player_id`.

| Column | Type | Source | Description |
|--------|------|--------|-------------|
| `game_id` | STRING | API | NBA game identifier (e.g. `0022401012`). First four digits are season year; next two are season type (02 = regular season); last five are sequential game number. |
| `player_id` | INT64 | API | NBA internal player identifier. Stable across seasons. |
| `game_date` | DATE | API (parsed) | Date the game was played. Raw API provides "APR 15, 2025" — parsed to DATE in Transform stage. **Partition key.** |
| `player_name` | STRING | API (normalised) | Full player name. Raw API returns inconsistent casing (e.g. "LEBRON JAMES") — normalised to Title Case in Transform. |
| `team_id` | INT64 | API | NBA team identifier for the player's team in this game. |
| `team_abbreviation` | STRING | API | 3-letter team code (e.g. LAL, GSW). |
| `min_played` | FLOAT64 | API (parsed) | Minutes played. Raw API returns "32:14" string — parsed to decimal float 32.23 in Transform. Players with 0 minutes are dropped in Validate. |
| `pts` | INT64 | API (cast) | Points scored. Raw API returns float — cast to INT in Transform. |
| `reb` | INT64 | API (cast) | Total rebounds (offensive + defensive). |
| `ast` | INT64 | API (cast) | Assists. |
| `stl` | INT64 | API (cast) | Steals. |
| `blk` | INT64 | API (cast) | Blocked shots. |
| `tov` | INT64 | API (cast) | Turnovers. |
| `fgm` | INT64 | API (cast) | Field goals made. |
| `fga` | INT64 | API (cast) | Field goals attempted. |
| `fg_pct` | FLOAT64 | API | Field goal percentage (0.0–1.0). Null when FGA = 0. Range validated in QA (hard rule). |
| `ftm` | INT64 | API (cast) | Free throws made. |
| `fta` | INT64 | API (cast) | Free throws attempted. |
| `ft_pct` | FLOAT64 | API | Free throw percentage (0.0–1.0). Null when FTA = 0. |
| `fg3m` | INT64 | API (cast) | 3-point field goals made. |
| `fg3a` | INT64 | API (cast) | 3-point field goals attempted. |
| `oreb` | INT64 | API (cast) | Offensive rebounds. |
| `dreb` | INT64 | API (cast) | Defensive rebounds. |
| `pf` | INT64 | API (cast) | Personal fouls. |
| `ts_pct` | FLOAT64 | **Derived** | True Shooting %. See formula below. Null when FGA = FTA = 0. Soft QA warning if null rate > 2%. |
| `game_score` | FLOAT64 | **Derived** | Hollinger Game Score. See formula below. Single-number performance summary. |
| `roll5_pts` | FLOAT64 | **Derived** | 5-game rolling average points for this player. Sorted by game_date per player. min_periods=1 (partial windows allowed). |
| `is_home` | BOOL | **Derived** | True if the player's team was the home team. Derived from MATCHUP field ("vs." = home, "@" = away). |
| `load_timestamp` | TIMESTAMP | **Audit** | UTC timestamp when this record was written by the pipeline. |
| `pipeline_run_id` | STRING | **Audit** | Unique 8-character identifier for the pipeline run (e.g. `A3F9C12B`). Links to `pipeline_audit_log`. |
| `source_system` | STRING | **Audit** | Always `nba_stats_api`. Identifies the upstream source for lineage tracking. |
| `is_current` | BOOL | **Audit** | Always `TRUE` on insert. Reserved for SCD Type 2 if historical versioning is added. |
| `row_hash` | STRING | **Audit** | MD5 hash of (game_id, player_id, pts, reb, ast, stl, blk, tov). Used in MERGE to skip updates where data hasn't changed. |

---

## Derived Metric Formulas

### True Shooting % (ts_pct)

Accounts for the relative value of 2-point FGs, 3-point FGs, and free throws.
Widely considered the best single-number measure of shooting efficiency.

```
ts_pct = pts / (2 × (fga + 0.44 × fta))
```

The 0.44 factor approximates the proportion of free throw trips that are part of
2-shot fouls (vs. 3-shot fouls, and-ones, technicals). League average ≈ 0.565.

### Hollinger Game Score (game_score)

Summarises a player's total statistical contribution in a single game.
Developed by John Hollinger. League average ≈ 10.

```
game_score = pts + 0.4×fgm − 0.7×fga − 0.4×(fta − ftm)
             + 0.7×oreb + 0.3×dreb + stl + 0.7×ast + 0.7×blk
             − 0.4×pf − tov
```

Reference scores: 10 = average; 25 = excellent; 40+ = all-time single game.

---

## Data Quality Rules

| Rule | Column | Type | Threshold | On Fail |
|------|--------|------|-----------|---------|
| Schema completeness | All expected columns | HARD | 100% present | Abort pipeline |
| Inactive players | `min_played` | HARD | MIN > 0 | Drop row |
| Null rate — TS% | `ts_pct` | SOFT | < 2% | Log warning, continue |
| FG% range | `fg_pct` | HARD | 0.0 – 1.0 | Quarantine row |
| PTS range | `pts` | HARD | 0 – 70 | Quarantine row |
| Overall drop rate | All | HARD | < 5% | Abort pipeline |

---

## Dimension Tables

### dim_players
One row per NBA player. Sourced from NBA Stats API `/commonallplayers`.
Updated nightly.

Key columns: `player_id` (PK), `player_name`, `position`, `is_active`.

### dim_teams
One row per NBA team. Static — updated only when franchises relocate or expand.

Key columns: `team_id` (PK), `team_abbreviation`, `conference`, `division`.

### dim_games
One row per NBA game. Sourced from NBA Stats API `/leaguegamefinder`.
Provides home/away context, attendance, and final scores.

Key columns: `game_id` (PK), `game_date`, `home_team_id`, `away_team_id`.
Partitioned by `game_date`.
