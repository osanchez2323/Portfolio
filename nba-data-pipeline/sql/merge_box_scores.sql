-- sql/merge_box_scores.sql
-- ============================================================
-- MERGE statement for upserting box score records from a
-- temporary staging table into the fact_box_scores table.
--
-- The MERGE is keyed on (game_id, player_id) — the natural
-- composite primary key for box scores.
--
-- WHEN MATCHED AND row_hash differs → UPDATE stats
--   Only updates rows where data has actually changed.
--   Unchanged records (same row_hash) are skipped, reducing
--   unnecessary write operations and query costs.
--
-- WHEN NOT MATCHED → INSERT new row
--   Inserts brand-new player-game records.
--
-- This pattern makes the pipeline fully idempotent:
--   re-running for the same date always produces the same result.
--
-- Parameters (substituted by bigquery_loader.py):
--   {target_table}  — fully-qualified fact table name
--   {staging_table} — fully-qualified staging table name
-- ============================================================

MERGE `{target_table}` AS target
USING `{staging_table}` AS source
ON target.game_id   = source.game_id
AND target.player_id = source.player_id

-- ── Update existing record only if data has changed ─────────
WHEN MATCHED AND target.row_hash != source.row_hash THEN
  UPDATE SET
    player_name         = source.player_name,
    team_id             = source.team_id,
    team_abbreviation   = source.team_abbreviation,
    min_played          = source.min_played,
    pts                 = source.pts,
    reb                 = source.reb,
    ast                 = source.ast,
    stl                 = source.stl,
    blk                 = source.blk,
    tov                 = source.tov,
    fgm                 = source.fgm,
    fga                 = source.fga,
    fg_pct              = source.fg_pct,
    ftm                 = source.ftm,
    fta                 = source.fta,
    ft_pct              = source.ft_pct,
    fg3m                = source.fg3m,
    fg3a                = source.fg3a,
    oreb                = source.oreb,
    dreb                = source.dreb,
    pf                  = source.pf,
    ts_pct              = source.ts_pct,
    game_score          = source.game_score,
    roll5_pts           = source.roll5_pts,
    is_home             = source.is_home,
    load_timestamp      = source.load_timestamp,
    pipeline_run_id     = source.pipeline_run_id,
    row_hash            = source.row_hash

-- ── Insert new record ───────────────────────────────────────
WHEN NOT MATCHED THEN
  INSERT ROW
;
