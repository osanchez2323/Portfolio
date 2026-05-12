-- sql/create_dim_tables.sql
-- ============================================================
-- DDL for the three dimension tables used in enrichment joins
-- during the Transform stage.
-- ============================================================


-- ── dim_players ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS `{project}.nba_dw.dim_players`
(
    player_id       INT64   NOT NULL  OPTIONS(description="NBA player identifier (PK)"),
    player_name     STRING            OPTIONS(description="Full name, title-cased"),
    position        STRING            OPTIONS(description="Primary position: G, F, C, G-F, F-C"),
    height_inches   INT64             OPTIONS(description="Height in inches"),
    weight_lbs      INT64             OPTIONS(description="Weight in pounds"),
    birth_date      DATE              OPTIONS(description="Date of birth"),
    draft_year      INT64             OPTIONS(description="Year player was drafted (null if undrafted)"),
    draft_round     INT64             OPTIONS(description="Draft round (null if undrafted)"),
    draft_pick      INT64             OPTIONS(description="Draft pick number (null if undrafted)"),
    college         STRING            OPTIONS(description="College attended (null if international/HS)"),
    country         STRING            OPTIONS(description="Country of origin"),
    is_active       BOOL              OPTIONS(description="True if player is currently on an NBA roster"),
    updated_at      TIMESTAMP         OPTIONS(description="Last updated timestamp")
)
OPTIONS (
    description = "NBA player dimension table — one row per player."
);


-- ── dim_teams ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS `{project}.nba_dw.dim_teams`
(
    team_id           INT64   NOT NULL  OPTIONS(description="NBA team identifier (PK)"),
    team_abbreviation STRING            OPTIONS(description="3-letter code, e.g. LAL"),
    team_name         STRING            OPTIONS(description="Full team name, e.g. Los Angeles Lakers"),
    team_city         STRING            OPTIONS(description="City, e.g. Los Angeles"),
    team_nickname     STRING            OPTIONS(description="Nickname, e.g. Lakers"),
    conference        STRING            OPTIONS(description="Eastern or Western"),
    division          STRING            OPTIONS(description="Atlantic / Central / Southeast / Northwest / Pacific / Southwest"),
    arena             STRING            OPTIONS(description="Home arena name"),
    arena_capacity    INT64             OPTIONS(description="Seating capacity of home arena"),
    founded_year      INT64             OPTIONS(description="Year the franchise was founded"),
    is_active         BOOL              OPTIONS(description="True if team is currently in the NBA"),
    updated_at        TIMESTAMP         OPTIONS(description="Last updated timestamp")
)
OPTIONS (
    description = "NBA team dimension table — one row per team."
);


-- ── dim_games ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS `{project}.nba_dw.dim_games`
(
    game_id           STRING  NOT NULL  OPTIONS(description="NBA game identifier (PK)"),
    game_date         DATE              OPTIONS(description="Date the game was played"),
    home_team_id      INT64             OPTIONS(description="FK to dim_teams — home team"),
    away_team_id      INT64             OPTIONS(description="FK to dim_teams — away team"),
    home_team_score   INT64             OPTIONS(description="Final score for home team"),
    away_team_score   INT64             OPTIONS(description="Final score for away team"),
    season            STRING            OPTIONS(description="Season identifier, e.g. 2024-25"),
    season_type       STRING            OPTIONS(description="Regular Season / Playoffs / Pre-Season"),
    arena             STRING            OPTIONS(description="Arena where game was played"),
    attendance        INT64             OPTIONS(description="Attendance count"),
    game_duration_min INT64             OPTIONS(description="Game duration in minutes (including OT if applicable)"),
    updated_at        TIMESTAMP         OPTIONS(description="Last updated timestamp")
)
PARTITION BY game_date
OPTIONS (
    description = "NBA game dimension table — one row per game."
);
