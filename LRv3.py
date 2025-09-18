# ou_first3_games_fixed.py
# Pull Oklahoma's first 3 games for 2025, handling camelCase and snake_case.
# Requires: pip install -U cfbd pandas

TOKEN = "xxZIIJc17ZxcBV0YrTflln9+LIGna7x+Fnmo/rUE5kSe6i4Y6WpAAHdlPoDNNSVs"
SEASON = 2025
TEAM = "Oklahoma"
SEASON_TYPE = "regular"   # "regular" | "postseason" | "both"
WEEKS = [1, 2, 3]

import pandas as pd
import cfbd

def first(*vals):
    """Return the first non-None, non-empty value."""
    for v in vals:
        if v is not None and v != "":
            return v
    return None

def enum_value(x):
    """Return enum .value when present, else plain x."""
    return getattr(x, "value", x)

def game_to_row(g):
    # cfbd models often serialize to camelCase dicts via to_dict()
    d = g.to_dict() if hasattr(g, "to_dict") else (g if isinstance(g, dict) else g.__dict__)

    # Core fields (support both snake_case and camelCase)
    season       = first(d.get("season"))
    season_type  = enum_value(first(d.get("season_type"), d.get("seasonType")))
    week         = first(d.get("week"))
    start_date   = first(d.get("start_date"), d.get("startDate"))
    game_id      = first(d.get("game_id"), d.get("id"))

    home_team    = first(d.get("home_team"), d.get("homeTeam"))
    away_team    = first(d.get("away_team"), d.get("awayTeam"))
    home_conf    = first(d.get("home_conference"), d.get("homeConference"))
    away_conf    = first(d.get("away_conference"), d.get("awayConference"))
    home_points  = first(d.get("home_points"), d.get("homePoints"))
    away_points  = first(d.get("away_points"), d.get("awayPoints"))
    neutral_site = first(d.get("neutral_site"), d.get("neutralSite"))
    conf_game    = first(d.get("conference_game"), d.get("conferenceGame"))
    venue        = first(d.get("venue"), d.get("venueName"))
    city         = first(d.get("venue_city"), d.get("city"))
    state        = first(d.get("venue_state"), d.get("state"))

    # Line scores (quarters), support both styles
    home_ls = first(d.get("home_line_scores"), d.get("homeLineScores")) or []
    away_ls = first(d.get("away_line_scores"), d.get("awayLineScores")) or []

    row = {
        "season": season,
        "season_type": season_type,
        "week": week,
        "start_date": start_date,
        "game_id": game_id,
        "home_team": home_team,
        "away_team": away_team,
        "home_conference": home_conf,
        "away_conference": away_conf,
        "home_points": home_points,
        "away_points": away_points,
        "neutral_site": neutral_site,
        "conference_game": conf_game,
        "venue": venue,
        "city": city,
        "state": state,
    }

    # Expand quarters up to 5 periods (Q1..Q5/OT)
    for i, v in enumerate(home_ls[:5], start=1):
        row[f"home_Q{i}"] = v
    for i, v in enumerate(away_ls[:5], start=1):
        row[f"away_Q{i}"] = v

    return row

def fetch_ou_games_first3(api_client):
    games_api = cfbd.GamesApi(api_client)
    rows = []
    for wk in WEEKS:
        try:
            games = games_api.get_games(year=SEASON, week=wk, team=TEAM, season_type=SEASON_TYPE) or []
        except Exception as e:
            print(f"[debug] get_games failed for week {wk}: {e}")
            games = []
        for g in games:
            rows.append(game_to_row(g))
    return pd.DataFrame(rows)

def attach_betting_lines(df, api_client):
    """
    Add two columns to df by game_id:
      - close_spread: median closing spread across providers (favored team negative)
      - close_total:  median closing over/under across providers

    Notes:
    - CFBD 'lines' can return multiple providers per game.
    - We only keep numeric values and take the median for robustness.
    - If a game has no lines returned, we leave NaN.
    """
    lines_api = cfbd.BettingApi(api_client)

    def to_num(x):
        # Convert '3.5', -7.5, or strings like '+3.5' -> float, else None
        if x is None:
            return None
        try:
            return float(str(x).replace('+', ''))
        except Exception:
            return None

    rows = []
    for gid in df["game_id"].dropna().tolist():
        try:
            # Ask CFBD for lines just for this game id (keeps payload tiny)
            # You can also pass year/season_type, but game_id is the key.
            games = lines_api.get_lines(game_id=int(gid), season_type=SEASON_TYPE) or []
        except Exception as e:
            print(f"[debug] get_lines failed for game {gid}: {e}")
            games = []

        if not games:
            continue

        # The response is a list of "game" containers; each has `.lines` for providers
        g0 = games[0].to_dict() if hasattr(games[0], "to_dict") else (
            games[0] if isinstance(games[0], dict) else games[0].__dict__
        )
        provider_lines = g0.get("lines") or []

        spreads, totals = [], []
        for ln in provider_lines:
            d = ln.to_dict() if hasattr(ln, "to_dict") else (ln if isinstance(ln, dict) else ln.__dict__)
            # CFBD uses 'spread' and 'overUnder' (camel) in many clients; some wrappers expose 'over_under'
            sp = to_num(d.get("spread"))
            ou = to_num(d.get("overUnder", d.get("over_under")))
            if sp is not None:
                spreads.append(sp)
            if ou is not None:
                totals.append(ou)

        if spreads or totals:
            rows.append({
                "game_id": int(gid),
                "close_spread": float(pd.Series(spreads).median()) if spreads else None,
                "close_total": float(pd.Series(totals).median()) if totals else None,
            })

    # Merge back to original df
    if not rows:
        return df
    lines_df = pd.DataFrame(rows)
    return df.merge(lines_df, on="game_id", how="left")

def attach_targets_and_baseline(df):
    """
    Add basic targets and baseline error columns using closing lines.

    Assumptions:
    - 'close_spread' is from the HOME team's perspective.
      If home spread is -3.5, home must win by > 3.5 to cover.
      A handy identity: cover check = (home_margin + close_spread).
    - 'close_total' is the closing over/under.

    Outputs:
    - home_margin, game_total
    - home_cover_margin, home_cover (cover/push/no_cover)
    - spread_error, abs_spread_error
    - total_error, abs_total_error, total_result (over/push/under)
    """
    df = df.copy()

    # Final score derived fields
    if {"home_points", "away_points"}.issubset(df.columns):
        df["home_margin"] = df["home_points"] - df["away_points"]
        df["game_total"] = df["home_points"] + df["away_points"]
    else:
        # If points are missing, nothing to do
        return df

    # Spread-based targets and baseline error
    if "close_spread" in df.columns:
        # Positive => home covered, 0 => push, negative => no cover
        df["home_cover_margin"] = df["home_margin"] + df["close_spread"]

        def _cover_label(x):
            if pd.isna(x):
                return pd.NA
            if x > 0:
                return "cover"
            if x == 0:
                return "push"
            return "no_cover"

        df["home_cover"] = df["home_cover_margin"].apply(_cover_label)

        # Baseline error vs the spread-implied margin (expected margin = -close_spread)
        # spread_error = actual_margin - expected_margin = home_margin - (-spread) = home_margin + spread
        df["spread_error"] = df["home_margin"] + df["close_spread"]
        df["abs_spread_error"] = df["spread_error"].abs()

    # Total-based baseline error
    if "close_total" in df.columns:
        df["total_error"] = df["game_total"] - df["close_total"]
        df["abs_total_error"] = df["total_error"].abs()

        def _ou_label(x):
            if pd.isna(x):
                return pd.NA
            if x > 0:
                return "over"
            if x == 0:
                return "push"
            return "under"

        df["total_result"] = df["total_error"].apply(_ou_label)

    return df

def main():
    config = cfbd.Configuration(access_token=TOKEN)
    with cfbd.ApiClient(config) as api_client:
        df = fetch_ou_games_first3(api_client)
        if df.empty:
            print("No games returned. Try changing WEEKS or SEASON_TYPE.")
            return
        bl_df = attach_betting_lines(df, api_client)
        bl_df = attach_targets_and_baseline(bl_df)

        # Pretty column order
        meta = [
            "season","season_type","week","start_date","game_id",
            "home_team","away_team","home_conference","away_conference",
            "home_points","away_points","neutral_site","conference_game",
            "venue","city","state",
        ]
        qcols = [c for c in df.columns if c.startswith("home_Q") or c.startswith("away_Q")]
        cols = [c for c in meta + qcols if c in df.columns]
        df = df[cols].sort_values(["week","start_date"], na_position="last").reset_index(drop=True)

        #print(bl_df.head())
        print(bl_df[[
        "season","week","home_team","away_team","home_points","away_points",
        "close_spread","home_margin","home_cover_margin","home_cover",
        "close_total","game_total","total_result"
        ]].head())

        #df.to_csv(f"oklahoma_first3_games_{SEASON}.csv", index=False)
        #print(f"Saved: oklahoma_first3_games_{SEASON}.csv")

if __name__ == "__main__":
    main()
