# -*- coding: utf-8 -*-
"""
NCAAF Over/Under modular pipeline (TeamRankings + optional CFBD)
- Clean scraping, no globals, typed, easy to extend with player stats
- Train/test by date to avoid leakage
"""

from __future__ import annotations
import os
import sys
from dataclasses import dataclass
from typing import Optional, List, Tuple
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss

UA = {"User-Agent": "Mozilla/5.0"}

TR_BASE = "https://www.teamrankings.com/college-football/team/{team_slug}"

@dataclass
class TeamData:
    games: pd.DataFrame  # per-game features & labels

def _read_tables(url: str) -> List[pd.DataFrame]:
    r = requests.get(url, headers=UA, timeout=20)
    r.raise_for_status()
    return pd.read_html(r.text)

def fetch_game_log(team_slug: str) -> pd.DataFrame:
    """
    Scrapes TeamRankings 'game log' for a team.
    """
    url = f"{TR_BASE.format(team_slug=team_slug)}/game-log"
    tables = _read_tables(url)
    if not tables:
        raise ValueError("No tables found on game log page.")
    df = tables[0].copy()

    # Normalize columns (TR sometimes changes casing/spacing)
    df.columns = [c.strip() for c in df.columns]

    # Split W/L and Score safely
    if "Score" not in df.columns:
        raise ValueError("Expected 'Score' column not found.")
    df["W/L"] = df["Score"].str[0]
    df["Score"] = df["Score"].str[2:]  # drop "W " or "L "
    pts = df["Score"].str.split("-", expand=True)
    df["Pts"] = pd.to_numeric(pts[0], errors="coerce")
    df["Opp Pts"] = pd.to_numeric(pts[1], errors="coerce")

    # Basic date if present
    for col in ["Date", "Game Date"]:
        if col in df.columns:
            df["game_date"] = pd.to_datetime(df[col], errors="coerce")
            break
    if "game_date" not in df.columns:
        df["game_date"] = pd.NaT

    # Home/Away flag from Opponent column if available (e.g., "at LSU")
    if "Opponent" in df.columns:
        df["is_away"] = df["Opponent"].astype(str).str.startswith("at ")
    else:
        df["is_away"] = False

    return df

def fetch_over_under(team_slug: str) -> pd.DataFrame:
    """
    Scrapes TeamRankings 'over/under results' for a team.
    We’ll merge on Score + nearest date as a fallback.
    """
    url = f"{TR_BASE.format(team_slug=team_slug)}/over-under-results"
    tables = _read_tables(url)
    if not tables:
        raise ValueError("No tables found on over/under page.")
    df = tables[0].copy()
    df.columns = [c.strip() for c in df.columns]
    # Keep Total (closing total), Result (Over/Under), Diff
    keep = [c for c in ["Score", "Total", "Result", "Diff", "Date", "Game Date"] if c in df.columns]
    return df[keep]

def assemble_team_frame(team_slug: str) -> TeamData:
    gl = fetch_game_log(team_slug)
    ou = fetch_over_under(team_slug)

    # Best-effort merge: try Score first
    merged = gl.merge(ou, on="Score", how="left", suffixes=("", "_ou"))

    # If some Total/Result still NA and we have dates, try date merge (nearest)
    if merged["Total"].isna().any() and "game_date" in merged and any(c in ou.columns for c in ["Date", "Game Date"]):
        ou2 = ou.copy()
        for col in ["Date", "Game Date"]:
            if col in ou2.columns:
                ou2["game_date"] = pd.to_datetime(ou2[col], errors="coerce")
                break
        if "game_date" in ou2:
            merged = merged.merge(
                ou2[["game_date", "Total", "Result", "Diff"]],
                on="game_date", how="left", suffixes=("", "_bydate")
            )
            # Prefer exact Score match, else fallback to date
            merged["Total"] = merged["Total"].fillna(merged.pop("Total_bydate"))
            merged["Result"] = merged["Result"].fillna(merged.pop("Result_bydate"))
            merged["Diff"] = merged["Diff"].fillna(merged.pop("Diff_bydate"))

    # Label: Over=1 else 0
    merged["Over_Under"] = (merged["Result"].astype(str).str.lower() == "over").astype(int)

    # Basic engineered features (you can add rolling forms later)
    merged["pts_allowed"] = merged["Opp Pts"]
    merged["total_line"] = pd.to_numeric(merged["Total"], errors="coerce")

    # Keep only rows with a target
    merged = merged.dropna(subset=["Over_Under", "Pts", "Opp Pts", "total_line"])

    return TeamData(games=merged)

def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Build X, y with a date index for time-wise CV.
    """
    # Simple numeric set; expand later with rolling stats/tempo/opponent strength
    features = ["Pts", "Opp Pts", "total_line", "is_away"]
    X = df[features].copy()
    X["is_away"] = X["is_away"].astype(int)
    y = df["Over_Under"].astype(int)
    dates = df["game_date"]
    return X, y, dates

def train_eval_time_split(X: pd.DataFrame, y: pd.Series, dates: pd.Series) -> Tuple[Pipeline, dict]:
    """
    TimeSeries CV for classification to avoid leakage.
    """
    model = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),  # sparse-safe; fine here
        ("lr", LogisticRegression(max_iter=200, class_weight="balanced"))
    ])

    # TimeSeriesSplit (k=3 by default); fall back if too few samples
    n_splits = 3 if len(X) >= 12 else 2
    tscv = TimeSeriesSplit(n_splits=n_splits)

    accs, aucs, briers = [], [], []
    for train_idx, test_idx in tscv.split(X):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = model.predict_proba(X.iloc[test_idx])[:, 1]
        pred = (proba >= 0.5).astype(int)
        accs.append(accuracy_score(y.iloc[test_idx], pred))
        try:
            aucs.append(roc_auc_score(y.iloc[test_idx], proba))
        except ValueError:
            aucs.append(np.nan)
        briers.append(brier_score_loss(y.iloc[test_idx], proba))

    model.fit(X, y)  # final fit on all history
    metrics = {
        "cv_acc_mean": float(np.nanmean(accs)),
        "cv_auc_mean": float(np.nanmean(aucs)),
        "cv_brier_mean": float(np.nanmean(briers)),
        "n_samples": int(len(X))
    }
    return model, metrics

# -------- Optional: Player stats via CollegeFootballData API --------
def fetch_cfbd_player_game_stats(season: int, team: Optional[str]=None, game_id: Optional[int]=None) -> pd.DataFrame:
    """
    Uses CFBD /games/players to get per-game player stats (passing/rushing/receiving).
    Requires CFBD_API_KEY in env.
    """
    api_key = os.getenv("CFBD_API_KEY")
    if not api_key:
        raise RuntimeError("Set CFBD_API_KEY environment variable for CollegeFootballData API.")
    params = {"year": season}
    if team:
        params["team"] = team
    if game_id:
        params["gameId"] = game_id
    r = requests.get(
        "https://api.collegefootballdata.com/games/players",
        headers={"Authorization": f"Bearer {api_key}"},
        params=params,
        timeout=30
    )
    r.raise_for_status()
    data = r.json()
    return pd.json_normalize(data)

def example_player_targets(team_name: str, season: int) -> pd.DataFrame:
    """
    Builds a toy frame for player receiving/rushing/passing yards from CFBD.
    """
    df = fetch_cfbd_player_game_stats(season=season, team=team_name)
    # Columns include things like 'player', 'statType', 'statYards', depends on feed structure
    # Example reshape to wide per player-game:
    if "athlete.name" in df.columns:
        df.rename(columns={"athlete.name": "player"}, inplace=True)
    # Extract common yard stats if present
    yard_cols = [c for c in df.columns if c.lower().endswith("yards")]
    keep = ["player", "team", "gameId", "home", "away", "id", "position"] + yard_cols
    keep = [c for c in keep if c in df.columns]
    return df[keep].copy()

# ----------------------------- CLI -----------------------------
def run(team_slug: str, team_name_for_cfbd: Optional[str] = None, season_for_cfbd: Optional[int] = None):
    td = assemble_team_frame(team_slug)
    X, y, dates = build_features(td.games)
    model, cv = train_eval_time_split(X, y, dates)

    print("CV metrics:", cv)

    # Example future prediction (replace with real upcoming total & context)
    future = pd.DataFrame([{"Pts": 34, "Opp Pts": 26.9, "total_line": 51.5, "is_away": 0}])
    prob_over = model.predict_proba(future)[:, 1][0]
    print(f"Predicted P(Over): {prob_over:.3f}")

    # Optional: Pull player stats
    if team_name_for_cfbd and season_for_cfbd:
        try:
            players = example_player_targets(team_name_for_cfbd, season_for_cfbd)
            print(players.head(10))
        except Exception as e:
            print(f"[CFBD] Skipped player stats: {e}")

if __name__ == "__main__":
    # Example: Ole Miss
    # team_slug is the TR slug used in URLs; team_name_for_cfbd is the human name CFBD expects
    run(team_slug="mississippi-rebels", team_name_for_cfbd="Ole Miss", season_for_cfbd=2025)
