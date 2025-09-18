# -*- coding: utf-8 -*-
"""
NCAAF O/U — simple multi-team (SEC) trainer
- Scrape current season from TeamRankings for several SEC teams
- Merge game log + O/U results per team
- Stack rows, train a small Logistic Regression (80/20)
- Minimal deps: requests, pandas, scikit-learn, lxml
"""

import numpy as np
import pandas as pd
import requests
from io import StringIO
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss

UA = {"User-Agent": "Mozilla/5.0"}
TR_BASE = "https://www.teamrankings.com/college-football/team/{team_slug}"

# Keep this short and editable. Add/remove teams as you like.
SEC_TEAMS = [
    "alabama-crimson-tide",
    "arkansas-razorbacks",
    "auburn-tigers",
    "florida-gators",
    "georgia-bulldogs",
    "kentucky-wildcats",
    "lsu-tigers",
    "mississippi-rebels",           # Ole Miss
    "mississippi-state-bulldogs",
    "missouri-tigers",
    "south-carolina-gamecocks",
    "tennessee-volunteers",
    "texas-am-aggies",
    "vanderbilt-commodores",
    "oklahoma-sooners",
    "texas-longhorns",
]

def read_tables(url: str) -> list[pd.DataFrame]:
    r = requests.get(url, headers=UA, timeout=20)
    r.raise_for_status()
    return pd.read_html(StringIO(r.text))

def fetch_game_log(team_slug: str) -> pd.DataFrame:
    url = f"{TR_BASE.format(team_slug=team_slug)}/game-log"
    tables = read_tables(url)
    if not tables:
        raise RuntimeError(f"[{team_slug}] No tables on game log page.")
    df = tables[0].copy()
    df.columns = [c.strip() for c in df.columns]

    if "Score" not in df.columns:
        raise RuntimeError(f"[{team_slug}] 'Score' column missing.")

    # Parse W/L and points
    df["W/L"] = df["Score"].str[0]
    df["Score"] = df["Score"].str[2:]
    pts = df["Score"].str.split("-", expand=True)
    df["Pts"] = pd.to_numeric(pts[0], errors="coerce")
    df["Opp Pts"] = pd.to_numeric(pts[1], errors="coerce")

    # Date
    date_col = None
    for col in ["Date", "Game Date"]:
        if col in df.columns:
            date_col = col
            break
    df["game_date"] = pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.NaT

    # Home/away flag
    if "Opponent" in df.columns:
        df["is_away"] = df["Opponent"].astype(str).str.startswith("at ").astype(int)
    else:
        df["is_away"] = 0

    df["team_slug"] = team_slug
    return df

def fetch_over_under(team_slug: str) -> pd.DataFrame:
    url = f"{TR_BASE.format(team_slug=team_slug)}/over-under-results"
    tables = read_tables(url)
    if not tables:
        raise RuntimeError(f"[{team_slug}] No tables on O/U page.")
    df = tables[0].copy()
    df.columns = [c.strip() for c in df.columns]
    keep = [c for c in ["Score", "Total", "Result", "Diff", "Date", "Game Date"] if c in df.columns]
    df = df[keep]
    df["team_slug"] = team_slug
    return df

def assemble_one_team(team_slug: str) -> pd.DataFrame:
    gl = fetch_game_log(team_slug)
    ou = fetch_over_under(team_slug)

    # Join on Score first (TR is consistent within a season)
    merged = gl.merge(ou, on=["Score", "team_slug"], how="left", suffixes=("", "_ou"))

    # Fallback: date merge for any missing
    if merged["Total"].isna().any():
        ou2 = ou.copy()
        dcol = next((c for c in ["Date", "Game Date"] if c in ou2.columns), None)
        if dcol:
            ou2["game_date"] = pd.to_datetime(ou2[dcol], errors="coerce")
            merged = merged.merge(
                ou2[["game_date", "Total", "Result", "Diff", "team_slug"]],
                on=["game_date", "team_slug"], how="left", suffixes=("", "_bydate")
            )
            merged["Total"] = merged["Total"].fillna(merged.pop("Total_bydate"))
            merged["Result"] = merged["Result"].fillna(merged.pop("Result_bydate"))
            merged["Diff"] = merged["Diff"].fillna(merged.pop("Diff_bydate"))

    merged["total_line"] = pd.to_numeric(merged["Total"], errors="coerce")
    merged["Over_Under"] = (merged["Result"].astype(str).str.lower() == "over").astype(int)

    # Keep rows we can actually use
    merged = merged.dropna(subset=["Pts", "Opp Pts", "total_line", "Over_Under"])
    return merged

def collect_sec_dataset() -> pd.DataFrame:
    frames = []
    for slug in SEC_TEAMS:
        try:
            df = assemble_one_team(slug)
            frames.append(df)
            print(f"[OK] {slug}: {len(df)} rows")
        except Exception as e:
            print(f"[Skip] {slug}: {e}")
    if frames:
        all_df = pd.concat(frames, ignore_index=True)
        # de-dup just in case
        all_df = all_df.drop_duplicates(subset=["team_slug", "Score", "game_date"])
        return all_df
    return pd.DataFrame()

def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    # NOTE: Using Pts/Opp Pts is leakage for true pregame predictions.
    # For now (learning phase), keep it simple; later replace with pregame proxies.
    X = df[["Pts", "Opp Pts", "total_line", "is_away"]].copy()
    y = df["Over_Under"].astype(int)
    return X, y

def train_eval(df: pd.DataFrame) -> None:
    df = df.sort_values("game_date").reset_index(drop=True)
    X, y = build_features(df)
    n = len(df)
    print(f"\nTotal rows: {n}, Over rate: {y.mean():.3f}")

    if n < 30 or y.nunique() < 2:
        baseline = float(y.mean()) if n else 0.5
        print(f"Not enough variety yet. Baseline P(Over)={baseline:.3f}.")
        return

    split = int(n * 0.8)
    Xtr, ytr = X.iloc[:split], y.iloc[:split]
    Xte, yte = X.iloc[split:], y.iloc[split:]

    if ytr.nunique() < 2:
        baseline = float(y.mean())
        print(f"Train is single-class. Baseline P(Over)={baseline:.3f}.")
        return

    model = LogisticRegression(max_iter=300, class_weight="balanced")
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)

    acc = accuracy_score(yte, pred)
    try:
        auc = roc_auc_score(yte, proba)
    except ValueError:
        auc = float("nan")
    brier = brier_score_loss(yte, proba)
    print(f"Holdout — ACC: {acc:.3f}  AUC: {auc:.3f}  Brier: {brier:.3f}  (n_test={len(Xte)})")

    # Demo prediction (replace with your game inputs)
    demo = pd.DataFrame([{"Pts": 31, "Opp Pts": 27, "total_line": 61.5, "is_away": 0}])
    p_over = model.predict_proba(demo)[:, 1][0]
    print(f"Demo: Predicted P(Over) = {p_over:.3f}")

def main():
    df = collect_sec_dataset()
    if df.empty:
        print("No data collected. Try again later or add more teams.")
        return
    train_eval(df)

if __name__ == "__main__":
    main()

# ---------------- LATER (notes only) ----------------
# - Replace Pts/Opp Pts with pregame proxies (rolling team offense/defense, pace, weather).
# - Expand to all FBS teams (build a slug list or scrape a team index).
# - Add TimeSeriesSplit once you have 100+ rows.
# - Try LightGBM/XGBoost for nonlinearity.
# - If you want clean historical data by season (and lines), switch to the CFBD API.
