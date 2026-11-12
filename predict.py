"""
Game predictions: win probability, predicted margin and total for every NFL game.

Every feature uses only what was known BEFORE kickoff (stats from earlier weeks, the Vegas line,
the injury report, the weather forecast). The model is trained on past seasons and saved to
model.json; predicting a week only needs the current season's data.

    python predict.py --train            # build features for 2016-2025, backtest, save model.json
    python predict.py                    # predict the next week's games
    python predict.py --week 10          # predict a specific week of the current season
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

import qb_rank as qr
import stadiums as st

qr.URLS.setdefault("injuries", qr.REL + "/injuries/injuries_{season}.csv")
qr.URLS.setdefault("depth", qr.REL + "/depth_charts/depth_charts_{season}.csv")

MODEL_FILE = "model.json"
FEATURE_VERSION = 4   # 4: no Next Gen Stats from later weeks in backtest features
FIRST_TRAIN_SEASON = 2016
PRIOR_GAMES = 4          # last season counts as this many games early in a season
BACKUP_QB_SCORE = 40     # a QB with no starts on record is rated like a typical backup

# ------------------------------------------------------------------- feature list + plain-English labels
VEGAS_FEATURES = ["spread", "total_line", "ml_home_prob"]
FEATURES = VEGAS_FEATURES + [
    "home_field", "rest_diff", "div_game", "primetime", "dome", "turf",
    "away_travel_miles", "away_tz_shift", "west_early",
    "temp", "wind", "rain",
    "coach_diff", "ref_home_win", "ref_points", "ref_penalties",
    "h2h", "revenge_home", "revenge_away",
    "adj_off_epa_diff", "adj_def_epa_diff", "adj_off_pass_diff", "adj_def_pass_diff",
    "adj_off_rush_diff", "adj_def_rush_diff", "adj_off_sr_diff", "adj_def_sr_diff", "points_diff",
    "form_margin_diff", "form_epa_diff",
    "qb_diff", "weapons_diff", "run_game_diff", "pass_pro_diff", "pass_d_diff",
    "weapons_out_diff", "defenders_out_diff",
]
LABELS = {
    "spread": "Vegas spread", "total_line": "Vegas total", "ml_home_prob": "Vegas moneyline",
    "home_field": "home field", "rest_diff": "rest advantage", "div_game": "division game",
    "primetime": "primetime game", "dome": "dome", "turf": "turf", "away_travel_miles": "opponent's travel",
    "away_tz_shift": "time-zone change", "west_early": "West Coast team in an early game",
    "temp": "temperature", "wind": "wind", "rain": "rain", "coach_diff": "head coach record",
    "ref_home_win": "referee crew's home record", "ref_points": "referee crew scoring", "ref_penalties": "referee penalties",
    "h2h": "head-to-head history", "revenge_home": "revenge game", "revenge_away": "revenge game",
    "off_epa_diff": "offense (EPA/play)", "def_epa_diff": "defense (EPA/play allowed)",
    "off_sr_diff": "offense success rate", "def_sr_diff": "defense success rate", "points_diff": "point differential",
    "adj_off_epa_diff": "offense (adjusted EPA)", "adj_def_epa_diff": "defense (adjusted EPA allowed)",
    "adj_off_pass_diff": "passing offense", "adj_def_pass_diff": "pass defense (EPA)", "adj_off_rush_diff": "rushing offense",
    "adj_def_rush_diff": "run defense", "adj_off_sr_diff": "offense success rate", "adj_def_sr_diff": "defense success rate",
    "form_margin_diff": "recent form (last 3)", "form_epa_diff": "recent form (EPA, last 3)",
    "qb_diff": "QB rating", "weapons_diff": "weapons", "run_game_diff": "run game", "pass_pro_diff": "pass protection",
    "pass_d_diff": "pass defense", "weapons_out_diff": "missing key weapons", "defenders_out_diff": "missing key defenders",
}


# ------------------------------------------------------------------- data helpers
def _csv(key: str, season: int, refresh=False, **kw) -> pd.DataFrame:
    try:
        return pd.read_csv(qr.fetch(key, season, refresh), low_memory=False, **kw)
    except Exception:
        return pd.DataFrame()


def all_games(refresh=False) -> pd.DataFrame:
    """Every game since 1999 (the schedule file), including upcoming ones."""
    g = pd.read_csv(qr.fetch("games", 0, refresh), low_memory=False)
    g = g[g["game_type"] == "REG"].copy()
    g["kickoff"] = pd.to_datetime(g["gameday"] + " " + g["gametime"].fillna("13:00"), errors="coerce")
    return g.sort_values("kickoff").reset_index(drop=True)


def _implied(ml):
    """Moneyline -> implied win probability (NaN if missing)."""
    try:
        ml = float(ml)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(ml) or ml == 0:
        return np.nan
    return -ml / (-ml + 100) if ml < 0 else 100 / (ml + 100)


# ------------------------------------------------------------------- weather (Open-Meteo, free, no key)
def _get_json(url: str, cache: Path | None, max_age_h: float):
    if cache and cache.exists() and (time.time() - cache.stat().st_mtime) < max_age_h * 3600:
        return json.loads(cache.read_text())
    req = urllib.request.Request(url, headers={"User-Agent": "qb-ranker"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read())
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data))
    return data


def rain_history(lat: float, lon: float, start: str, end: str) -> dict:
    """{date: inches of rain} at a stadium, from Open-Meteo's historical archive."""
    q = urllib.parse.urlencode({"latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
                                "daily": "precipitation_sum", "precipitation_unit": "inch", "timezone": "America/New_York"})
    cache = qr.CACHE / "weather" / f"archive_{lat:.3f}_{lon:.3f}_{start}_{end}.json"
    try:
        d = _get_json("https://archive-api.open-meteo.com/v1/archive?" + q, cache, 24 * 365)
        return dict(zip(d["daily"]["time"], d["daily"]["precipitation_sum"]))
    except Exception:
        return {}


def forecast(lat: float, lon: float, kickoff: pd.Timestamp) -> dict:
    """Forecast temp (F), wind (mph) at kickoff and rain (inches) for the day. Empty if too far ahead."""
    day = kickoff.strftime("%Y-%m-%d")
    q = urllib.parse.urlencode({"latitude": lat, "longitude": lon, "start_date": day, "end_date": day,
                                "hourly": "temperature_2m,wind_speed_10m", "daily": "precipitation_sum",
                                "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
                                "precipitation_unit": "inch", "timezone": "America/New_York"})
    cache = qr.CACHE / "weather" / f"forecast_{lat:.3f}_{lon:.3f}_{day}.json"
    try:
        d = _get_json("https://api.open-meteo.com/v1/forecast?" + q, cache, 3)
        hrs = d["hourly"]["time"]
        i = min(range(len(hrs)), key=lambda j: abs(pd.Timestamp(hrs[j]) - kickoff.floor("h")))
        return {"temp": d["hourly"]["temperature_2m"][i], "wind": d["hourly"]["wind_speed_10m"][i],
                "rain": d["daily"]["precipitation_sum"][0]}
    except Exception:
        return {}


# ------------------------------------------------------------------- history-based features (all seasons)
def coach_records(g: pd.DataFrame) -> pd.DataFrame:
    """Each coach's career win % going into each game (pulled toward .500 by 5 wins + 5 losses)."""
    done = g[g["home_score"].notna()]
    rows = []
    for side, other in (("home", "away"), ("away", "home")):
        d = done[["game_id", "kickoff", f"{side}_coach", f"{side}_score", f"{other}_score"]].copy()
        d.columns = ["game_id", "kickoff", "coach", "pf", "pa"]
        d["w"] = (d["pf"] > d["pa"]) + 0.5 * (d["pf"] == d["pa"])
        rows.append(d)
    h = pd.concat(rows).sort_values("kickoff")
    h["g_before"] = h.groupby("coach").cumcount()
    h["w_before"] = h.groupby("coach")["w"].cumsum() - h["w"]
    return h


def coach_lookup(hist: pd.DataFrame) -> dict:
    """{coach: (sorted kickoff times, running win total)} for fast career-record lookups."""
    return {c: (d["kickoff"].to_numpy(), d["w"].cumsum().to_numpy()) for c, d in hist.groupby("coach")}


def coach_pct(by: dict, coach, when) -> float:
    """Career win % before `when` (pulled toward .500 by 5 wins + 5 losses)."""
    if coach not in by:
        return 0.5
    times, cum = by[coach]
    i = int(np.searchsorted(times, np.datetime64(when), side="left"))
    w = cum[i - 1] if i > 0 else 0.0
    return (w + 5) / (i + 10)


def referee_table(g: pd.DataFrame, penalties: pd.Series) -> pd.DataFrame:
    done = g[g["home_score"].notna() & g["referee"].notna()].copy()
    done["home_win"] = (done["home_score"] > done["away_score"]) + 0.5 * (done["home_score"] == done["away_score"])
    done["points"] = done["home_score"] + done["away_score"]
    done["penalties"] = done["game_id"].map(penalties)
    return done[["game_id", "kickoff", "referee", "home_win", "points", "penalties"]]


def ref_features(refs: pd.DataFrame, ref, when, lg) -> tuple[float, float, float]:
    if pd.isna(ref):
        return lg
    d = refs[(refs["referee"] == ref) & (refs["kickoff"] < when)]
    n, k = len(d), 20  # regress toward league average with 20 games' worth
    if n == 0:
        return lg
    pen = d["penalties"].dropna()
    return ((d["home_win"].sum() + k * lg[0]) / (n + k),
            (d["points"].sum() + k * lg[1]) / (n + k),
            (pen.sum() + k * lg[2]) / (len(pen) + k))


def h2h_features(g: pd.DataFrame, home, away, when) -> tuple[float, int, int]:
    d = g[g["home_score"].notna() & (g["kickoff"] < when)
          & (((g["home_team"] == home) & (g["away_team"] == away)) | ((g["home_team"] == away) & (g["away_team"] == home)))]
    d = d.tail(5)
    if d.empty:
        return 0.5, 0, 0
    home_won = np.where(d["home_team"] == home, d["home_score"] > d["away_score"], d["away_score"] > d["home_score"])
    last = d.iloc[-1]
    last_home_won = (last["home_score"] > last["away_score"]) if last["home_team"] == home else (last["away_score"] > last["home_score"])
    tie = last["home_score"] == last["away_score"]
    return (home_won.sum() + 1) / (len(d) + 2), int(not last_home_won and not tie), int(last_home_won and not tie)


# ------------------------------------------------------------------- team strength from play-by-play
def team_games(season: int, refresh=False) -> pd.DataFrame:
    """Per team-game: offensive/defensive EPA per play, success rate, points; plus penalties per game."""
    cols = ["game_id", "week", "season_type", "posteam", "defteam", "play_type", "epa", "success", "penalty"]
    p = _csv("pbp", season, refresh, usecols=cols)
    if p.empty:
        return pd.DataFrame()
    p = p[p["season_type"] == "REG"]
    pen = p.groupby("game_id")["penalty"].sum()
    p = p[p["play_type"].isin(["pass", "run"]) & p["epa"].notna()]
    off = p.groupby(["game_id", "week", "posteam"]).agg(off_epa=("epa", "mean"), off_sr=("success", "mean")).reset_index()
    de = p.groupby(["game_id", "week", "defteam"]).agg(def_epa=("epa", "mean"), def_sr=("success", "mean")).reset_index()
    t = off.rename(columns={"posteam": "team"}).merge(de.rename(columns={"defteam": "team"}), on=["game_id", "week", "team"])
    t.attrs["penalties"] = pen
    return t


def strength_before(tg: pd.DataFrame, prev_tg: pd.DataFrame, g: pd.DataFrame, team: str, week: int) -> dict:
    """Season-to-date strength before `week`, blended with last season (PRIOR_GAMES games' worth)."""
    cur = tg[(tg["team"] == team) & (tg["week"] < week)]
    prv = prev_tg[prev_tg["team"] == team] if len(prev_tg) else prev_tg
    out = {}
    for c in ["off_epa", "def_epa", "off_sr", "def_sr"]:
        lg = tg[c].mean() if len(tg) else 0
        pv = prv[c].mean() if len(prv) else lg
        n = len(cur)
        out[c] = (cur[c].sum() + PRIOR_GAMES * pv) / (n + PRIOR_GAMES) if n else pv
    # points per game & recent form from the schedule
    mine = g[g["home_score"].notna() & (g["week"] < week) & ((g["home_team"] == team) | (g["away_team"] == team))]
    margin = np.where(mine["home_team"] == team, mine["home_score"] - mine["away_score"], mine["away_score"] - mine["home_score"])
    out["points"] = float(np.mean(margin)) if len(margin) else 0.0
    out["form_margin"] = float(np.mean(margin[-3:])) if len(margin) else 0.0
    last3 = cur.tail(3)
    out["form_epa"] = float((last3["off_epa"] - last3["def_epa"]).mean()) if len(last3) else 0.0
    return out



# ------------------------------------------------------------------- opponent-adjusted, recency-weighted ratings
RATING_HALF_LIFE = 6     # weeks: a game 6 weeks ago counts half as much as this week's
OFFSEASON_GAP = 8        # last season's games count as if this many extra weeks older (rosters change)
RATING_RIDGE = 400       # plays of "average" blended into each team's rating (stops wild early-season swings)
GARBAGE_WP = (0.10, 0.90)  # plays outside this win-probability range (game decided) are left out


def team_plays(season: int, refresh=False) -> pd.DataFrame:
    """Per (game, offense, defense): plays, EPA, success, pass/run split - garbage time removed."""
    cache = qr.CACHE / f"team_plays_{season}.csv"
    if cache.exists() and not refresh and season < qr.current_season():
        return pd.read_csv(cache)
    cols = ["game_id", "week", "season_type", "posteam", "defteam", "home_team", "play_type", "epa", "success", "wp"]
    p = _csv("pbp", season, refresh, usecols=cols)
    if p.empty:
        return pd.DataFrame()
    p = p[(p["season_type"] == "REG") & p["play_type"].isin(["pass", "run"]) & p["epa"].notna()]
    p = p[p["wp"].between(*GARBAGE_WP)]
    p = p.assign(is_pass=(p["play_type"] == "pass").astype(int))
    p["pass_epa"], p["rush_epa"] = p["epa"] * p["is_pass"], p["epa"] * (1 - p["is_pass"])
    g = p.groupby(["game_id", "week", "posteam", "defteam", "home_team"]).agg(
        n=("epa", "size"), epa=("epa", "sum"), sr=("success", "sum"), n_pass=("is_pass", "sum"),
        pass_epa=("pass_epa", "sum"), rush_epa=("rush_epa", "sum")).reset_index()
    g["n_rush"] = g["n"] - g["n_pass"]
    g["is_home"] = (g["posteam"] == g["home_team"]).astype(int)
    g["season"] = season
    if season < qr.current_season():
        g.to_csv(cache, index=False)
    return g


def _solve_ratings(rows: pd.DataFrame, value: str, count: str, teams: list[str]) -> tuple[dict, dict]:
    """Weighted ridge fit of  per-play value = average + offense_team + defense_opponent + home  ->  (off, def)."""
    r = rows[rows[count] > 0]
    if r.empty:
        return {}, {}
    idx = {t: i for i, t in enumerate(teams)}
    n_t = len(teams)
    y = (r[value] / r[count]).to_numpy(float)
    w = (r[count] * r["decay"]).to_numpy(float)
    X = np.zeros((len(r), 2 * n_t + 2))
    X[:, 0] = 1
    X[:, 1] = r["is_home"].to_numpy(float)
    X[np.arange(len(r)), 2 + r["posteam"].map(idx).to_numpy()] = 1
    X[np.arange(len(r)), 2 + n_t + r["defteam"].map(idx).to_numpy()] = 1
    reg = np.full(X.shape[1], float(RATING_RIDGE))
    reg[:2] = 0
    A = X.T @ (X * w[:, None]) + np.diag(reg)
    beta = np.linalg.solve(A, X.T @ (w * y))
    off = {t: beta[2 + i] for t, i in idx.items()}
    de = {t: beta[2 + n_t + i] for t, i in idx.items()}
    return off, de


def ratings_by_week(season: int, refresh=False) -> pd.DataFrame:
    """Each team's opponent-adjusted offense and defense ratings GOING INTO every week of a season
    (EPA per play, pass EPA, run EPA, success rate). Defense values: higher = allows more = worse."""
    cache = qr.CACHE / f"ratings_v1_{season}.csv"
    if cache.exists() and not refresh and season < qr.current_season():
        return pd.read_csv(cache)
    cur, prev = team_plays(season, refresh), team_plays(season - 1, refresh)
    if cur.empty and prev.empty:
        return pd.DataFrame()
    teams = sorted(set(cur.get("posteam", [])) | set(cur.get("defteam", [])) | set(prev.get("posteam", []))
                   | set(prev.get("defteam", [])))
    out = []
    last_week = int(cur["week"].max()) if len(cur) else 0
    for w in range(1, max(last_week + 2, 19)):
        a = cur[cur["week"] < w].copy() if len(cur) else cur
        b = prev.copy() if len(prev) else prev
        if len(a):
            a["decay"] = 0.5 ** ((w - a["week"]) / RATING_HALF_LIFE)
        if len(b):
            b["decay"] = 0.5 ** ((w + OFFSEASON_GAP + (18 - b["week"])) / RATING_HALF_LIFE)
        rows = pd.concat([x for x in (a, b) if len(x)], ignore_index=True)
        res = {}
        for name, value, count in (("epa", "epa", "n"), ("pass", "pass_epa", "n_pass"),
                                   ("rush", "rush_epa", "n_rush"), ("sr", "sr", "n")):
            res[name] = _solve_ratings(rows, value, count, teams)
        for t in teams:
            out.append({"week": w, "team": t,
                        **{f"adj_off_{k}": res[k][0].get(t, 0.0) for k in res},
                        **{f"adj_def_{k}": res[k][1].get(t, 0.0) for k in res}})
    df = pd.DataFrame(out)
    if season < qr.current_season():
        df.to_csv(cache, index=False)
    return df


RATING_FEATURES = ["adj_off_epa_diff", "adj_def_epa_diff", "adj_off_pass_diff", "adj_def_pass_diff",
                   "adj_off_rush_diff", "adj_def_rush_diff", "adj_off_sr_diff", "adj_def_sr_diff"]


def add_ratings(f: pd.DataFrame, refresh=False) -> pd.DataFrame:
    """Attach home-minus-away opponent-adjusted ratings (as of each game's week) to a feature table."""
    f = f.drop(columns=[c for c in RATING_FEATURES if c in f], errors="ignore").copy()
    parts = []
    for s, grp in f.groupby("season"):
        r = ratings_by_week(int(s), refresh)
        if r.empty:
            for c in RATING_FEATURES:
                grp[c] = 0.0
            parts.append(grp)
            continue
        h = grp.merge(r.add_prefix("h_"), left_on=["week", "home_team"], right_on=["h_week", "h_team"], how="left")
        h = h.merge(r.add_prefix("a_"), left_on=["week", "away_team"], right_on=["a_week", "a_team"], how="left")
        for side in ("off", "def"):
            for k in ("epa", "pass", "rush", "sr"):
                h[f"adj_{side}_{k}_diff"] = (h[f"h_adj_{side}_{k}"] - h[f"a_adj_{side}_{k}"]).fillna(0.0)
        h.index = grp.index
        parts.append(h[list(grp.columns) + RATING_FEATURES])
    return pd.concat(parts).loc[f.index]



# ------------------------------------------------------------------- turnovers: tendency, cost (when/where), response
DRIVE_POINTS = {"Touchdown": 7, "Field goal": 3, "Opp touchdown": -7, "Safety": -2}
TURNOVER_FEATURES = ["to_give_diff", "to_take_diff", "fumble_luck_diff", "to_cost_diff",
                     "sudden_change_diff", "takeaway_conv_diff"]
USE_TURNOVERS = False   # set by the backtest experiment: only on if it makes predictions better


def team_turnovers(season: int, refresh=False) -> pd.DataFrame:
    """Per (game, team): drives, giveaways/takeaways, fumbles, win-prob cost of its giveaways (which
    reflects WHEN and WHERE they happened), and what happened on the drive right after each turnover."""
    cache = qr.CACHE / f"turnovers_v1_{season}.csv"
    if cache.exists() and not refresh and season < qr.current_season():
        return pd.read_csv(cache)
    cols = ["game_id", "week", "season_type", "posteam", "defteam", "fixed_drive", "fixed_drive_result",
            "ep", "wpa", "interception", "fumble", "fumble_lost", "play_type"]
    p = _csv("pbp", season, refresh, usecols=cols)
    if p.empty:
        return pd.DataFrame()
    p = p[(p["season_type"] == "REG") & p["posteam"].notna() & p["fixed_drive"].notna()]
    # each drive: who had it, how it started (expected points) and how many points it produced
    d = p.groupby(["game_id", "fixed_drive"]).agg(week=("week", "first"), team=("posteam", "first"),
                                                   opp=("defteam", "first"), result=("fixed_drive_result", "first"),
                                                   start_ep=("ep", "first")).reset_index()
    d["points"] = d["result"].map(DRIVE_POINTS).fillna(0)
    d["over_exp"] = d["points"] - d["start_ep"].fillna(0)
    nxt = d[["game_id", "fixed_drive", "team", "over_exp"]].copy()
    nxt["fixed_drive"] -= 1                                    # key it by the PREVIOUS drive number
    nxt = nxt.rename(columns={"team": "next_team", "over_exp": "next_over_exp"})
    to = p[(p["interception"] == 1) | (p["fumble_lost"] == 1)][["game_id", "week", "fixed_drive", "posteam", "defteam", "wpa"]]
    to = to.merge(nxt, on=["game_id", "fixed_drive"], how="left")
    to = to[to["next_team"] == to["defteam"]]                  # the next possession went to the team that took it away
    give = to.groupby(["game_id", "posteam"]).agg(giveaways=("wpa", "size"), give_wpa=("wpa", "sum"),
                                                   sc_def=("next_over_exp", "sum")).reset_index().rename(columns={"posteam": "team"})
    take = to.groupby(["game_id", "defteam"]).agg(takeaways=("wpa", "size"), tk_conv=("next_over_exp", "sum")
                                                  ).reset_index().rename(columns={"defteam": "team"})
    fum = p[p["fumble"] == 1].groupby(["game_id", "posteam"]).agg(fumbles=("fumble", "sum"), fumbles_lost=("fumble_lost", "sum")
                                                                  ).reset_index().rename(columns={"posteam": "team"})
    drives = d.groupby(["game_id", "week", "team"]).size().rename("drives").reset_index()
    opp_drives = d.groupby(["game_id", "opp"]).size().rename("opp_drives").reset_index().rename(columns={"opp": "team"})
    t = drives.merge(opp_drives, on=["game_id", "team"], how="left")
    for x in (give, take, fum):
        t = t.merge(x, on=["game_id", "team"], how="left")
    t = t.fillna(0)
    t["season"] = season
    if season < qr.current_season():
        t.to_csv(cache, index=False)
    return t


def turnovers_by_week(season: int, refresh=False) -> pd.DataFrame:
    """Each team's turnover profile GOING INTO every week, recency-weighted (like the team ratings) and
    pulled toward the league average for small samples."""
    cache = qr.CACHE / f"turnover_ratings_v1_{season}.csv"
    if cache.exists() and not refresh and season < qr.current_season():
        return pd.read_csv(cache)
    cur, prev = team_turnovers(season, refresh), team_turnovers(season - 1, refresh)
    both = pd.concat([x for x in (cur, prev) if len(x)], ignore_index=True)
    if both.empty:
        return pd.DataFrame()
    lg_give = both["giveaways"].sum() / max(both["drives"].sum(), 1)
    lg_cost = both["give_wpa"].sum() / max(both["giveaways"].sum(), 1)
    teams = sorted(both["team"].unique())
    out = []
    last_week = int(cur["week"].max()) if len(cur) else 0
    for w in range(1, max(last_week + 2, 19)):
        a = cur[cur["week"] < w].copy() if len(cur) else cur
        b = prev.copy() if len(prev) else prev
        if len(a):
            a["decay"] = 0.5 ** ((w - a["week"]) / RATING_HALF_LIFE)
        if len(b):
            b["decay"] = 0.5 ** ((w + OFFSEASON_GAP + (18 - b["week"])) / RATING_HALF_LIFE)
        rows = pd.concat([x for x in (a, b) if len(x)], ignore_index=True)
        cols = ["drives", "opp_drives", "giveaways", "takeaways", "give_wpa", "sc_def", "tk_conv", "fumbles", "fumbles_lost"]
        s = rows[cols].mul(rows["decay"], axis=0).groupby(rows["team"]).sum()
        for t in teams:
            r = s.loc[t] if t in s.index else pd.Series(0.0, index=cols)
            out.append({"week": w, "team": t,
                        "give": (r["giveaways"] + 40 * lg_give) / (r["drives"] + 40),
                        "take": (r["takeaways"] + 40 * lg_give) / (r["opp_drives"] + 40),
                        "fumble_luck": (r["fumbles_lost"] + 5) / (r["fumbles"] + 10),     # share of own fumbles lost
                        "cost": (r["give_wpa"] + 8 * lg_cost) / (r["giveaways"] + 8),      # win prob lost per giveaway
                        "sudden": r["sc_def"] / (r["giveaways"] + 8),                       # opp. points over expected after own giveaway
                        "conv": r["tk_conv"] / (r["takeaways"] + 8)})                       # own points over expected after takeaway
    df = pd.DataFrame(out)
    if season < qr.current_season():
        df.to_csv(cache, index=False)
    return df


def add_turnovers(f: pd.DataFrame, refresh=False) -> pd.DataFrame:
    f = f.drop(columns=[c for c in TURNOVER_FEATURES if c in f], errors="ignore").copy()
    parts = []
    names = {"give": "to_give_diff", "take": "to_take_diff", "fumble_luck": "fumble_luck_diff", "cost": "to_cost_diff",
             "sudden": "sudden_change_diff", "conv": "takeaway_conv_diff"}
    for s, grp in f.groupby("season"):
        r = turnovers_by_week(int(s), refresh)
        if r.empty:
            for c in TURNOVER_FEATURES:
                grp[c] = 0.0
            parts.append(grp)
            continue
        h = grp.merge(r.add_prefix("h_"), left_on=["week", "home_team"], right_on=["h_week", "h_team"], how="left")
        h = h.merge(r.add_prefix("a_"), left_on=["week", "away_team"], right_on=["a_week", "a_team"], how="left")
        for k, col in names.items():
            h[col] = (h[f"h_{k}"] - h[f"a_{k}"]).fillna(0.0)
        h.index = grp.index
        parts.append(h[list(grp.columns) + TURNOVER_FEATURES])
    return pd.concat(parts).loc[f.index]



# ------------------------------------------------------------------- interception matchup: QB tendency vs defense tendency
INT_FEATURES = ["int_matchup_diff"]
USE_INT_MATCHUP = False      # set by the backtest: only on if it makes predictions better
INT_PRIOR_QB_ATT = 150       # attempts of league-average play blended into each QB's rate
INT_PRIOR_DEF_ATT = 200      # opponent attempts of league-average play blended into each defense's rate
INT_PREV_WEIGHT = 0.5        # last season counts half


def int_tables(season: int, refresh=False) -> dict:
    """Interceptions per attempt going into every week: each QB's (thrown) and each defense's (taken),
    plus each QB's usual attempts per game. Uses this season's earlier weeks + half of last season."""
    key = ("int_tables", season)
    if key in _INT_CACHE:
        return _INT_CACHE[key]
    def qb_rows(s):
        try:
            w = qr.load_data(s, refresh)["weekly"]
        except Exception:
            return pd.DataFrame()
        w = w[(w["position"] == "QB") & (w["attempts"].fillna(0) > 0)]
        return w[["player_id", "week", "team", "opponent_team", "attempts", "passing_interceptions"]].fillna(0)
    cur, prev = qb_rows(season), qb_rows(season - 1)
    lg = pd.concat([cur, prev])
    lg_rate = lg["passing_interceptions"].sum() / max(lg["attempts"].sum(), 1)
    pq = prev.groupby("player_id")[["attempts", "passing_interceptions"]].sum() * INT_PREV_WEIGHT
    pdf = prev.groupby("opponent_team")[["attempts", "passing_interceptions"]].sum() * INT_PREV_WEIGHT
    pg = prev.groupby("player_id").agg(att=("attempts", "sum"), g=("week", "nunique"))
    weeks = range(1, (int(cur["week"].max()) if len(cur) else 0) + 2)
    qb, de, att = {}, {}, {}
    for w in weeks:
        c = cur[cur["week"] < w]
        q = c.groupby("player_id")[["attempts", "passing_interceptions"]].sum().add(pq, fill_value=0)
        d = c.groupby("opponent_team")[["attempts", "passing_interceptions"]].sum().add(pdf, fill_value=0)
        qb[w] = ((q["passing_interceptions"] + INT_PRIOR_QB_ATT * lg_rate) / (q["attempts"] + INT_PRIOR_QB_ATT)).to_dict()
        de[w] = ((d["passing_interceptions"] + INT_PRIOR_DEF_ATT * lg_rate) / (d["attempts"] + INT_PRIOR_DEF_ATT)).to_dict()
        g = c.groupby("player_id").agg(att=("attempts", "sum"), g=("week", "nunique")).add(pg, fill_value=0)
        att[w] = ((g["att"] + 3 * 32) / (g["g"] + 3)).to_dict()     # attempts per game, pulled toward 32
    out = {"qb": qb, "def": de, "att": att, "league": lg_rate}
    _INT_CACHE[key] = out
    return out


_INT_CACHE: dict = {}


def expected_ints(t: dict, qb_id, defense: str, week: int) -> tuple[float, float, float]:
    """(expected interceptions this game, QB's INT rate, defense's INT rate)."""
    lg = t["league"] or 0.025
    w = min(week, max(t["qb"])) if t["qb"] else week
    q = t["qb"].get(w, {}).get(qb_id, lg)
    d = t["def"].get(w, {}).get(defense, lg)
    a = t["att"].get(w, {}).get(qb_id, 32.0)
    return q * d / lg * a, q, d


def add_int_matchup(f: pd.DataFrame, refresh=False) -> pd.DataFrame:
    """Expected interceptions for each QB against this defense; feature = home QB's minus away QB's."""
    f = f.copy()
    cols = {"home_exp_int": [], "away_exp_int": [], "home_qb_int_rate": [], "away_qb_int_rate": [],
            "home_def_int_rate": [], "away_def_int_rate": []}
    for r in f.itertuples(index=False):
        t = int_tables(int(r.season), refresh)
        he, hq, ad = expected_ints(t, r.home_qb, r.away_team, int(r.week))
        ae, aq, hd = expected_ints(t, r.away_qb, r.home_team, int(r.week))
        for k, v in (("home_exp_int", he), ("away_exp_int", ae), ("home_qb_int_rate", hq), ("away_qb_int_rate", aq),
                     ("home_def_int_rate", hd), ("away_def_int_rate", ad)):
            cols[k].append(v)
    for k, v in cols.items():
        f[k] = v
    f["int_matchup_diff"] = f["home_exp_int"] - f["away_exp_int"]
    return f


# ------------------------------------------------------------------- QB, supporting cast, injuries (from qb_rank)
def season_context(season: int, refresh=False, only_weeks=None):
    """Everything per season that needs qb_rank: data, week-by-week QB scores and team grades, defense ratings.
    only_weeks: just the weeks being predicted (much faster than the whole season)."""
    data = qr.load_data(season, refresh)
    cfg = qr.load_weights()
    log = qr.starter_log(data["games"], data["weekly"])
    weeks = sorted(int(w) for w in data["weekly"]["week"].unique())
    by_week = {}
    todo = sorted(only_weeks) if only_weeks else range(2, (max(weeks) if weeks else 1) + 2)
    for w in todo:
        if w < 2:
            continue
        try:
            q, _, ctx = qr.rank(data, cfg, min_starts=1, prior_starts=qr.DEFAULT_PRIOR_STARTS, weeks=(1, w - 1),
                                all_starters=True)
            by_week[w] = (q.set_index("player_id")["qb_score_exact"].to_dict(), ctx)
        except Exception:
            pass
    try:
        dr = qr.defense_ratings_by_week(data["weekly"], data["prev_weekly"], cfg.get("defense"))
        tough = dr.set_index(["week", "defense"])["tough"].to_dict()
    except Exception:
        tough = {}
    # last season's final QB scores (for week 1 and for QBs without starts yet)
    prev_qb = {}
    try:
        pd_ = qr.load_data(season - 1, refresh)
        prev_qb = qr.rank(pd_, cfg, min_starts=1, all_starters=True)[0].set_index("player_id")["qb_score_exact"].to_dict()
    except Exception:
        pass
    inj = _csv("injuries", season, refresh)
    if len(inj):
        inj = inj[(inj.get("game_type", "REG") == "REG") & inj["report_status"].isin(["Out", "Doubtful"])]
    kd = {}
    for w in (sorted(only_weeks) if only_weeks else range(1, (max(weeks) if weeks else 1) + 2)):
        try:
            kd[w] = qr.key_defenders({**data, "weekly_all": data["weekly"]}, w, 4)
        except Exception:
            kd[w] = pd.DataFrame(columns=["team", "player_id", "share"])
    return {"data": data, "cfg": cfg, "log": log, "by_week": by_week, "tough": tough, "prev_qb": prev_qb, "inj": inj,
            "key_def": kd}


def qb_score(sc: dict, qb_id, week: int) -> float:
    cur = sc["by_week"].get(week, ({}, None))[0]
    if qb_id in cur:
        return cur[qb_id]
    return sc["prev_qb"].get(qb_id, BACKUP_QB_SCORE)


def cast(sc: dict, team: str, week: int) -> tuple[float, float, float]:
    ctx = sc["by_week"].get(week, (None, None))[1]
    if ctx is None or team not in ctx.index:
        return 0.0, 0.0, 0.0
    r = ctx.loc[team]
    return float(r.get("weapons_z", 0)), float(r.get("run_game_z", 0)), float(r.get("pass_pro_z", 0))


def injured_share(sc: dict, team: str, week: int) -> tuple[float, float]:
    """Share of a team's key weapons and key defenders listed Out/Doubtful for this week."""
    inj = sc["inj"]
    if inj is None or not len(inj):
        return 0.0, 0.0
    out = set(inj[(inj["team"] == team) & (inj["week"] == week)]["gsis_id"])
    if not out:
        return 0.0, 0.0
    wk = sc["data"]["weekly"]
    pc = wk[(wk["team"] == team) & (wk["week"] < week) & wk["position"].isin(qr.PASS_CATCHERS)]
    share = pc.groupby("player_id")["targets"].sum()
    share = share / share.sum() if share.sum() else share
    key_w = share[share >= 0.10]
    w_out = float(key_w[key_w.index.isin(out)].sum())
    kd = sc["key_def"].get(week)
    kd = kd[kd["team"] == team] if kd is not None and len(kd) else None
    d_out = float(kd[kd["player_id"].isin(out)]["share"].sum()) if kd is not None else 0.0
    return w_out, d_out


def expected_starter(sc: dict, team: str, week: int, game_id=None, depth: pd.DataFrame | None = None):
    """Past game: the QB who actually started (box score). Upcoming: depth-chart QB1 unless he's
    listed Out/Doubtful, else the team's most recent starter."""
    log = sc["log"]
    if game_id is not None:
        r = log[(log["game_id"] == game_id) & (log["team"] == team)]
        if len(r):
            return r["player_id"].iloc[0]
    out = set()
    if sc["inj"] is not None and len(sc["inj"]):
        out = set(sc["inj"][(sc["inj"]["team"] == team) & (sc["inj"]["week"] == week)]["gsis_id"])
    if depth is not None and len(depth):
        qbs = depth[(depth["team"] == team) & (depth["pos_abb"] == "QB")].sort_values("pos_rank")
        for pid in qbs["gsis_id"]:
            if pid not in out:
                return pid
    prev = log[(log["team"] == team) & (log["week"] < week)].sort_values("week")
    return prev["player_id"].iloc[-1] if len(prev) else None


# ------------------------------------------------------------------- build features for a set of games
def build_features(season: int, games: pd.DataFrame, allg: pd.DataFrame, upcoming=False, refresh=False,
                   progress=None) -> pd.DataFrame:
    sc = season_context(season, refresh, only_weeks=set(int(w) for w in games["week"].unique()) if upcoming or
                        games["week"].nunique() <= 2 else None)
    tg = team_games(season, refresh)
    prev_tg = team_games(season - 1, refresh)
    pens = [t.attrs.get("penalties") for t in (tg, prev_tg) if len(t) and t.attrs.get("penalties") is not None]
    pen = pd.concat(pens) if pens else pd.Series(dtype=float)
    # penalties per game for referee history: this season + earlier seasons already cached
    refs = referee_table(allg, pen)
    lg = (refs["home_win"].mean() if len(refs) else 0.55, refs["points"].mean() if len(refs) else 44,
          refs["penalties"].mean() if refs["penalties"].notna().any() else 12)
    coaches = coach_lookup(coach_records(allg))
    depth = _csv("depth", season, refresh) if upcoming else None
    if depth is not None and len(depth) and "dt" in depth:
        depth = depth[depth["dt"] == depth["dt"].max()]

    # rain history for past games at each site
    rain = {}
    if not upcoming:
        for key, grp in games.groupby(["home_team", "stadium_id", "location", "stadium"], dropna=False):
            home, sid, loc, sname = key
            s = st.site(home, sid, loc == "Neutral", sname)
            if s[3]:
                continue
            days = rain_history(s[0], s[1], grp["gameday"].min(), grp["gameday"].max())
            for gid, day in zip(grp["game_id"], grp["gameday"]):
                rain[gid] = days.get(day)

    rows = []
    for i, r in enumerate(games.itertuples(index=False)):
        if progress:
            progress(i, len(games))
        w, home, away, when = int(r.week), r.home_team, r.away_team, r.kickoff
        neutral = r.location == "Neutral"
        s = st.site(home, r.stadium_id, neutral, r.stadium)
        indoor = s[3] or str(r.roof) in ("dome", "closed")
        ah = st.home_of(away)
        f = {"game_id": r.game_id, "season": season, "week": w, "home_team": home, "away_team": away,
             "kickoff": when, "home_score": r.home_score, "away_score": r.away_score}
        # Vegas
        f["spread"] = r.spread_line if pd.notna(r.spread_line) else 0.0
        f["total_line"] = r.total_line if pd.notna(r.total_line) else 44.0
        ph, pa = _implied(r.home_moneyline), _implied(r.away_moneyline)
        f["ml_home_prob"] = float(ph / (ph + pa)) if pd.notna(ph) and pd.notna(pa) and (ph + pa) > 0 else 0.5
        # situation
        f["home_field"] = 0 if neutral else 1
        f["rest_diff"] = (r.home_rest if pd.notna(r.home_rest) else 7) - (r.away_rest if pd.notna(r.away_rest) else 7)
        f["div_game"] = int(r.div_game == 1)
        gt = str(r.gametime or "13:00")
        f["primetime"] = int(r.weekday in ("Monday", "Thursday") or gt >= "19:00")
        f["dome"] = int(indoor)
        f["turf"] = int(str(r.surface).lower() not in ("grass", "nan", "") and "grass" not in str(r.surface).lower())
        f["away_travel_miles"] = st.miles(ah, s) / 1000
        f["away_tz_shift"] = abs(s[2] - ah[2])
        f["west_early"] = int(ah[2] <= -3 and s[2] >= -1 and gt < "14:00")
        # weather
        if indoor:
            f["temp"], f["wind"], f["rain"] = 70.0, 0.0, 0.0
        elif upcoming:
            fc = forecast(s[0], s[1], when) if pd.notna(when) else {}
            f["temp"], f["wind"], f["rain"] = fc.get("temp", 55.0), fc.get("wind", 7.0), fc.get("rain", 0.0)
        else:
            f["temp"] = r.temp if pd.notna(r.temp) else 55.0
            f["wind"] = r.wind if pd.notna(r.wind) else 7.0
            f["rain"] = rain.get(r.game_id) if rain.get(r.game_id) is not None else 0.0
        # coaches, referee, head-to-head
        f["coach_diff"] = coach_pct(coaches, r.home_coach, when) - coach_pct(coaches, r.away_coach, when)
        f["ref_home_win"], f["ref_points"], f["ref_penalties"] = ref_features(refs, r.referee, when, lg)
        f["h2h"], f["revenge_home"], f["revenge_away"] = h2h_features(allg, home, away, when)
        # team strength and form
        hs, as_ = strength_before(tg, prev_tg, sc["data"]["games"], home, w), strength_before(tg, prev_tg, sc["data"]["games"], away, w)
        f["off_epa_diff"] = hs["off_epa"] - as_["off_epa"]
        f["def_epa_diff"] = hs["def_epa"] - as_["def_epa"]   # lower is better for a defense
        f["off_sr_diff"] = hs["off_sr"] - as_["off_sr"]
        f["def_sr_diff"] = hs["def_sr"] - as_["def_sr"]
        f["points_diff"] = hs["points"] - as_["points"]
        f["form_margin_diff"] = hs["form_margin"] - as_["form_margin"]
        f["form_epa_diff"] = hs["form_epa"] - as_["form_epa"]
        # QB and supporting cast
        gid = None if upcoming else r.game_id
        hq, aq = expected_starter(sc, home, w, gid, depth), expected_starter(sc, away, w, gid, depth)
        f["home_qb"], f["away_qb"] = hq, aq
        f["home_qb_score"], f["away_qb_score"] = qb_score(sc, hq, w), qb_score(sc, aq, w)
        f["qb_diff"] = f["home_qb_score"] - f["away_qb_score"]
        hc, ac = cast(sc, home, w), cast(sc, away, w)
        f["weapons_diff"], f["run_game_diff"], f["pass_pro_diff"] = hc[0] - ac[0], hc[1] - ac[1], hc[2] - ac[2]
        f["pass_d_diff"] = sc["tough"].get((w, home), 0.0) - sc["tough"].get((w, away), 0.0)
        hw, hd = injured_share(sc, home, w)
        aw, ad = injured_share(sc, away, w)
        f["weapons_out_diff"], f["defenders_out_diff"] = hw - aw, hd - ad
        rows.append(f)
    return pd.DataFrame(rows)


def season_features(season: int, allg: pd.DataFrame, refresh=False, progress=None) -> pd.DataFrame:
    """Features for every completed regular-season game of a season (cached once a season is over)."""
    cache = qr.CACHE / f"features_v{FEATURE_VERSION}_{season}.csv"
    done = allg[(allg["season"] == season) & allg["home_score"].notna()]
    if cache.exists() and not refresh:
        f = pd.read_csv(cache, parse_dates=["kickoff"])
        if len(f) == len(done):
            return f
    f = build_features(season, done, allg, refresh=refresh, progress=progress)
    f.to_csv(cache, index=False)
    return f


# ------------------------------------------------------------------- models (numpy only)
def _design(df: pd.DataFrame, feats: list[str], mu=None, sd=None):
    X = df[feats].astype(float).fillna(0).to_numpy()
    if mu is None:
        mu, sd = X.mean(0), X.std(0)
        sd[sd == 0] = 1
    return np.column_stack([np.ones(len(X)), (X - mu) / sd]), mu, sd


def fit_logistic(X, y, lam=2.0, iters=50):
    w = np.zeros(X.shape[1])
    reg = np.full(X.shape[1], lam)
    reg[0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + np.diag(reg)
        g = X.T @ (y - p) - reg * w
        step = np.linalg.solve(H, g)
        w += step
        if np.abs(step).max() < 1e-7:
            break
    return w


def fit_ridge_w(X, y, lam, w):
    reg = np.full(X.shape[1], float(lam))
    reg[0] = 0
    return np.linalg.solve(X.T @ (X * w[:, None]) + np.diag(reg), X.T @ (w * y))


def season_weights(seasons: pd.Series, decay: float) -> np.ndarray:
    return decay ** (seasons.max() - seasons).to_numpy(float)


MODEL_SETS = {"full": lambda: FEATURES, "football": lambda: [f for f in FEATURES if f not in VEGAS_FEATURES]}


def fit_all(train: pd.DataFrame, params: dict | None = None) -> dict:
    """Two versions: 'full' (uses the Vegas line) and 'football' (football stats only, no Vegas).
    Each predicts the margin and total; win % comes from the predicted margin, so they always agree."""
    params = params or {}
    train = train[train["home_score"].notna()]
    margin = (train["home_score"] - train["away_score"]).to_numpy(float)
    total = (train["home_score"] + train["away_score"]).to_numpy(float)
    out = {}
    for name, feats in MODEL_SETS.items():
        feats = feats()
        p = params.get(name, {})
        w = season_weights(train["season"], p.get("decay", 1.0))
        X, mu, sd = _design(train, feats)
        out[name] = {"features": feats, "mu": mu.tolist(), "sd": sd.tolist(),
                     "margin": fit_ridge_w(X, margin, p.get("lam", 10), w).tolist(),
                     "total": fit_ridge_w(X, total, p.get("lam_total", 30), w).tolist(),
                     "win_scale": p.get("win_scale", 0.15), "params": p}
    return out


def apply(model: dict, df: pd.DataFrame, which="full") -> pd.DataFrame:
    m = model[which]
    X, _, _ = _design(df, m["features"], np.array(m["mu"]), np.array(m["sd"]))
    margin = X @ np.array(m["margin"])
    return pd.DataFrame({"win_prob": 1 / (1 + np.exp(-m["win_scale"] * margin)), "margin": margin,
                         "total": X @ np.array(m["total"])}, index=df.index)


def _oof(data: pd.DataFrame, feats: list[str], target: np.ndarray, lam: float, decay: float) -> np.ndarray:
    """Out-of-fold predictions: each season predicted by a model trained on the other seasons."""
    pred = np.zeros(len(data))
    for s in sorted(data["season"].unique()):
        tr, te = (data["season"] != s).to_numpy(), (data["season"] == s).to_numpy()
        X, mu, sd = _design(data[tr], feats)
        Xt, _, _ = _design(data[te], feats, mu, sd)
        beta = fit_ridge_w(X, target[tr], lam, season_weights(data.loc[tr, "season"], decay))
        pred[te] = Xt @ beta
    return pred


def _logloss(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def evaluate(bt: pd.DataFrame) -> dict:
    """Backtest scorecard. bt needs p_full, m_full, m_foot plus results and the Vegas line."""
    g = bt[bt["home_score"] != bt["away_score"]]
    y = (g["home_score"] > g["away_score"]).astype(int).to_numpy()
    mg = (g["home_score"] - g["away_score"]).to_numpy()
    vegas_pick = np.where(g["spread"] == 0, g["ml_home_prob"] > 0.5, g["spread"] > 0)
    p = g["p_full"].to_numpy()
    # calibration: when the model says X%, does the pick win X% of the time?
    conf = np.maximum(p, 1 - p)
    won = np.where(p >= 0.5, y == 1, y == 0)
    bins = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.01]
    calib = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        k = (conf >= lo) & (conf < hi)
        if k.sum():
            calib.append({"range": f"{lo:.0%}-{min(hi, 1):.0%}", "games": int(k.sum()),
                          "predicted": float(conf[k].mean()), "actual": float(won[k].mean())})
    # against the spread: football-stats side vs the Vegas line (pushes don't count)
    a = bt.copy()
    a["edge"] = a["m_foot"] - a["spread"]
    a["cover_margin"] = (a["home_score"] - a["away_score"]) - a["spread"]
    a = a[a["cover_margin"] != 0]
    ats = {}
    for lab, thr in (("all", 0), ("edge_3", 3), ("edge_5", 5)):
        k = a["edge"].abs() >= max(thr, 0.0001)
        right = (np.sign(a.loc[k, "edge"]) == np.sign(a.loc[k, "cover_margin"]))
        ats[lab] = {"games": int(k.sum()), "wins": int(right.sum()), "pct": float(right.mean()) if k.sum() else 0.0}
    return {
        "games": int(len(g)),
        "model_accuracy": float(((p > 0.5) == y).mean()),
        "football_only_accuracy": float(((g["m_foot"] > 0) == y).mean()),
        "vegas_accuracy": float((vegas_pick == y).mean()),
        "model_brier": float(((p - y) ** 2).mean()), "vegas_brier": float(((g["ml_home_prob"] - y) ** 2).mean()),
        "model_logloss": _logloss(p, y), "vegas_logloss": _logloss(g["ml_home_prob"].to_numpy(), y),
        "model_margin_mae": float(np.abs(g["m_full"] - mg).mean()), "vegas_margin_mae": float(np.abs(g["spread"] - mg).mean()),
        "calibration": calib, "ats": ats,
        "by_season": {int(s): float(((gg["p_full"] > 0.5) == (gg["home_score"] > gg["away_score"])).mean())
                      for s, gg in g.groupby("season")},
    }


def tune(data: pd.DataFrame, log=print) -> dict:
    """Pick each model's settings by out-of-fold error: ridge strength, how much recent seasons count,
    and how sharply the predicted margin turns into a win %."""
    margin = (data["home_score"] - data["away_score"]).to_numpy(float)
    total = (data["home_score"] + data["away_score"]).to_numpy(float)
    y = (margin > 0).astype(float)
    nz = margin != 0
    params, oof = {}, {}
    for name, feats in MODEL_SETS.items():
        feats = feats()
        best = None
        for lam in (3, 10, 30, 100, 300, 1000):
            for decay in (1.0, 0.9, 0.8, 0.7):
                pr = _oof(data, feats, margin, lam, decay)
                err = float(np.mean((pr - margin) ** 2))
                if best is None or err < best[0]:
                    best = (err, lam, decay, pr)
        _, lam, decay, pr = best
        ks = np.linspace(0.04, 0.30, 53)
        k = float(ks[np.argmin([_logloss(1 / (1 + np.exp(-kk * pr[nz])), y[nz]) for kk in ks])])
        bt_total = min(((float(np.mean((_oof(data, feats, total, lt, decay) - total) ** 2)), lt)
                        for lt in (10, 30, 100, 300, 1000)))[1]
        params[name] = {"lam": lam, "decay": decay, "win_scale": k, "lam_total": bt_total}
        oof[name] = pr
        log(f"  {name}: ridge {lam}, season weight x{decay} per year back, win scale {k:.3f}, total ridge {bt_total}")
    return params, oof


# ------------------------------------------------------------------- training + backtest
def train(seasons: list[int] | None = None, refresh=False, log=print) -> dict:
    allg = all_games(refresh)
    cur = qr.current_season()
    seasons = seasons or list(range(FIRST_TRAIN_SEASON, cur))
    frames = []
    for s in seasons:
        t0 = time.time()
        f = season_features(s, allg, refresh)
        frames.append(f)
        log(f"  {s}: {len(f)} games ({time.time() - t0:.0f}s)")
    data = add_ratings(pd.concat(frames, ignore_index=True), refresh).reset_index(drop=True)

    log("Tuning settings on the backtest...")
    params, oof = tune(data, log)
    bt = data.copy()
    bt["m_full"], bt["m_foot"] = oof["full"], oof["football"]
    bt["p_full"] = 1 / (1 + np.exp(-params["full"]["win_scale"] * bt["m_full"]))
    metrics = evaluate(bt)

    previous = None
    try:
        previous = json.loads((qr.HERE / MODEL_FILE).read_text()).get("backtest")
    except Exception:
        pass
    model = fit_all(data, params)
    model.update({"version": FEATURE_VERSION, "trained_on": seasons, "trained": dt.date.today().isoformat(),
                  "backtest": metrics, "previous_backtest": previous, "labels": LABELS, "params": params})
    out = qr.HERE / MODEL_FILE
    out.write_text(json.dumps(model, indent=1))
    log(f"Saved {out}")
    return model


# ------------------------------------------------------------------- pick log: predictions saved before kickoff, graded after
def _pick_log_path() -> Path:
    return qr.CACHE / "pick_log.csv"


def log_picks(pred: pd.DataFrame) -> None:
    """Save predictions for games that haven't kicked off yet (the latest pre-kickoff prediction wins)."""
    now = pd.Timestamp.now()
    new = pred[pred["home_score"].isna() & (pred["kickoff"] > now)].copy()
    if new.empty:
        return
    new["season"] = qr.current_season()
    new["logged_at"] = now.isoformat(timespec="minutes")
    cols = ["season", "week", "game_id", "kickoff", "away_team", "home_team", "pick", "pick_prob", "pred_margin",
            "spread", "football_margin", "edge_vs_vegas", "logged_at"]
    path = _pick_log_path()
    old = pd.read_csv(path, parse_dates=["kickoff"]) if path.exists() else pd.DataFrame(columns=cols)
    old = old[~old["game_id"].isin(new["game_id"])]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([old, new[cols]], ignore_index=True).to_csv(path, index=False)


def graded_picks(season: int | None = None, refresh=False) -> pd.DataFrame:
    """Logged pre-kickoff picks joined with final scores."""
    path = _pick_log_path()
    if not path.exists():
        return pd.DataFrame()
    lg = pd.read_csv(path, parse_dates=["kickoff"])
    if season:
        lg = lg[lg["season"] == season]
    res = all_games(refresh)[["game_id", "home_score", "away_score"]]
    lg = lg.merge(res, on="game_id", how="left")
    lg = lg[lg["home_score"].notna()].copy()
    if lg.empty:
        return lg
    winner = np.where(lg["home_score"] > lg["away_score"], lg["home_team"],
                      np.where(lg["home_score"] < lg["away_score"], lg["away_team"], "TIE"))
    lg["correct"] = lg["pick"] == winner
    cover = (lg["home_score"] - lg["away_score"]) - lg["spread"]
    lg["ats_result"] = np.where(lg["spread"].isna() | lg["edge_vs_vegas"].isna(), "no line",
                                np.where(cover == 0, "push",
                                         np.where(np.sign(cover) == np.sign(lg["edge_vs_vegas"]), "win", "loss")))
    return lg


def pick_record(season: int | None = None) -> dict:
    g = graded_picks(season)
    if g.empty:
        return {"games": 0}
    big = g[g["edge_vs_vegas"].abs() >= 3]
    return {"games": len(g), "wins": int(g["correct"].sum()), "losses": int((~g["correct"]).sum()),
            "ats_edge3_wins": int((big["ats_result"] == "win").sum()), "ats_edge3_losses": int((big["ats_result"] == "loss").sum())}


# Related inputs overlap (e.g. pass defense and defensive EPA), so a single input's piece can look backwards.
# Key factors are shown per group, where the total is meaningful.
FACTOR_GROUPS = {
    "team strength": ["adj_off_epa_diff", "adj_def_epa_diff", "adj_off_pass_diff", "adj_def_pass_diff",
                      "adj_off_rush_diff", "adj_def_rush_diff", "adj_off_sr_diff", "adj_def_sr_diff",
                      "points_diff", "pass_d_diff"],
    "recent form": ["form_margin_diff", "form_epa_diff"],
    "QB": ["qb_diff"],
    "supporting cast": ["weapons_diff", "run_game_diff", "pass_pro_diff"],
    "home field & travel": ["home_field", "away_travel_miles", "away_tz_shift", "west_early"],
    "injuries": ["weapons_out_diff", "defenders_out_diff"],
    "coaching": ["coach_diff"],
    "rest & schedule": ["rest_diff", "div_game", "primetime"],
    "matchup history": ["h2h", "revenge_home", "revenge_away"],
    "weather & field": ["temp", "wind", "rain", "dome", "turf"],
    "referee": ["ref_home_win", "ref_points", "ref_penalties"],
}


def factors(model: dict, row: pd.Series, n=4, which="football") -> list[tuple[str, float]]:
    """Biggest reasons behind the predicted margin: (group, points toward the home team), compared with
    an average matchup."""
    m = model[which]
    z = {f: ((row[f] - mu) / sd if pd.notna(row[f]) else 0) for f, mu, sd in zip(m["features"], m["mu"], m["sd"])}
    coef = dict(zip(m["features"], m["margin"][1:]))
    grouped = {g: sum(coef.get(f, 0) * z.get(f, 0) for f in fs) for g, fs in FACTOR_GROUPS.items()}
    return sorted(grouped.items(), key=lambda kv: -abs(kv[1]))[:n]


# ------------------------------------------------------------------- predicting
def load_model() -> dict:
    for p in (qr.USER_DIR / MODEL_FILE, qr.BUNDLE / MODEL_FILE, qr.HERE / MODEL_FILE):
        if p.exists():
            return json.loads(p.read_text())
    raise FileNotFoundError("model.json not found - run: python predict.py --train")


def next_week(allg: pd.DataFrame, season: int) -> int | None:
    up = allg[(allg["season"] == season) & allg["home_score"].isna()]
    return int(up["week"].min()) if len(up) else None


def predict_week(season: int | None = None, week: int | None = None, refresh=False, model=None) -> pd.DataFrame:
    """Predictions for one week. Games already played show the actual result too."""
    model = model or load_model()
    allg = all_games(refresh)
    season = season or qr.current_season()
    week = week or next_week(allg, season)
    if week is None:
        return pd.DataFrame()
    games = allg[(allg["season"] == season) & (allg["week"] == week)]
    upcoming = games["home_score"].isna().any()
    f = add_ratings(build_features(season, games, allg, upcoming=bool(upcoming), refresh=refresh), refresh)
    f = add_int_matchup(f, refresh)   # shown on every prediction; only part of the model if USE_INT_MATCHUP
    full, foot = apply(model, f, "full"), apply(model, f, "football")
    names = qr.load_data(season, refresh)["weekly"].drop_duplicates("player_id").set_index("player_id")["player_display_name"]
    out = f[["game_id", "week", "kickoff", "away_team", "home_team", "home_score", "away_score", "spread", "total_line",
             "temp", "wind", "rain", "dome", "home_qb_score", "away_qb_score",
             "home_exp_int", "away_exp_int", "home_qb_int_rate", "away_qb_int_rate", "home_def_int_rate", "away_def_int_rate"]].copy()
    out["home_qb_name"] = f["home_qb"].map(names)
    out["away_qb_name"] = f["away_qb"].map(names)
    out["home_qb_ovr"], out["away_qb_ovr"] = qr.to_ovr(f["home_qb_score"]), qr.to_ovr(f["away_qb_score"])
    # Vegas only posts lines about a week ahead. Without one, the full model would read the missing line as a
    # pick'em, so those games use the football-stats model instead.
    raw = games.set_index("game_id")
    has_line = (f["game_id"].map(raw["spread_line"]).notna() & f["game_id"].map(raw["home_moneyline"]).notna()).to_numpy()
    out["has_line"] = has_line
    for col in ("win_prob", "margin", "total"):
        full[col] = np.where(has_line, full[col], foot[col])
    out.loc[~has_line, ["spread", "total_line"]] = np.nan
    # how far ahead: weeks between the last completed week and this one
    done = allg[(allg["season"] == season) & allg["home_score"].notna()]
    last_done = int(done["week"].max()) if len(done) else 0
    out["weeks_ahead"] = np.maximum(out["week"] - last_done, 0)
    out["home_win_prob"] = full["win_prob"].clip(0.02, 0.98)
    out["pick"] = np.where(out["home_win_prob"] >= 0.5, out["home_team"], out["away_team"])
    out["pick_prob"] = np.maximum(out["home_win_prob"], 1 - out["home_win_prob"])
    out["pred_margin"], out["pred_total"] = full["margin"], full["total"]
    out["pred_home_pts"] = ((out["pred_total"] + out["pred_margin"]) / 2).round(0)
    out["pred_away_pts"] = ((out["pred_total"] - out["pred_margin"]) / 2).round(0)
    # a score that rounds to a tie shows the picked team one point ahead
    tie = out["pred_home_pts"] == out["pred_away_pts"]
    out.loc[tie & (out["pick"] == out["home_team"]), "pred_home_pts"] += 1
    out.loc[tie & (out["pick"] == out["away_team"]), "pred_away_pts"] += 1
    out["football_margin"] = foot["margin"]
    out["edge_vs_vegas"] = (out["football_margin"] - out["spread"]).round(1)   # + = football stats like home more than Vegas
    # (NaN when there's no line yet)
    out["factors"] = [factors(model, f.loc[i]) for i in f.index]
    out["factors_text"] = [", ".join(f"{lab} ({r.home_team if v > 0 else r.away_team} {abs(v):.1f})" for lab, v in fx)
                           for fx, r in zip(out["factors"], out.itertuples())]
    out["vegas_line"] = ["No line yet" if pd.isna(r.spread) else
                         (f"{r.home_team} -{abs(r.spread):g}" if r.spread > 0 else f"{r.away_team} -{abs(r.spread):g}")
                         if r.spread else "PK" for r in out.itertuples()]
    # Vegas's pick: the favorite by the spread (a pick'em goes to whoever the moneyline favors)
    out["vegas_pick"] = np.where(~has_line, "", np.where(f["spread"] > 0, f["home_team"], np.where(
        f["spread"] < 0, f["away_team"], np.where(f["ml_home_prob"] >= 0.5, f["home_team"], f["away_team"]))))
    out["agrees_with_vegas"] = (out["pick"] == out["vegas_pick"]) | ~has_line
    played = out["home_score"].notna()
    out["result"] = np.where(played, out["away_team"] + " " + out["away_score"].fillna(0).astype(int).astype(str) + " - "
                             + out["home_team"] + " " + out["home_score"].fillna(0).astype(int).astype(str), "")
    out["correct"] = np.where(played, np.where(out["home_score"] > out["away_score"], out["home_team"], out["away_team"]) == out["pick"], None)
    out = out.sort_values("kickoff").reset_index(drop=True)
    if season == qr.current_season():
        try:
            log_picks(out)
        except Exception:
            pass
    return out


# ------------------------------------------------------------------- CLI
def main():
    p = argparse.ArgumentParser(description="NFL game predictions")
    p.add_argument("--train", action="store_true", help="build features for 2016-last season, backtest, save model.json")
    p.add_argument("--season", type=int, default=None)
    p.add_argument("--week", type=int, default=None)
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--record", action="store_true", help="grade the picks saved before kickoff this season")
    a = p.parse_args()
    if a.record:
        g = graded_picks(a.season or qr.current_season())
        if g.empty:
            sys.exit("No graded picks yet. Picks are saved whenever you predict a week before its games kick off.")
        print(f"Straight-up: {int(g['correct'].sum())}-{int((~g['correct']).sum())}")
        big = g[g["edge_vs_vegas"].abs() >= 3]
        print(f"Against the spread when football stats disagree with Vegas by 3+: "
              f"{int((big['ats_result'] == 'win').sum())}-{int((big['ats_result'] == 'loss').sum())}")
        return
    if a.train:
        m = train(refresh=a.refresh)
        b, old = m["backtest"], m.get("previous_backtest") or {}
        def cmp(key, fmt):
            return f"   (before: {format(old[key], fmt)})" if key in old else ""
        print(f"\nBacktest ({b['games']} games, each season predicted by a model trained on the others):")
        print(f"  picks correct            {b['model_accuracy']:.1%}{cmp('model_accuracy', '.1%')}   Vegas favorite {b['vegas_accuracy']:.1%}")
        print(f"  football stats only      {b['football_only_accuracy']:.1%}{cmp('football_only_accuracy', '.1%')}")
        print(f"  win-prob error (Brier)   {b['model_brier']:.4f}{cmp('model_brier', '.4f')}   Vegas {b['vegas_brier']:.4f}  (lower is better)")
        print(f"  log loss                 {b['model_logloss']:.4f}   Vegas {b['vegas_logloss']:.4f}  (lower is better)")
        print(f"  margin miss per game     {b['model_margin_mae']:.2f} pts{cmp('model_margin_mae', '.2f')}   Vegas {b['vegas_margin_mae']:.2f}")
        print("  calibration (pick confidence -> how often the pick won):")
        for c in b["calibration"]:
            print(f"     {c['range']:>9}: predicted {c['predicted']:.0%}, actual {c['actual']:.0%}  ({c['games']} games)")
        for lab, name in (("all", "every game"), ("edge_3", "stats disagree with Vegas by 3+"), ("edge_5", "disagree by 5+")):
            x = b["ats"][lab]
            print(f"  against the spread, {name}: {x['wins']}-{x['games'] - x['wins']} ({x['pct']:.1%}; 52.4% breaks even)")
        return
    pr = predict_week(a.season, a.week, a.refresh)
    if pr.empty:
        sys.exit("No games found for that week.")
    print(f"\n=== Week {int(pr['week'].iloc[0])} predictions ===\n")
    for r in pr.itertuples():
        when = r.kickoff.strftime("%a %b %d %I:%M %p") if pd.notna(r.kickoff) else ""
        wx = "dome" if r.dome else f"{r.temp:.0f}F, wind {r.wind:.0f} mph" + (f", rain {r.rain:.2f} in" if r.rain else "")
        print(f"{r.away_team} @ {r.home_team}  ({when}, {wx})")
        print(f"  QBs: {r.away_qb_name} ({r.away_qb_ovr}) vs {r.home_qb_name} ({r.home_qb_ovr})")
        print(f"  Interception matchup: {r.away_qb_name} ~{r.away_exp_int:.1f} (throws {r.away_qb_int_rate:.1%}, "
              f"{r.home_team} D picks off {r.home_def_int_rate:.1%}); {r.home_qb_name} ~{r.home_exp_int:.1f} "
              f"(throws {r.home_qb_int_rate:.1%}, {r.away_team} D picks off {r.away_def_int_rate:.1%})")
        print(f"  Pick: {r.pick} ({r.pick_prob:.0%})   Vegas pick: {r.vegas_pick or 'none yet'}"
              f"{'' if r.agrees_with_vegas else '  <- disagrees'}   Predicted: {r.away_team} {r.pred_away_pts:.0f} - "
              f"{r.home_team} {r.pred_home_pts:.0f}   Vegas: {r.vegas_line}"
              + ("" if pd.isna(r.total_line) else f", total {r.total_line:g}"))
        lean = "no Vegas line yet (prediction uses the football stats only)" if pd.isna(r.edge_vs_vegas) else \
            "agrees with Vegas" if abs(r.edge_vs_vegas) < 0.5 else \
            f"{abs(r.edge_vs_vegas):.1f} pts more toward {r.home_team if r.edge_vs_vegas > 0 else r.away_team} than Vegas"
        print(f"  Football stats: {lean}")
        print(f"  Key factors: {r.factors_text}")
        if r.result:
            print(f"  Result: {r.result} -> {'correct' if r.correct else 'wrong'}")
        print()


if __name__ == "__main__":
    main()
