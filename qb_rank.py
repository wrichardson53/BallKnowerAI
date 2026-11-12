"""
QB Ranker - who is the true best quarterback in the NFL?

Only STARTING quarterbacks are ranked, and only the games they started count.
Each QB is graded on ~15 production metrics (Passing, Rushing, Efficiency, Ball
Security, Winning) and then adjusted for his supporting cast:

  * Weapons  - how good his WR/TE/RB pass catchers are, measured with Next Gen
               Stats that mostly reflect the receiver, not the QB: average
               separation from defenders and YAC over expected.
  * Run game - how effective the team's non-QB rushing attack is: rushing EPA per
               carry, yards per carry, and RB rush yards over expected.

A QB who produces with a weak cast gains points; one with an elite cast loses some.
Every weight lives in weights.json.

Usage:
    python qb_rank.py                        # current season
    python qb_rank.py --season 2025 --top 10
    python qb_rank.py --explain "Josh Allen"
    python qb_rank.py --teams                # supporting-cast grades for all 32 teams
    python qb_rank.py --csv rankings.csv
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
# Packaged as a standalone app (PyInstaller): bundled files live in a read-only temp folder, so downloaded
# data and the user's editable weights.json go in %LOCALAPPDATA%\QB Ranker instead.
FROZEN = getattr(sys, "frozen", False)
BUNDLE = Path(getattr(sys, "_MEIPASS", HERE))
USER_DIR = (Path(os.environ.get("LOCALAPPDATA", Path.home())) / "QB Ranker") if FROZEN else HERE
CACHE = USER_DIR / "data"


def weights_path() -> Path:
    """The weights.json to use. Packaged app: the user's copy (created from the bundled default on first run)."""
    p = USER_DIR / "weights.json"
    if FROZEN and not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text((BUNDLE / "weights.json").read_text(encoding="utf-8"), encoding="utf-8")
    return p
REL = "https://github.com/nflverse/nflverse-data/releases/download"
URLS = {
    "weekly": REL + "/stats_player/stats_player_week_{season}.csv",
    "games": "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv",
    "ngs_rec": REL + "/nextgen_stats/ngs_receiving.csv.gz",
    "ngs_rush": REL + "/nextgen_stats/ngs_rushing.csv.gz",
    "ngs_pass": REL + "/nextgen_stats/ngs_passing.csv.gz",
    "pfr_pass": REL + "/pfr_advstats/advstats_week_pass_{season}.csv",
    "pbp": REL + "/pbp/play_by_play_{season}.csv.gz",
    "ftn": REL + "/ftn_charting/ftn_charting_{season}.csv",
    "snaps": REL + "/snap_counts/snap_counts_{season}.csv",
    "season_stats": REL + "/stats_player/stats_player_reg_{season}.csv",
    "draft": REL + "/draft_picks/draft_picks.csv",
}
PASS_CATCHERS = ["WR", "TE", "RB"]
CACHE_HOURS = 12


# --------------------------------------------------------------------------- data
def fetch(key: str, season: int, refresh: bool) -> Path:
    url = URLS[key].format(season=season)
    name = url.rsplit("/", 1)[-1]
    dest = CACHE / name
    fresh = dest.exists() and (time.time() - dest.stat().st_mtime) < CACHE_HOURS * 3600
    if fresh and not refresh:
        return dest
    CACHE.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {name} ...", file=sys.stderr)
    req = urllib.request.Request(url, headers={"User-Agent": "qb-ranker"})
    with urllib.request.urlopen(req, timeout=120) as r:
        dest.write_bytes(r.read())
    return dest


def load_data(season: int, refresh: bool) -> dict[str, pd.DataFrame]:
    weekly = pd.read_csv(fetch("weekly", season, refresh), low_memory=False)
    weekly = weekly[weekly["season_type"] == "REG"]

    g = pd.read_csv(fetch("games", season, refresh), low_memory=False)
    g = g[(g["season"] == season) & (g["game_type"] == "REG") & g["home_score"].notna()]

    def ngs(key, weekly_rows=False):
        d = pd.read_csv(fetch(key, season, refresh), low_memory=False)
        # week 0 = full-season aggregate row per player; weeks 1+ = single games (qualifying players only).
        # NGS spells the Rams "LAR", other sources "LA".
        wk_filter = (d["week"] > 0) if weekly_rows else (d["week"] == 0)
        d = d[(d["season"] == season) & (d["season_type"] == "REG") & wk_filter].copy()
        d["team_abbr"] = d["team_abbr"].replace({"LAR": "LA"})
        return d

    # The two previous seasons, for each pass catcher's track record (any team, any QB).
    prev = []
    for s in (season - 1, season - 2):
        try:
            p = pd.read_csv(fetch("season_stats", s, refresh), low_memory=False)
        except Exception:  # season not published (e.g. very old) -> track record just uses fewer years
            continue
        prev.append(p[p["position"].isin(PASS_CATCHERS)].assign(season=s))
    prev = pd.concat(prev, ignore_index=True) if prev else pd.DataFrame()
    draft = pd.read_csv(fetch("draft", season, refresh), low_memory=False)

    # Pressures / hits / hurries / blitzes per QB per game (Pro Football Reference, 2018+), and NGS time to throw
    try:
        pfr_pass = pd.read_csv(fetch("pfr_pass", season, refresh), low_memory=False)
        pfr_pass = pfr_pass[pfr_pass["game_type"] == "REG"]
    except Exception:
        pfr_pass = pd.DataFrame()
    try:
        ngs_pass_weekly = ngs("ngs_pass", weekly_rows=True)
    except Exception:
        ngs_pass_weekly = pd.DataFrame()

    # Throw quality from FTN play charting (2022+): interception-worthy throws, credited to the passer via play-by-play
    throws = pd.DataFrame()
    try:
        pbp = pd.read_csv(fetch("pbp", season, refresh), low_memory=False,
                          usecols=["game_id", "play_id", "week", "season_type", "posteam", "passer_player_id",
                                   "pass_attempt", "sack", "interception"])
        pbp = pbp[(pbp["season_type"] == "REG") & (pbp["pass_attempt"] == 1) & (pbp["sack"] != 1)
                  & pbp["passer_player_id"].notna()]
        ftn = pd.read_csv(fetch("ftn", season, refresh), low_memory=False,
                          usecols=["nflverse_game_id", "nflverse_play_id", "is_interception_worthy"])
        x = pbp.merge(ftn, left_on=["game_id", "play_id"], right_on=["nflverse_game_id", "nflverse_play_id"], how="inner")
        x["worthy"] = x["is_interception_worthy"].astype(str).str.lower().eq("true")
        x["int"] = x["interception"].fillna(0).astype(int) == 1
        throws = (x.assign(charted=1, worthy_int=x["worthy"] & x["int"], worthy_dropped=x["worthy"] & ~x["int"],
                           int_not_worthy=x["int"] & ~x["worthy"])
                  .groupby(["game_id", "week", "posteam", "passer_player_id"])
                  [["charted", "worthy", "int", "worthy_int", "worthy_dropped", "int_not_worthy"]].sum().reset_index()
                  .rename(columns={"posteam": "team", "passer_player_id": "player_id"}))
    except Exception:  # no charting for this season (before 2022) -> neutral
        pass

    # Snap counts: who actually played each game (to spot missing key players)
    try:
        snaps = pd.read_csv(fetch("snaps", season, refresh), low_memory=False)
        snaps = snaps[snaps["game_type"] == "REG"]
    except Exception:
        snaps = pd.DataFrame()

    # Last season's games, so each defense has a starting rating before it has played this season
    try:
        pw = pd.read_csv(fetch("weekly", season - 1, refresh), low_memory=False)
        prev_weekly = pw[(pw["season_type"] == "REG") & (pw["position"] == "QB")]
    except Exception:
        prev_weekly = pd.DataFrame()

    return {"weekly": weekly, "games": g, "ngs_rec": ngs("ngs_rec"), "ngs_rush": ngs("ngs_rush"),
            "ngs_rec_weekly": ngs("ngs_rec", weekly_rows=True), "ngs_rush_weekly": ngs("ngs_rush", weekly_rows=True),
            "prev": prev, "draft": draft,
            "prev_weekly": prev_weekly, "pfr_pass": pfr_pass, "ngs_pass_weekly": ngs_pass_weekly, "throws": throws,
            "snaps": snaps}


def starter_log(games: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, team) with that team's QB of record and the result.

    The schedule's home_qb_id/away_qb_id is unreliable (e.g. it credits a backup with games the
    starter played start to finish), so the QB is taken from the box score: whoever had the most
    dropbacks (attempts + sacks) for that team in that game. The schedule is only a fallback.
    """
    rows = []
    for _, r in games.iterrows():
        diff = r["home_score"] - r["away_score"]
        for team, opp, qb, d in ((r["home_team"], r["away_team"], r["home_qb_id"], diff),
                                 (r["away_team"], r["home_team"], r["away_qb_id"], -diff)):
            rows.append({"game_id": r["game_id"], "week": r["week"], "team": team, "opp": opp, "sched_qb": qb,
                         "win": int(d > 0), "loss": int(d < 0), "tie": int(d == 0)})
    log = pd.DataFrame(rows)

    qb = weekly[weekly["position"] == "QB"].copy()
    qb["dropbacks"] = qb["attempts"].fillna(0) + qb["sacks_suffered"].fillna(0)
    top = qb.sort_values("dropbacks").groupby(["game_id", "team"]).tail(1)[["game_id", "team", "player_id"]]
    log = log.merge(top, on=["game_id", "team"], how="left")
    log["player_id"] = log["player_id"].fillna(log["sched_qb"])
    return log.drop(columns="sched_qb")


# ------------------------------------------------------------- supporting cast
def zs(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    return (s - s.mean()) / (sd if sd else 1.0)


def wavg(df: pd.DataFrame, val: str, w: str, by: str) -> pd.Series:
    d = df.dropna(subset=[val])
    return (d[val] * d[w]).groupby(d[by]).sum() / d[w].groupby(d[by]).sum()


# How the weapons grade is built. Relative weights; override with "weapon_parts" in weights.json.
WEAPON_PARTS = {
    "track_record": 25,      # receiving yards/game: this season (x3) + last 2 seasons (x2, x1), any team, target-weighted
    "track_efficiency": 10,  # receiving yards/target, same seasons and weighting
    "star_power": 10,        # the best pass catcher's track record (WR1 / elite TE)
    "depth": 5,              # the 3rd-best option's track record
    "accolades": 5,          # career Pro Bowls + 2 x All-Pros (target-weighted)
    "pedigree": 5,           # draft position (target-weighted)
    "form_adj": 10,          # this season: yds & EPA per target AFTER removing what the passer's accuracy (CPOE) explains
    "sep_adj": 15,           # separation vs. expected for route depth and cushion (this season, NGS)
    "yac_oe": 15,            # yards after catch over expected (this season, NGS)
}
WEAPON_LABELS = {
    "track_record": "Track record (yds/g)", "track_efficiency": "Efficiency (yds/target)",
    "star_power": "Star power (best option)", "depth": "Depth (3rd option)", "accolades": "Pro Bowls / All-Pros",
    "pedigree": "Draft pedigree", "form_adj": "This season's form (accuracy-adj)",
    "sep_adj": "Separation (depth-adj)", "yac_oe": "YAC over expected",
}
TRACK_SEASON_WEIGHTS = {0: 3, 1: 2, 2: 1}  # this season, last season, two seasons ago
TRACK_PRIOR_GAMES = 6     # weighted games of "typical player at his draft slot" blended into each track record
TRACK_PRIOR_TARGETS = 40  # weighted targets of "typical player at his position" blended into yards/target


def _round_bucket(r) -> str:
    return "UDFA" if pd.isna(r) else ("1" if r == 1 else "2" if r == 2 else "3" if r == 3 else "4+")


def receiver_talent(data: dict[str, pd.DataFrame], season: int) -> pd.DataFrame:
    """Receiving yards per game (and per target) each WR/TE/RB has produced this season and the
    two before it, on any team with any QB. This season counts 3x, last season 2x, two ago 1x.

    Players with little or no history (rookies, injuries) are blended toward what players at
    their position drafted in the same round produced as rookies.
    """
    prev, draft = data["prev"], data["draft"]
    rnd = draft.dropna(subset=["gsis_id"]).drop_duplicates("gsis_id").set_index("gsis_id")
    if prev.empty:
        return pd.DataFrame(columns=["talent_ypg"])

    prev = prev.copy()
    prev["w"] = (season - prev["season"]).map(TRACK_SEASON_WEIGHTS).fillna(0)
    prev["bucket"] = prev["player_id"].map(rnd["round"]).map(_round_bucket)

    # Draft-slot baseline: rookie-year yards/game by position and round
    rookies = prev[prev["player_id"].map(rnd["season"]) == prev["season"]]
    base = (rookies.groupby(["position", "bucket"])["receiving_yards"].sum()
            / rookies.groupby(["position", "bucket"])["games"].sum())
    pos_floor = prev.groupby("position")["receiving_yards"].sum() / prev.groupby("position")["games"].sum() / 2

    g = prev.assign(wy=prev["receiving_yards"].fillna(0) * prev["w"], wg=prev["games"].fillna(0) * prev["w"],
                    wt=prev["targets"].fillna(0) * prev["w"])
    hist = g.groupby("player_id").agg(wy=("wy", "sum"), wg=("wg", "sum"), wt=("wt", "sum"))

    # This season so far (weighted most heavily)
    cw = TRACK_SEASON_WEIGHTS[0]
    now = data["weekly"][data["weekly"]["position"].isin(PASS_CATCHERS)].groupby("player_id").agg(
        y=("receiving_yards", "sum"), n=("week", "nunique"), t=("targets", "sum"))
    hist = hist.add(pd.DataFrame({"wy": now["y"] * cw, "wg": now["n"] * cw, "wt": now["t"] * cw}), fill_value=0)
    pos_ypt = prev.groupby("position")["receiving_yards"].sum() / prev.groupby("position")["targets"].sum()

    # Everyone catching passes this season
    cur = data["weekly"][data["weekly"]["position"].isin(PASS_CATCHERS)]
    players = cur.groupby("player_id").agg(position=("position", "first"))
    players["bucket"] = players.index.map(rnd["round"]).map(_round_bucket)
    players = players.join(hist).fillna({"wy": 0, "wg": 0, "wt": 0})
    prior = [base.get((p, b), pos_floor.get(p, 10)) for p, b in zip(players["position"], players["bucket"])]
    players["prior_ypg"] = pd.Series(prior, index=players.index).fillna(10)
    k = TRACK_PRIOR_GAMES
    players["talent_ypg"] = (players["wy"] + k * players["prior_ypg"]) / (players["wg"] + k)
    kt = TRACK_PRIOR_TARGETS
    players["talent_ypt"] = (players["wy"] + kt * players["position"].map(pos_ypt).fillna(7)) / (players["wt"] + kt)

    # Compare each player with others at his POSITION (100 = average WR / TE / RB pass catcher), so a team
    # that uses its backs and tight ends in the passing game isn't penalized for their naturally lower yardage.
    tgt = cur.groupby("player_id")["targets"].sum().reindex(players.index).fillna(0)
    for col in ["talent_ypg", "talent_ypt"]:
        avg = (players[col] * tgt).groupby(players["position"]).sum() / tgt.groupby(players["position"]).sum()
        players[f"{col}_idx"] = players[col] / players["position"].map(avg) * 100

    # Honors and draft capital (players not in the draft table were undrafted)
    players["accolades"] = (players.index.map(rnd["probowls"]).fillna(0)
                            + 2 * players.index.map(rnd["allpro"]).fillna(0)).astype(float)
    pick = pd.Series(players.index.map(rnd["pick"]), index=players.index).astype(float)
    players["pedigree"] = (1 - np.log(pick) / np.log(262)).clip(lower=0).fillna(0)  # #1 pick = 1, last pick ~ 0
    return players


def depth_adjusted_separation(rec: pd.DataFrame) -> pd.Series:
    """Separation minus what's expected for how deep the routes are and how much cushion
    defenders give. Deep threats naturally get less separation; this removes that bias."""
    d = rec.dropna(subset=["avg_separation", "avg_intended_air_yards", "avg_cushion"])
    if len(d) < 5:
        return pd.Series(np.nan, index=rec.index)
    X = np.column_stack([np.ones(len(d)), d["avg_intended_air_yards"], d["avg_cushion"]])
    coef, *_ = np.linalg.lstsq(X, d["avg_separation"].to_numpy(), rcond=None)
    return pd.Series(d["avg_separation"].to_numpy() - X @ coef, index=d.index)


# ------------------------------------------------------------ pass protection (offensive line)
# How the pass-protection grade is built (relative weights); override with "pass_pro_parts" in weights.json.
PASS_PRO_PARTS = {
    "pressure_rate": 40,  # pressures allowed per dropback  (after removing what time-to-throw and blitzes explain)
    "sack_rate": 20,      # sacks per dropback               (same adjustment)
    "hit_rate": 15,       # QB hits per dropback             (same adjustment)
    "hurry_rate": 10,     # hurries per dropback             (same adjustment)
    "time_given": 15,     # estimated time the line gives before pressure arrives (see time_given())
}
PASS_PRO_LABELS = {"pressure_rate": "Pressure rate allowed", "sack_rate": "Sack rate allowed",
                   "hit_rate": "QB hit rate allowed", "hurry_rate": "Hurry rate allowed",
                   "time_given": "Time given by the line"}
_PP_COUNTS = {"pressure_rate": "times_pressured", "sack_rate": "times_sacked", "hit_rate": "times_hit",
              "hurry_rate": "times_hurried"}


def _residual_multi(y: pd.Series, xs: list[pd.Series], w: pd.Series) -> pd.Series:
    """y minus its weighted linear fit on several x's."""
    ok = y.notna() & (w > 0)
    for x in xs:
        ok &= x.notna()
    if ok.sum() < len(xs) + 2:
        return y - y.mean()
    sw = np.sqrt(w[ok].to_numpy())
    X = np.column_stack([np.ones(ok.sum())] + [x[ok].to_numpy() for x in xs])
    coef, *_ = np.linalg.lstsq(X * sw[:, None], y[ok].to_numpy() * sw, rcond=None)
    Xall = np.column_stack([np.ones(len(y))] + [x.to_numpy() for x in xs])
    return (y - Xall @ coef).where(ok)


def pass_protection_games(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per (game, team): dropbacks, pressures/sacks/hits/hurries/blitzes, time to throw,
    and each rate adjusted for time-to-throw and blitz rate (positive = more than expected)."""
    pfr, wk, ngs = data.get("pfr_pass", pd.DataFrame()), data["weekly"], data.get("ngs_pass_weekly", pd.DataFrame())
    if pfr.empty:
        return pd.DataFrame()
    cnt = pfr.groupby(["game_id", "team", "week"])[list(_PP_COUNTS.values()) + ["times_blitzed"]].sum()
    qb = wk[wk["position"] == "QB"]
    db = (qb["attempts"].fillna(0) + qb["sacks_suffered"].fillna(0)).groupby([qb["game_id"], qb["team"]]).sum()
    g = cnt.reset_index().join(db.rename("dropbacks"), on=["game_id", "team"])
    g = g[g["dropbacks"] > 0].copy()
    if len(ngs):
        n = ngs.assign(tx=ngs["avg_time_to_throw"] * ngs["attempts"])
        tt = n.groupby(["team_abbr", "week"])[["tx", "attempts"]].sum()
        g = g.join((tt["tx"] / tt["attempts"]).rename("time_to_throw"), on=["team", "week"])
    else:
        g["time_to_throw"] = np.nan
    g["time_to_throw"] = g["time_to_throw"].fillna(g.groupby("team")["time_to_throw"].transform("mean"))
    g["time_to_throw"] = g["time_to_throw"].fillna(g["time_to_throw"].mean())
    g["blitz_rate"] = g["times_blitzed"] / g["dropbacks"]
    for k, col in _PP_COUNTS.items():
        g[k] = g[col] / g["dropbacks"]
        g[f"{k}_adj"] = _residual_multi(g[k], [g["time_to_throw"], g["blitz_rate"]], g["dropbacks"])
    return g


def pressure_clock(g: pd.DataFrame) -> tuple[float, float]:
    """How pressure builds the longer a QB holds the ball, fit on every team-game this season.

    Model: P(pressured on a dropback) = 1 - exp(-(T / L)^k), where T = time to throw and L is how
    long that line holds up. Returns (k = how sharply pressure ramps up, q = league pressure rate).
    """
    p = g["pressure_rate"].clip(0.02, 0.9).to_numpy()
    T = g["time_to_throw"].to_numpy()
    w = np.sqrt(g["dropbacks"].to_numpy())
    ok = np.isfinite(p) & np.isfinite(T) & (T > 0)
    k = 1.8
    if ok.sum() > 20:
        A = np.column_stack([np.ones(ok.sum()), np.log(T[ok])])
        (_, slope), *_ = np.linalg.lstsq(A * w[ok, None], np.log(-np.log(1 - p[ok])) * w[ok], rcond=None)
        k = float(slope) if 0.5 < slope < 6 else k
    q = g["times_pressured"].sum() / g["dropbacks"].sum()
    return k, q


def time_given(pressure_rate, ttt, k: float, q: float):
    """Time the line gives (seconds): how long until pressure would reach the league-average rate,
    given this line's pressure rate at its QBs' time to throw. An average line gives about the
    league-average time to throw; a better line gives more."""
    p = np.clip(pressure_rate, 0.02, 0.9)
    return ttt * ((-np.log(1 - q)) / (-np.log(1 - p))) ** (1 / k)


def pass_protection(data: dict[str, pd.DataFrame], parts: dict | None = None) -> pd.DataFrame:
    """Team pass-protection grade from every dropback behind that line this season (all QBs).
    Higher = better protection. Pressure-type rates are judged after removing what the QBs' time
    to throw and the blitzes faced explain, so a QB who holds the ball doesn't sink his line."""
    parts = parts or PASS_PRO_PARTS
    g = pass_protection_games(data)
    if g.empty:
        return pd.DataFrame()
    w = g["dropbacks"]
    by = lambda c: (g[c] * w).groupby(g["team"]).sum() / w.groupby(g["team"]).sum()
    t = pd.DataFrame({c: by(c) for c in [*_PP_COUNTS, *[f"{k}_adj" for k in _PP_COUNTS], "time_to_throw", "blitz_rate"]})
    t["dropbacks"] = w.groupby(g["team"]).sum()
    k, q = pressure_clock(g)
    t["time_given"] = time_given(t["pressure_rate"], t["time_to_throw"], k, q)
    good = {}  # +z = better protection
    for k in _PP_COUNTS:
        good[k] = -zs(t[f"{k}_adj"])
    good["time_given"] = zs(t["time_given"])
    good["time_to_throw"] = good["time_given"]  # older weights.json files used this name
    tot = sum(parts.values()) or 1
    for k in parts:
        t[f"pp_{k}_grade"] = (50 + 10 * good[k]).round(0)
    t["pass_pro_z"] = zs(sum(wt / tot * good[k] for k, wt in parts.items()))
    t["pass_pro_grade"] = (50 + 10 * t["pass_pro_z"]).round(0).astype(int)
    return t


def team_game_receiving(wk: pd.DataFrame) -> pd.DataFrame:
    """Pass-catcher production per (game, team), and what's left after the passer's accuracy.

    ypt_res / ept_res = yards / EPA per target minus what the passers' CPOE in that game explains
    (a weighted fit across every team-game). Positive = receivers produced more than the
    accuracy of the throws they got would predict.
    """
    pc = (wk[wk["position"].isin(PASS_CATCHERS)].groupby(["game_id", "team", "week"])
          [["targets", "receiving_yards", "receiving_epa"]].sum().reset_index())
    qbs = wk[(wk["position"] == "QB") & (wk["attempts"] > 0)].copy()
    qbs["cx"] = qbs["passing_cpoe"] * qbs["attempts"]
    qbs["ca"] = qbs["attempts"].where(qbs["passing_cpoe"].notna(), 0)
    acc = qbs.groupby(["game_id", "team"])[["cx", "ca"]].sum()
    acc["cpoe"] = acc["cx"] / acc["ca"].replace(0, np.nan)
    g = pc.join(acc["cpoe"], on=["game_id", "team"])
    g["ypt"] = g["receiving_yards"] / g["targets"].replace(0, np.nan)
    g["ept"] = g["receiving_epa"] / g["targets"].replace(0, np.nan)
    g["ypt_res"] = _residual(g["ypt"], g["cpoe"], g["targets"])
    g["ept_res"] = _residual(g["ept"], g["cpoe"], g["targets"])
    return g


# Per-player weapons grade: the team parts that make sense for one player (star power / depth are team-only).
PLAYER_PART_LABELS = {
    "track_record": "Track record", "track_efficiency": "Efficiency", "form": "This season",
    "accolades": "Honors", "pedigree": "Draft", "sep_adj": "Separation", "yac_oe": "YAC",
}
_PLAYER_FROM_TEAM_PART = {"form": "form_adj"}  # player part -> its weight key in weapon_parts
PLAYER_POOL_MIN_TARGETS = 15  # players compared against pass catchers with at least this many targets


def player_weapons(data: dict[str, pd.DataFrame], weapon_parts: dict | None = None) -> pd.DataFrame:
    """Every WR/TE/RB who has caught passes this season, with a grade for each part and overall.

    Grades are vs. others at the same position (50 = average WR / TE / RB with 15+ targets, +10 = 1 std dev).
    Weights come from weapon_parts. A part with no data for a player (e.g. not tracked by Next Gen Stats)
    is left out of his overall grade and the rest are re-weighted.
    """
    wk = data["weekly"]
    season = int(wk["season"].iloc[0])
    parts = weapon_parts or WEAPON_PARTS
    pc = wk[wk["position"].isin(PASS_CATCHERS)]
    if pc.empty:
        return pd.DataFrame()

    # This season: totals, and the team he's caught the most targets for
    p = pc.groupby("player_id").agg(
        name=("player_display_name", "first"), position=("position", "first"), games=("week", "nunique"),
        targets=("targets", "sum"), receptions=("receptions", "sum"), rec_yds=("receiving_yards", "sum"),
        rec_td=("receiving_tds", "sum"), carries=("carries", "sum"), rush_yds=("rushing_yards", "sum"))
    by_team = pc.groupby(["player_id", "team"])["targets"].sum().reset_index()
    p["team"] = by_team.sort_values("targets").groupby("player_id")["team"].last()
    team_tg = pc.groupby("team")["targets"].sum()
    mine = by_team.set_index(["player_id", "team"])["targets"]
    p["target_share"] = [mine.get((i, t), 0) / team_tg.get(t, 1) * 100 for i, t in zip(p.index, p["team"])]
    p["ypg"] = p["rec_yds"] / p["games"].clip(lower=1)
    p["ypt"] = p["rec_yds"] / p["targets"].replace(0, np.nan)

    # Track record, honors, draft (2 prior seasons + this one, position-relative)
    tal = receiver_talent(data, season)
    p = p.join(tal[["talent_ypg", "talent_ypg_idx", "talent_ypt_idx", "accolades", "pedigree"]])

    # This season's efficiency vs. his position (yards per target)
    # Comparison pool: 15+ targets once the season is underway; early on (e.g. week 1) scale the cutoff
    # to the games played so far so there's always a group to compare against
    weeks_played = max(int(wk["week"].nunique()), 1)
    pool_min = min(PLAYER_POOL_MIN_TARGETS, max(2, round(PLAYER_POOL_MIN_TARGETS * weeks_played / 6)))
    pool = p["targets"] >= pool_min
    pos_ypt = (p.loc[pool, "rec_yds"].groupby(p.loc[pool, "position"]).sum()
               / p.loc[pool, "targets"].groupby(p.loc[pool, "position"]).sum())
    p["form"] = p["ypt"] / p["position"].map(pos_ypt) * 100

    # Next Gen Stats (only receivers NGS tracks)
    rec = data["ngs_rec"].copy()
    if len(rec):
        rec["sep_adj"] = depth_adjusted_separation(rec)
        n = rec.drop_duplicates("player_gsis_id").set_index("player_gsis_id")
        p["separation"] = n["avg_separation"].reindex(p.index)
        p["sep_adj"] = n["sep_adj"].reindex(p.index)
        p["yac_oe"] = n["avg_yac_above_expectation"].reindex(p.index)
    else:
        p["separation"] = p["sep_adj"] = p["yac_oe"] = np.nan
    p["ngs_tracked"] = p["sep_adj"].notna()

    src = {"track_record": "talent_ypg_idx", "track_efficiency": "talent_ypt_idx", "form": "form",
           "accolades": "accolades", "pedigree": "pedigree", "sep_adj": "sep_adj", "yac_oe": "yac_oe"}
    z = {}
    for k, col in src.items():
        x = p[col].astype(float)
        if k in ("accolades", "pedigree", "sep_adj", "yac_oe"):  # compare within position
            ref = x[pool]
            mu = ref.groupby(p.loc[pool, "position"]).mean()
            sd = ref.groupby(p.loc[pool, "position"]).std(ddof=0).replace(0, np.nan)
            z[k] = (x - p["position"].map(mu)) / p["position"].map(sd)
        else:  # already 100 = position average
            sd = x[pool].std(ddof=0) or 1
            z[k] = (x - 100) / sd
        z[k] = z[k].clip(-3, 3)
        p[f"{k}_grade"] = (50 + 10 * z[k]).round(0)

    w = {k: parts.get(_PLAYER_FROM_TEAM_PART.get(k, k), 0) for k in src}
    num = sum(z[k].fillna(0) * wt for k, wt in w.items())
    den = sum(z[k].notna() * wt for k, wt in w.items()).replace(0, np.nan)
    combo = num / den
    sd = combo[pool].std(ddof=0) or 1
    raw = (combo - combo[pool].mean()) / sd
    # Few targets = small sample: pull toward average (15 targets keeps half of his difference from average)
    k = pool_min
    p["player_grade"] = (50 + 10 * raw * p["targets"] / (p["targets"] + k)).round(0)
    p["player_ovr"] = to_ovr(50 + 10 * raw * p["targets"] / (p["targets"] + k))  # display only
    return p.reset_index().sort_values(["team", "targets"], ascending=[True, False])


def team_weapons(players: pd.DataFrame, team: str) -> pd.DataFrame:
    """One team's pass catchers, most-targeted first."""
    return players[players["team"] == team].sort_values("targets", ascending=False).reset_index(drop=True)


def _nth_best(s: pd.Series, n: int) -> float:
    v = s.sort_values(ascending=False)
    return v.iloc[n - 1] if len(v) >= n else v.min()


def team_context(data: dict[str, pd.DataFrame], weapon_parts: dict | None = None,
                 pass_pro_parts: dict | None = None) -> pd.DataFrame:
    rec, rush, wk = data["ngs_rec"].copy(), data["ngs_rush"], data["weekly"]
    season = int(wk["season"].iloc[0])
    parts = weapon_parts or WEAPON_PARTS

    # --- weapons: who the team actually throws to this season, rated on track record + this season's NGS
    talent = receiver_talent(data, season)
    tg = wk[wk["position"].isin(PASS_CATCHERS)].groupby(["team", "player_id"])["targets"].sum().reset_index()
    tg = tg[tg["targets"] > 0].join(talent[["talent_ypg_idx", "talent_ypt_idx", "accolades", "pedigree"]], on="player_id")
    tg["share"] = tg["targets"] / tg.groupby("team")["targets"].transform("sum")
    rec["sep_adj"] = depth_adjusted_separation(rec)

    # This season's form: production beyond what the passers' accuracy explains
    tgr = team_game_receiving(wk).dropna(subset=["ypt_res"])
    tw = tgr.groupby("team")["targets"].sum()
    form_ypt = (tgr["ypt_res"] * tgr["targets"]).groupby(tgr["team"]).sum() / tw
    form_ept = (tgr["ept_res"] * tgr["targets"]).groupby(tgr["team"]).sum() / tw

    t = pd.DataFrame({
        "track_record": wavg(tg, "talent_ypg_idx", "targets", "team"),
        "track_efficiency": wavg(tg, "talent_ypt_idx", "targets", "team"),
        "star_power": tg[tg["share"] >= 0.10].groupby("team")["talent_ypg_idx"].max(),
        "depth": tg[tg["share"] >= 0.05].groupby("team")["talent_ypg_idx"].apply(_nth_best, n=3),
        "accolades": wavg(tg, "accolades", "targets", "team"),
        "pedigree": wavg(tg, "pedigree", "targets", "team"),
        "form_adj": (zs(form_ypt) + zs(form_ept)) / 2,
        "separation": wavg(rec, "avg_separation", "targets", "team_abbr"),
        "sep_adj": wavg(rec, "sep_adj", "targets", "team_abbr"),
        "yac_oe": wavg(rec, "avg_yac_above_expectation", "receptions", "team_abbr"),
    })
    nonqb = wk[wk["position"] != "QB"].groupby("team")[["carries", "rushing_yards", "rushing_epa"]].sum()
    t["rb_epa_per_carry"] = nonqb["rushing_epa"] / nonqb["carries"]
    t["rb_ypc"] = nonqb["rushing_yards"] / nonqb["carries"]
    rb = rush[rush["player_position"] == "RB"]
    t["ryoe_per_att"] = wavg(rb, "rush_yards_over_expected_per_att", "rush_attempts", "team_abbr")
    t = t.dropna(how="all")

    t = t[t.index.isin(wk["team"].unique())]  # only teams that actually played

    def zf(col: str) -> pd.Series:
        """z-score with gaps filled by the average; a stat with no data this season (e.g. RYOE before 2018) is neutral."""
        x = t[col]
        return zs(x.fillna(x.mean())) if x.notna().any() else pd.Series(0.0, index=t.index)

    live = {k: w for k, w in parts.items() if t[k].notna().any()} or parts
    tot = sum(live.values()) or 1
    for k in parts:
        t[f"{k}_grade"] = (50 + 10 * zf(k)).round(0)  # per-part grade for display
    t["weapons_z"] = zs(sum(w / tot * zf(k) for k, w in live.items()))
    run_parts = [c for c in ["rb_epa_per_carry", "rb_ypc", "ryoe_per_att"] if t[c].notna().any()]
    t["run_game_z"] = zs(sum(zf(c) for c in run_parts) / len(run_parts))
    t["weapons_grade"] = (50 + 10 * t["weapons_z"]).round(0).astype(int)
    t["run_game_grade"] = (50 + 10 * t["run_game_z"]).round(0).astype(int)

    pp = pass_protection(data, pass_pro_parts)
    if len(pp):
        t = t.join(pp, how="left")
        t["pass_pro_z"] = t["pass_pro_z"].fillna(0)
        t["pass_pro_grade"] = t["pass_pro_grade"].fillna(50).astype(int)
    else:  # no pressure data for this season (pre-2018): neutral
        t["pass_pro_z"], t["pass_pro_grade"] = 0.0, 50
    return t


# ------------------------------------------------------------------------ metrics
def passer_rating(cmp, att, yds, td, ints):
    clip = lambda s: s.clip(0, 2.375)
    a = clip((cmp / att - 0.3) * 5)
    b = clip((yds / att - 3) * 0.25)
    c = clip(td / att * 20)
    d = clip(2.375 - ints / att * 25)
    return (a + b + c + d) / 6 * 100


SUM_COLS = ["completions", "attempts", "passing_yards", "passing_tds", "passing_interceptions",
            "sacks_suffered", "sack_yards_lost", "sack_fumbles_lost", "passing_first_downs", "passing_epa",
            "carries", "rushing_yards", "rushing_tds", "rushing_fumbles_lost", "rushing_first_downs", "rushing_epa"]


# "With or without you": how his pass catchers played in games he did NOT start.
# A backup who can't put the ball on target makes receivers look worse than they are, so the
# receivers are judged only on what they control: getting open, gaining yards after the catch,
# and production AFTER removing what the passer's accuracy (CPOE) explains.
WITHOUT_QB = {
    "max_share": 0.5,   # most the without-him games can make up of his weapons grade
    "prior_games": 3,   # share = max_share * games / (games + prior_games); 3 games missed -> 25%
    "parts": {"getting_open": 40, "after_catch": 20, "production_adj": 40},
}
WITHOUT_QB_LABELS = {"getting_open": "Getting open (depth-adj separation)", "after_catch": "YAC over expected",
                     "production_adj": "Production, adjusted for passer accuracy"}


def _residual(y: pd.Series, x: pd.Series, w: pd.Series) -> pd.Series:
    """y minus its weighted linear fit on x (what's left after x is accounted for)."""
    ok = y.notna() & x.notna() & (w > 0)
    sw = np.sqrt(w[ok].to_numpy())
    X = np.column_stack([np.ones(ok.sum()), x[ok].to_numpy()])
    coef, *_ = np.linalg.lstsq(X * sw[:, None], y[ok].to_numpy() * sw, rcond=None)
    return (y - (coef[0] + coef[1] * x)).where(ok)


def weapons_without_qb(wk: pd.DataFrame, log: pd.DataFrame, ngs_weekly: pd.DataFrame,
                       parts: dict | None = None) -> pd.DataFrame:
    """For each QB: how his team's WR/TE/RBs played in the games someone else started.

    Three receiver-controlled measures, each scaled so 1.0 = one standard deviation between teams:
      getting_open   - depth-adjusted separation in those games (weekly Next Gen Stats)
      after_catch    - YAC over expected in those games (weekly Next Gen Stats)
      production_adj - receiving yards & EPA per target, minus what the passer's CPOE explains
                       (fit on every team-game this season)
    """
    parts = parts or WITHOUT_QB["parts"]

    # --- team-game production, and what's left after the passer's accuracy
    g = (log.merge(team_game_receiving(wk).drop(columns="week"), on=["game_id", "team"], how="inner")
         .rename(columns={"player_id": "qb"}))

    # --- weekly tracking: getting open and after the catch, per team-game
    n = ngs_weekly.copy()
    if len(n):
        n["sep_adj"] = depth_adjusted_separation(n)
        n["sx"], n["yx"] = n["sep_adj"] * n["targets"], n["avg_yac_above_expectation"] * n["receptions"]
        tw = n.groupby(["team_abbr", "week"])[["sx", "targets", "yx", "receptions"]].sum()
        tw.columns = ["sx", "s_tg", "yx", "y_rec"]
        g = g.join(tw, on=["team", "week"])
    else:
        g = g.assign(sx=np.nan, s_tg=np.nan, yx=np.nan, y_rec=np.nan)

    def team_rates(d: pd.DataFrame) -> dict:
        tg = d["targets"].where(d["ypt_res"].notna(), 0)
        return {
            "getting_open": d["sx"].sum() / d["s_tg"].sum() if d["s_tg"].sum() else np.nan,
            "after_catch": d["yx"].sum() / d["y_rec"].sum() if d["y_rec"].sum() else np.nan,
            "ypt_res": (d["ypt_res"] * tg).sum() / tg.sum() if tg.sum() else np.nan,
            "ept_res": (d["ept_res"] * tg).sum() / tg.sum() if tg.sum() else np.nan,
        }

    # Scale: spread of each measure between teams over the season
    league = pd.DataFrame({t: team_rates(d) for t, d in g.groupby("team")}).T
    sd = league.std(ddof=0).replace(0, 1)
    mean = league.mean()

    rows = []
    for (qb, tm), n_starts in g.groupby(["qb", "team"]).size().items():
        wo = g[(g["team"] == tm) & (g["qb"] != qb)]
        if wo.empty:
            continue
        r = team_rates(wo)
        z = {"getting_open": (r["getting_open"] - mean["getting_open"]) / sd["getting_open"],
             "after_catch": (r["after_catch"] - mean["after_catch"]) / sd["after_catch"],
             "production_adj": ((r["ypt_res"] - mean["ypt_res"]) / sd["ypt_res"]
                                + (r["ept_res"] - mean["ept_res"]) / sd["ept_res"]) / 2}
        z = {k: float(np.clip(v, -3, 3)) if pd.notna(v) else v for k, v in z.items()}  # one fluky game can't run away
        have = {k: w for k, w in parts.items() if w and pd.notna(z.get(k))}
        if not have:
            continue
        cp = wo["cpoe"].dropna()
        rows.append({"player_id": qb, "n_starts": n_starts, "wo_games": wo["game_id"].nunique(),
                     "wo_z": sum(z[k] * w for k, w in have.items()) / sum(have.values()),
                     **{f"wo_{k}_z": z[k] for k in WITHOUT_QB_LABELS},
                     "wo_ypt": wo["receiving_yards"].sum() / max(wo["targets"].sum(), 1),
                     "wo_passer_cpoe": cp.mean() if len(cp) else np.nan})
    cols = ["wo_games", "wo_z", "wo_ypt", "wo_passer_cpoe", *[f"wo_{k}_z" for k in WITHOUT_QB_LABELS]]
    if not rows:
        return pd.DataFrame(columns=cols, dtype=float)  # typed, so a join doesn't create object columns
    r = pd.DataFrame(rows)
    # A QB who started for two teams: weight each team's without-him sample by his starts there
    out = {"wo_games": r.groupby("player_id")["wo_games"].sum()}
    for c in cols[1:]:
        v = r.dropna(subset=[c])
        out[c] = (v[c] * v["n_starts"]).groupby(v["player_id"]).sum() / v.groupby("player_id")["n_starts"].sum()
    return pd.DataFrame(out)


# ------------------------------------------------------------ missing players (snap counts)
# Who actually played each game comes from snap counts: a player with 0 snaps missed it.
MISSING = {
    "key_weapon_min_share": 10,   # pass catcher with >= this % of his team's targets = a key weapon
    "replacement_grade": 42,      # a missing weapon's targets go to a typical backup with this player grade
    "key_defenders": 4,           # each defense's top N players (by production + Pro Bowls) are its key players
    "defense_impact": 1.0,        # losing ALL key defenders makes a defense this many std devs weaker
}


def _norm_name(n) -> str:
    """'Michael Penix Jr.' -> 'michaelpenix' (so stat names and snap-count names line up)."""
    n = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", str(n).lower())
    return re.sub(r"[^a-z]", "", n)


def _played(snaps: pd.DataFrame, side: str) -> set:
    """{(name_key, team, week)} for everyone with at least one offensive / defensive snap."""
    col = "offense_snaps" if side == "offense" else "defense_snaps"
    s = snaps[snaps[col].fillna(0) > 0]
    return set(zip(s["player"].map(_norm_name), s["team"], s["week"]))


def missing_weapons(data: dict[str, pd.DataFrame], week: int, cfg: dict | None = None,
                    weapon_parts: dict | None = None) -> pd.DataFrame:
    """For each team that played in `week`: which key pass catchers missed the game, and how much
    that lowers the team's weapons grade for that game (in std devs between teams).

    Key weapon = at least key_weapon_min_share % of the team's targets this season (through that week).
    His share of the targets is treated as going to a typical backup (replacement_grade).
    """
    cfg = {**MISSING, **(cfg or {})}
    snaps = data.get("snaps", pd.DataFrame())
    if snaps.empty:
        return pd.DataFrame(columns=["weapons_delta_z", "missing_weapons"])
    players = player_weapons(filter_weeks(data, 1, week), weapon_parts)
    players = players[players["player_grade"].notna() & (players["target_share"] > 0)]
    played = _played(snaps[snaps["week"] == week], "offense")
    teams_playing = set(snaps.loc[snaps["week"] == week, "team"])

    rows = {}
    for team, P in players.groupby("team"):
        full = (P["player_grade"] * P["target_share"]).sum() / P["target_share"].sum()
        absent = pd.Series([(_norm_name(n), team, week) not in played for n in P["name"]], index=P.index)
        gone = P[(P["target_share"] >= cfg["key_weapon_min_share"]) & absent] if team in teams_playing else P.iloc[0:0]
        g = P["player_grade"].where(~P.index.isin(gone.index), cfg["replacement_grade"])
        now = (g * P["target_share"]).sum() / P["target_share"].sum()
        rows[team] = {"full_avg": full, "week_avg": now,
                      "missing_weapons": ", ".join(f"{n} ({pos}, {sh:.0f}% of targets)" for n, pos, sh in
                                                   zip(gone["name"], gone["position"], gone["target_share"]))}
    if not rows:
        return pd.DataFrame(columns=["weapons_delta_z", "missing_weapons"])
    t = pd.DataFrame(rows).T
    sd = t["full_avg"].astype(float).std(ddof=0) or 1
    t["weapons_delta_z"] = (t["week_avg"].astype(float) - t["full_avg"].astype(float)) / sd
    return t[["weapons_delta_z", "missing_weapons"]]


_MEMO: dict = {}  # results reused within one run for the exact same data (keyed by object identity)


def _memo_key(*objs) -> tuple:
    return tuple((id(o), len(o)) if isinstance(o, pd.DataFrame) else o for o in objs)


def key_defenders(data: dict[str, pd.DataFrame], week: int, n: int = 4) -> pd.DataFrame:
    wk = data.get("weekly_all", data["weekly"])
    key = ("kd", *_memo_key(wk, data["draft"]), week, n)
    if key not in _MEMO:
        _MEMO[key] = _key_defenders(data, week, n)
    return _MEMO[key]


def _key_defenders(data: dict[str, pd.DataFrame], week: int, n: int = 4) -> pd.DataFrame:
    """Each defense's top-n players going into `week`: production per game in earlier weeks
    (sacks, QB hits, interceptions, passes defended, tackles for loss, forced fumbles) + Pro Bowls.
    Returns rows (team, name, position, value, share of the team's key-player value)."""
    wk = data.get("weekly_all", data["weekly"])
    d = wk[(wk["position_group"].isin(["DL", "LB", "DB"])) & (wk["week"] < week)]
    if d.empty:
        return pd.DataFrame(columns=["team", "player_id", "name", "position", "value", "share"])
    f = lambda c: d[c].fillna(0) if c in d else 0
    d = d.assign(v=2 * f("def_sacks") + 0.7 * f("def_qb_hits") + 3 * f("def_interceptions")
                 + 1 * f("def_pass_defended") + 0.5 * f("def_tackles_for_loss") + 1.5 * f("def_fumbles_forced"))
    by = d.groupby(["player_id", "team"]).agg(name=("player_display_name", "first"), position=("position", "first"),
                                              g=("week", "nunique"), v=("v", "sum")).reset_index()
    team_games = d.groupby("team")["week"].nunique()
    by = by[by["g"] >= np.ceil(by["team"].map(team_games) / 2)]  # regulars only
    pb = data["draft"].dropna(subset=["gsis_id"]).drop_duplicates("gsis_id").set_index("gsis_id")
    honors = by["player_id"].map(pb["probowls"]).fillna(0) + 2 * by["player_id"].map(pb["allpro"]).fillna(0)
    by["value"] = by["v"] / by["g"] + 0.5 * honors
    top = by.sort_values("value", ascending=False).groupby("team").head(n).copy()
    top["share"] = top["value"] / top.groupby("team")["value"].transform("sum")
    return top[["team", "player_id", "name", "position", "value", "share"]]


def defense_missing_impact(data: dict[str, pd.DataFrame], weeks: list[int], cfg: dict | None = None) -> pd.DataFrame:
    """Rows (week, defense): how much weaker the defense was that week from missing key players
    (in std devs, positive = weaker), and who was missing."""
    cfg = {**MISSING, **(cfg or {})}
    snaps = data.get("snaps", pd.DataFrame())
    cols = ["week", "defense", "def_missing_z", "def_missing"]
    dtypes = {"week": int, "def_missing_z": float}  # keep numeric even when empty (e.g. a single week)
    if snaps.empty:
        return pd.DataFrame(columns=cols).astype(dtypes)
    out = []
    for w in weeks:
        played = _played(snaps[snaps["week"] == w], "defense")
        playing = set(snaps.loc[snaps["week"] == w, "team"])
        kd = key_defenders(data, w, int(cfg["key_defenders"]))
        kd = kd[kd["team"].isin(playing)]
        kd = kd[np.array([(_norm_name(nm), t, w) not in played for nm, t in zip(kd["name"], kd["team"])], dtype=bool)]
        for t, g in kd.groupby("team"):
            out.append({"week": w, "defense": t, "def_missing_z": cfg["defense_impact"] * g["share"].sum(),
                        "def_missing": ", ".join(f"{nm} ({pos})" for nm, pos in zip(g["name"], g["position"]))})
    return pd.DataFrame(out, columns=cols).astype(dtypes)


# ------------------------------------------------------------ defenses faced
# How a pass defense is rated (relative weights): what it allowed to every QB it faced.
DEFENSE = {
    "parts": {"epa_per_dropback": 50, "anya": 30, "sack_rate": 20},
    "prior_games": 4,  # last season's rating counts as this many games, fading as the season goes on
}
DEFENSE_LABELS = {"epa_per_dropback": "EPA/dropback allowed", "anya": "ANY/A allowed", "sack_rate": "Sack rate"}
_DEF_SUMS = ["attempts", "sacks_suffered", "passing_yards", "passing_tds", "passing_interceptions",
             "sack_yards_lost", "passing_epa"]
_DEF_SIGN = {"epa_per_dropback": -1, "anya": -1, "sack_rate": 1}  # +z = better (tougher) defense


def _def_rates(t: pd.DataFrame) -> pd.DataFrame:
    db = (t["attempts"] + t["sacks_suffered"]).clip(lower=1)
    return pd.DataFrame({
        "epa_per_dropback": t["passing_epa"] / db,
        "anya": (t["passing_yards"] + 20 * t["passing_tds"] - 45 * t["passing_interceptions"]
                 - t["sack_yards_lost"]) / db,
        "sack_rate": t["sacks_suffered"] / db * 100,
    }, index=t.index)


def _def_games(qb_rows: pd.DataFrame) -> pd.DataFrame:
    """What all QBs did against each defense, one row per (game, defense)."""
    q = qb_rows[(qb_rows["attempts"].fillna(0) + qb_rows["sacks_suffered"].fillna(0)) > 0]
    return (q.groupby(["game_id", "week", "opponent_team"])[_DEF_SUMS].sum().fillna(0)
            .reset_index().rename(columns={"opponent_team": "defense"}))


def defense_ratings_by_week(wk: pd.DataFrame, prev_weekly: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    key = ("dr", *_memo_key(wk, prev_weekly), json.dumps(cfg, sort_keys=True, default=str))
    if key not in _MEMO:
        _MEMO[key] = _defense_ratings_by_week(wk, prev_weekly, cfg)
    return _MEMO[key]


def _defense_ratings_by_week(wk: pd.DataFrame, prev_weekly: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Each pass defense's rating GOING INTO each week: built only from games before that week,
    plus last season's rating counted as a few games (so week 1 isn't blank).

    Returns rows (week, defense) with toughness z (higher = tougher, vs. the other 31 defenses
    that week), its 1-32 rank (1 = best pass defense), and the stats behind it.
    """
    cfg = {**DEFENSE, **(cfg or {})}
    parts, k = cfg["parts"], cfg["prior_games"]
    tot = sum(parts.values()) or 1
    games = _def_games(wk[wk["position"] == "QB"])

    prior = pd.DataFrame(columns=_DEF_SUMS, dtype=float)
    if len(prev_weekly):
        pg = _def_games(prev_weekly)
        n = pg.groupby("defense")["game_id"].nunique()
        prior = pg.groupby("defense")[_DEF_SUMS].sum().div(n, axis=0) * k  # k games' worth of last season

    out = []
    weeks = sorted(set(games["week"]) | set(wk["week"].unique()))
    for w in weeks + [max(weeks) + 1]:  # the extra week = ratings right now (going into next week)
        cum = games[games["week"] < w].groupby("defense")[_DEF_SUMS].sum()
        tot_sums = cum.add(prior, fill_value=0)
        if tot_sums.empty:
            continue
        r = _def_rates(tot_sums)
        mean, sd = r.mean(), r.std(ddof=0).replace(0, 1)
        tough = sum(w_ / tot * _DEF_SIGN[c] * (r[c] - mean[c]) / sd[c] for c, w_ in parts.items())
        r["tough"] = tough
        r["rank"] = tough.rank(ascending=False, method="min").astype(int)
        r["games_so_far"] = games[games["week"] < w].groupby("defense")["game_id"].nunique()
        out.append(r.assign(week=w).rename_axis("defense").reset_index())
    res = pd.concat(out, ignore_index=True)
    res["games_so_far"] = res["games_so_far"].fillna(0).astype(int)
    res["defense_grade"] = (50 + 10 * res["tough"] / res.groupby("week")["tough"].transform("std")).round(0)
    return res


def defenses_faced(wk: pd.DataFrame, log: pd.DataFrame, prev_weekly: pd.DataFrame,
                   cfg: dict | None = None, data: dict | None = None,
                   missing_cfg: dict | None = None) -> tuple[pd.Series, pd.DataFrame]:
    """Average toughness of the pass defenses in each QB's starts, each defense rated as of the
    week of the game. Returns (per-QB toughness, per-start detail with the opponent's rank)."""
    ratings = defense_ratings_by_week(wk, prev_weekly, cfg)
    starts = log.merge(ratings[["week", "defense", "tough", "rank"]],
                       left_on=["week", "opp"], right_on=["week", "defense"], how="left")
    if data is not None and len(starts):
        mi = defense_missing_impact(data, sorted(starts["week"].unique()), missing_cfg)
        starts = starts.merge(mi, on=["week", "defense"], how="left")
        starts["def_missing_z"] = starts["def_missing_z"].fillna(0)
        starts["tough"] = starts["tough"] - starts["def_missing_z"]  # missing key defenders = easier opponent
    else:
        starts["def_missing_z"], starts["def_missing"] = 0.0, np.nan
    return starts.groupby("player_id")["tough"].mean(), starts


def build_qbs(data: dict[str, pd.DataFrame], ctx: pd.DataFrame, min_starts: int,
              without_qb: dict | None = None, defense_cfg: dict | None = None,
              pool_min: int | None = None, missing_cfg: dict | None = None,
              all_starters: bool = False) -> pd.DataFrame:
    log = starter_log(data["games"], data["weekly"])
    wk = data["weekly"]
    qbw = wk[wk["position"] == "QB"].merge(log, on=["game_id", "team", "player_id"], how="inner").copy()
    qbw["cpoe_x_att"] = qbw["passing_cpoe"] * qbw["attempts"]
    qbw["cpoe_att"] = qbw["attempts"].where(qbw["passing_cpoe"].notna(), 0)

    agg = qbw.groupby("player_id").agg(
        name=("player_display_name", "first"), starts=("game_id", "nunique"),
        wins=("win", "sum"), losses=("loss", "sum"), ties=("tie", "sum"),
        **{c: (c, "sum") for c in SUM_COLS + ["cpoe_x_att", "cpoe_att"]})

    # Team = where he started most (latest start breaks ties). Context = start-weighted across teams.
    by_team = log.groupby(["player_id", "team"]).agg(n=("game_id", "size"), last=("week", "max")).reset_index()
    by_team = by_team.join(ctx[["weapons_z", "run_game_z", "pass_pro_z", "weapons_grade", "run_game_grade",
                                  "pass_pro_grade"]], on="team")
    agg["team"] = by_team.sort_values(["n", "last"]).groupby("player_id")["team"].last()
    for c in ["weapons_z", "run_game_z", "pass_pro_z", "weapons_grade", "run_game_grade", "pass_pro_grade"]:
        agg[c] = (by_team[c] * by_team["n"]).groupby(by_team["player_id"]).sum() / by_team.groupby("player_id")["n"].sum()

    # Who counts as a starter: his team's primary starter OR the QB who started the team's latest game.
    team_starts = log.groupby(["team", "player_id"]).size().rename("n").reset_index()
    primary = set(team_starts.sort_values("n").groupby("team")["player_id"].last())
    current = set(log.sort_values("week").groupby("team")["player_id"].last())
    agg["role"] = ["Starter" if p in primary else "Current" if p in current else "" for p in agg.index]
    keep_role = (agg["role"] != "") | all_starters  # all_starters: also backups who started (used by predictions)
    agg = agg[keep_role & (agg["starts"] >= min_starts)].copy()

    # Blend in how his weapons played without him (only applies if he missed games)
    wcfg = {**WITHOUT_QB, **(without_qb or {})}
    wo = weapons_without_qb(wk, log, data["ngs_rec_weekly"], wcfg.get("parts"))
    agg = agg.join(wo)
    agg["wo_games"] = agg["wo_games"].fillna(0).astype(int)
    share = wcfg["max_share"] * agg["wo_games"] / (agg["wo_games"] + wcfg["prior_games"])
    agg["wo_share"] = share.where(agg["wo_z"].notna(), 0)
    agg["weapons_team_grade"] = agg["weapons_grade"]
    agg["weapons_z"] = (1 - agg["wo_share"]) * agg["weapons_z"] + agg["wo_share"] * agg["wo_z"].fillna(0)
    agg["weapons_grade"] = 50 + 10 * agg["weapons_z"]
    agg["wo_grade"] = 50 + 10 * agg["wo_z"]

    # Defenses faced in his starts (higher = tougher), scaled across the ranked starters
    tough, sched = defenses_faced(data.get("weekly_all", wk), log, data.get("prev_weekly", pd.DataFrame()), defense_cfg,
                                  data, missing_cfg)
    _t = agg.index.to_series().map(tough).fillna(0)
    agg["opp_defense_z"] = zs_vs(_t, agg["starts"] >= (pool_min or min_starts))
    sched = sched.sort_values("week")
    agg["schedule"] = sched.groupby("player_id").apply(
        lambda d: ", ".join((f"W{w} {o} (#{r})" if pd.notna(r) else f"W{w} {o}") + ("*" if mz > 0 else "")
                            for w, o, r, mz in zip(d["week"], d["opp"], d["rank"], d["def_missing_z"])),
        include_groups=False)
    agg["opp_defense_grade"] = 50 + 10 * agg["opp_defense_z"]
    last = sched.groupby("player_id").tail(1).set_index("player_id")
    agg["last_opp"], agg["last_opp_rank"] = last["opp"], last["rank"]
    agg["last_opp_missing"] = last["def_missing"] if "def_missing" in last else np.nan

    # Interception-worthy throws (his own passes, games he started)
    th = data.get("throws", pd.DataFrame())
    if len(th):
        mine_t = th.merge(log[["game_id", "team", "player_id"]], on=["game_id", "team", "player_id"])
        tq = mine_t.groupby("player_id")[["charted", "worthy", "int", "worthy_dropped", "int_not_worthy"]].sum()
        agg = agg.join(tq.add_prefix("tq_"))
        agg["tq_worthy_rate"] = agg["tq_worthy"] / agg["tq_charted"].replace(0, np.nan) * 100
        agg["tq_league_rate"] = th["worthy"].sum() / th["charted"].sum() * 100

    # His own protection numbers in the games he started (pressures etc. are per team-game)
    ppg = pass_protection_games(data)
    if len(ppg):
        mine = log.merge(ppg.drop(columns="week"), on=["game_id", "team"])
        s_ = mine.groupby("player_id")[["dropbacks", "times_pressured", "times_hit", "times_hurried",
                                         "times_sacked", "times_blitzed"]].sum()
        ttt = (mine["time_to_throw"] * mine["dropbacks"]).groupby(mine["player_id"]).sum() / s_["dropbacks"]
        agg = agg.join(s_.add_prefix("pp_")).join(ttt.rename("pp_ttt"))
        agg["pp_pressure_pct"] = agg["pp_times_pressured"] / agg["pp_dropbacks"] * 100
        agg["pp_league_pressure_pct"] = ppg["times_pressured"].sum() / ppg["dropbacks"].sum() * 100
        agg["pp_league_ttt"] = (ppg["time_to_throw"] * ppg["dropbacks"]).sum() / ppg["dropbacks"].sum()
        # Time his line gave him vs. time he took to throw
        kk, qq = pressure_clock(ppg)
        agg["pp_time_given"] = time_given(agg["pp_pressure_pct"] / 100, agg["pp_ttt"], kk, qq)
        agg["pp_margin"] = agg["pp_time_given"] - agg["pp_ttt"]

    f = lambda c: agg[c]
    gp, att = f("starts"), f("attempts").clip(lower=1)
    dropbacks = (f("attempts") + f("sacks_suffered")).clip(lower=1)
    plays = (dropbacks + f("carries")).clip(lower=1)

    m = agg[["name", "team", "role", "starts", "weapons_z", "run_game_z", "pass_pro_z"]].copy()
    m["weapons_grade"] = agg["weapons_grade"].round(0).astype(int)
    m["weapons_team_grade"] = agg["weapons_team_grade"].round(0).astype(int)
    m["wo_games"], m["wo_share"] = agg["wo_games"], agg["wo_share"].round(2)
    m["wo_grade"], m["wo_ypt"] = agg["wo_grade"].round(0), agg["wo_ypt"].round(2)
    m["wo_passer_cpoe"] = agg["wo_passer_cpoe"].round(1)
    m["opp_defense_z"] = agg["opp_defense_z"]
    m["opp_defense_grade"] = agg["opp_defense_grade"].round(0).astype(int)
    m["schedule"] = agg["schedule"]
    m["last_opp"], m["last_opp_rank"] = agg["last_opp"], agg["last_opp_rank"]
    m["last_opp_missing"] = agg["last_opp_missing"]
    for k in WITHOUT_QB_LABELS:
        m[f"wo_{k}_grade"] = (50 + 10 * agg[f"wo_{k}_z"]).round(0)
    m["run_game_grade"] = agg["run_game_grade"].round(0).astype(int)
    m["pass_pro_grade"] = agg["pass_pro_grade"].round(0).astype(int)
    for c in ["pp_dropbacks", "pp_times_pressured", "pp_times_hit", "pp_times_hurried", "pp_times_sacked",
              "pp_times_blitzed", "pp_pressure_pct", "pp_ttt", "pp_league_pressure_pct", "pp_league_ttt",
              "pp_time_given", "pp_margin"]:
        m[c] = agg[c] if c in agg else np.nan
    m["record"] = [f"{w}-{l}" + (f"-{t}" if t else "") for w, l, t in zip(agg.wins, agg.losses, agg.ties)]
    m["pass_yds"], m["pass_td"], m["ints"] = f("passing_yards"), f("passing_tds"), f("passing_interceptions")
    m["rush_yds"], m["rush_td"] = f("rushing_yards"), f("rushing_tds")
    m["pass_yds_pg"] = f("passing_yards") / gp
    m["pass_td_pg"] = f("passing_tds") / gp
    m["att_pg"] = f("attempts") / gp
    m["cmp_pct"] = f("completions") / att * 100
    m["rush_yds_pg"] = f("rushing_yards") / gp
    m["rush_td_pg"] = f("rushing_tds") / gp
    m["epa_per_play"] = (f("passing_epa") + f("rushing_epa")) / plays
    m["anya"] = (f("passing_yards") + 20 * f("passing_tds") - 45 * f("passing_interceptions") - f("sack_yards_lost")) / dropbacks
    m["passer_rating"] = passer_rating(f("completions"), att, f("passing_yards"), f("passing_tds"), f("passing_interceptions"))
    m["cpoe"] = f("cpoe_x_att") / f("cpoe_att").clip(lower=1)
    m["first_down_rate"] = (f("passing_first_downs") + f("rushing_first_downs")) / plays * 100
    m["int_rate"] = f("passing_interceptions") / att * 100
    m["fumbles_lost_pg"] = (f("sack_fumbles_lost") + f("rushing_fumbles_lost")) / gp
    m["sack_rate"] = f("sacks_suffered") / dropbacks * 100
    m["win_pct"] = (agg["wins"] + 0.5 * agg["ties"] + 2) / (gp + 4) * 100  # regressed toward .500
    # Time margin: time his line gave him minus time he took (positive = ball out before pressure). 0 if no data.
    m["pocket_margin"] = agg["pp_margin"].fillna(agg["pp_margin"].mean()).fillna(0) if "pp_margin" in agg else 0.0
    # Interception-worthy throws per 100 attempts (league average if no charting this season)
    if "tq_worthy_rate" in agg:
        m["int_worthy_rate"] = agg["tq_worthy_rate"].fillna(agg["tq_league_rate"])
        for c in ["tq_charted", "tq_worthy", "tq_int", "tq_worthy_dropped", "tq_int_not_worthy", "tq_league_rate"]:
            m[c] = agg[c]
    else:
        m["int_worthy_rate"] = 0.0
    return m.reset_index()


# ------------------------------------------------------------------------ scoring
def zs_vs(s: pd.Series, pool: pd.Series) -> pd.Series:
    """z-score of every value against the mean / spread of the comparison group only."""
    ref = s[pool]
    sd = ref.std(ddof=0)
    return (s - ref.mean()) / (sd if sd else 1.0)


def shrink(q: pd.DataFrame, key: str, prior_starts: float, pool: pd.Series | None = None) -> pd.Series:
    """Blend a QB's stat toward the average starter, weighted by sample size.

    adjusted = (starts * his_value + prior_starts * starter_avg) / (starts + prior_starts)
    With prior_starts=3, a 2-start QB keeps 40% of his own number, an 8-start QB keeps 73%.
    starter_avg comes from the comparison group.
    """
    if prior_starts <= 0:
        return q[key]
    g = q if pool is None else q[pool]
    avg = (g[key] * g["starts"]).sum() / g["starts"].sum()  # per-start average across the comparison group
    return (q["starts"] * q[key] + prior_starts * avg) / (q["starts"] + prior_starts)


# win_pct is already regressed toward .500 when it is built, so it is not shrunk again.
NO_SHRINK = {"win_pct"}


def score(q: pd.DataFrame, cfg: dict, prior_starts: float, pool_min: int | None = None) -> tuple[pd.DataFrame, dict]:
    """Grade every QB in q against the comparison group (QBs with >= pool_min starts; all of q if None).
    QBs outside the group are graded on the same scale but don't affect anyone else's grade."""
    q = q.copy()
    pool = q["starts"] >= pool_min if pool_min else pd.Series(True, index=q.index)
    if pool.sum() < 3:  # too few to compare against: use everyone
        pool = pd.Series(True, index=q.index)
    metrics, context = cfg["metrics"], cfg.get("context", {})
    perf_w = sum(v["weight"] for v in metrics.values())
    total_w = perf_w + sum(v["weight"] for v in context.values())

    cats: dict[str, list[str]] = {}
    for key, spec in metrics.items():
        q[f"s_{key}"] = q[key] if key in NO_SHRINK else shrink(q, key, prior_starts, pool)
        q[f"avg_{key}"] = q.loc[pool, key].mean()  # comparison-group average (for display)
        z = zs_vs(q[f"s_{key}"], pool).clip(-3, 3) * (1 if spec["higher_is_better"] else -1)
        q[f"z_{key}"] = z
        q[f"c_{key}"] = z * spec["weight"] / total_w
        cats.setdefault(spec["category"], []).append(key)
    for key, spec in context.items():
        # Supporting cast: better help -> fewer points. Defenses faced: tougher -> more points.
        sign = 1 if spec.get("credit_when") == "higher" else -1
        q[f"c_{key}"] = sign * q[f"{key}_z"] * spec["weight"] / total_w
        q[f"avg_{key}_grade"] = q.loc[pool, f"{key}_grade"].mean()

    raw = sum(q[f"z_{k}"] * metrics[k]["weight"] for k in metrics) / perf_w
    adj = sum(q[f"c_{k}"] for k in list(metrics) + list(context))
    q["raw_score"] = (50 + 10 * zs_vs(raw, pool)).round(1)
    q["qb_score_exact"] = 50 + 10 * zs_vs(adj, pool)  # unrounded: breaks ties between equal displayed scores
    q["qb_score"] = q["qb_score_exact"].round(1)
    q["cast_adj"] = (q["qb_score"] - q["raw_score"]).round(1)
    q["ovr"] = to_ovr(q["qb_score_exact"])                          # display only
    q["raw_ovr"] = to_ovr(q["raw_score"])
    q["ctx_ovr"] = (q["ovr"] - q["raw_ovr"]).round(1)

    q = q.copy()  # defragment after adding many columns
    for cat, keys in cats.items():
        w = sum(metrics[k]["weight"] for k in keys)
        q[f"cat_{cat}"] = (50 + 10 * sum(q[f"z_{k}"] * metrics[k]["weight"] for k in keys) / w).round(0).astype(int)

    q["in_comparison_group"] = pool
    q = q.sort_values("qb_score_exact", ascending=False, kind="mergesort").reset_index(drop=True)
    q.insert(0, "rank", q.index + 1)
    return q, cats


# ------------------------------------------------------------------------ output
def fmt(x):
    return f"{x:.2f}" if abs(x) < 5 else f"{x:.1f}"


def print_table(q, cats, top, season, min_starts):
    cols = ["rank", "name", "team", "record", "ovr", "raw_ovr", "ctx_ovr",
            *[f"cat_{c}" for c in cats], "weapons_grade", "run_game_grade", "pass_pro_grade", "opp_defense_grade",
            "pass_yds", "pass_td", "ints", "rush_yds", "rush_td", "passer_rating", "epa_per_play"]
    ren = {f"cat_{c}": c[:6] for c in cats} | {
        "rank": "#", "name": "Quarterback", "team": "Tm", "record": "Rec", "ovr": "OVR",
        "raw_ovr": "Raw", "ctx_ovr": "+/-Ctx", "weapons_grade": "Weapons", "run_game_grade": "RunGm",
        "pass_pro_grade": "PassPro", "opp_defense_grade": "OppDef",
        "pass_yds": "PaYd", "pass_td": "PaTD", "ints": "INT", "rush_yds": "RuYd", "rush_td": "RuTD",
        "passer_rating": "Rate", "epa_per_play": "EPA/P"}
    print(f"\n=== {season} NFL Starting QB Rankings  ({len(q)} shown, >= {min_starts} starts; stats from starts only) ===")
    print(f"Everyone is graded against QBs with {COMPARISON_MIN_STARTS}+ starts; the minimum only hides QBs, it never changes scores.")
    print("OVR = Madden-style overall (99.0 needs a score of 70+, 78.0 = average starter). Raw = his own production only. +/-Ctx = OVR gained or lost for supporting cast and defenses faced.")
    print("Weapons / RunGm = quality of the help around him (50 = average team; higher = better help).")
    print("PassPro = his offensive line's pass protection (50 = average; higher = better protection).")
    print("OppDef = toughness of pass defenses in his starts, each rated going into that game (higher = tougher).")
    print("+/-Ctx = points from context: supporting cast (weak help = credit) and defenses faced (tough = credit).")
    print("Stats are pulled toward the starter average for QBs with few starts (see --prior-starts).\n")
    print(q.head(top)[cols].rename(columns=ren).to_string(index=False, float_format=fmt))
    b = q.iloc[0]
    print(f"\n>>> The true best QB of {season}: {b['name']} ({b['team']}) - {b['ovr']:.1f} OVR")


def print_teams(ctx):
    short = {"track_record": "Track", "track_efficiency": "Eff", "star_power": "Star", "depth": "Depth",
             "accolades": "Honors", "pedigree": "Draft", "form_adj": "Form", "sep_adj": "Sep", "yac_oe": "YAC"}
    parts = [k for k in short if f"{k}_grade" in ctx]
    v = ctx.sort_values("weapons_z", ascending=False)[
        ["weapons_grade", *[f"{k}_grade" for k in parts], "run_game_grade", "rb_epa_per_carry", "rb_ypc", "ryoe_per_att",
         "pass_pro_grade", "pressure_rate", "sack_rate", "hit_rate", "time_given", "time_to_throw"]]
    v.columns = ["WEAPONS", *[short[k] for k in parts], "RUN GM", "RB EPA/c", "RB YPC", "RYOE/att",
                 "PASS PRO", "Press%", "Sack%", "Hit%", "Given", "TTT"]
    for c in ["Press%", "Sack%", "Hit%"]:
        v[c] = v[c] * 100
    print("\n=== Supporting cast by team (every grade: 50 = league average, +10 = 1 std dev) ===\n")
    print(v.to_string(float_format=lambda x: f"{x:.2f}" if abs(x) < 10 else f"{x:.0f}"))


def weapons_note(r) -> str:
    """One-line explanation of a QB's weapons grade, including games he missed."""
    if not r["wo_games"] or pd.isna(r["wo_grade"]):
        return f"Weapons grade {r['weapons_grade']} (team grade; he hasn't missed a game)."
    verdict = ("held up without him" if r["wo_grade"] >= r["weapons_team_grade"] + 3 else
               "dropped off without him" if r["wo_grade"] <= r["weapons_team_grade"] - 3 else
               "played about the same without him")
    g = lambda k: "n/a" if pd.isna(r.get(f"wo_{k}_grade")) else f"{r[f'wo_{k}_grade']:.0f}"
    cpoe = "" if pd.isna(r.get("wo_passer_cpoe")) else f", backup CPOE {r['wo_passer_cpoe']:+.1f}"
    return (f"Weapons grade {r['weapons_grade']}: team grade {r['weapons_team_grade']}, blended "
            f"{r['wo_share'] * 100:.0f}% with how they played in the {r['wo_games']} game(s) he missed "
            f"(grade {r['wo_grade']:.0f}). His receivers {verdict}. In those games - getting open {g('getting_open')}, "
            f"after the catch {g('after_catch')}, production adjusted for the backup's accuracy "
            f"{g('production_adj')} (raw {r['wo_ypt']:.1f} yds/target{cpoe}).")


def int_note(r) -> str:
    """One-line summary of his interceptions vs. interception-worthy throws."""
    if pd.isna(r.get("tq_worthy")):
        return f"Interceptions: {int(r['ints'])} ({r['int_rate']:.1f}% of attempts). No throw charting for this season."
    return (f"Interceptions: {int(r['ints'])} ({r['int_rate']:.1f}%). Interception-worthy throws: {int(r['tq_worthy'])} "
            f"({r['int_worthy_rate']:.1f}% of attempts, league {r['tq_league_rate']:.1f}%). "
            f"{int(r['tq_worthy_dropped'])} bad throw(s) dropped by defenders; "
            f"{int(r['tq_int_not_worthy'])} interception(s) not his fault (tips, drops). "
            f"Based on {int(r['tq_charted'])} charted attempts (the newest games may not be charted yet).")


def oline_note(r) -> str:
    """One-line summary of his pass protection."""
    base = f"Pass protection grade {r['pass_pro_grade']} (his line; 50 = average, higher = better protection)."
    if pd.isna(r.get("pp_pressure_pct")):
        return base + " No pressure data for this season."
    return (f"{base} In his starts: pressured on {r['pp_pressure_pct']:.1f}% of dropbacks "
            f"(league {r['pp_league_pressure_pct']:.1f}%), hit {int(r['pp_times_hit'])} times, "
            f"hurried {int(r['pp_times_hurried'])}, sacked {int(r['pp_times_sacked'])}, "
            f"blitzed {int(r['pp_times_blitzed'])}. Time his line gave him {r['pp_time_given']:.2f}s vs. time he "
            f"took to throw {r['pp_ttt']:.2f}s (league {r['pp_league_ttt']:.2f}s): margin {r['pp_margin']:+.2f}s "
            f"({'gets it out before pressure' if r['pp_margin'] > 0.1 else 'holds it into pressure' if r['pp_margin'] < -0.1 else 'about even'}).")


def explain(q, cfg, who):
    hit = q[q["name"].str.contains(who, case=False, regex=False)]
    if hit.empty:
        sys.exit(f"No starting QB matching '{who}'.")
    r = hit.iloc[0]
    print(f"\n#{r['rank']} {r['name']} ({r['team']}, {r['record']} in {r['starts']} starts)  "
          f"{r['ovr']:.1f} OVR  (raw {r['raw_ovr']:.1f}, context {r['ctx_ovr']:+.1f})  [score {r['qb_score']}]")
    print(weapons_note(r))
    print(oline_note(r))
    print(int_note(r))
    print(f"Defenses faced grade {r['opp_defense_grade']} (pass-defense rank going into each game): {r['schedule']}\n")
    print(f"{'Metric':<14}{'Actual':>9}{'Sample-adj':>12}{'Starter avg':>13}{'Points':>8}")
    rows = [(spec["label"], r[k], r[f"s_{k}"], r[f"avg_{k}"], r[f"c_{k}"]) for k, spec in cfg["metrics"].items()]
    rows += [(spec["label"] + " grade", r[f"{k}_grade"], r[f"{k}_grade"], r[f"avg_{k}_grade"], r[f"c_{k}"])
             for k, spec in cfg.get("context", {}).items()]
    for label, val, adj, avg, c in sorted(rows, key=lambda t: -abs(t[4])):
        print(f"{label:<14}{val:>9.2f}{adj:>12.2f}{avg:>13.2f}{c * 100:>8.1f}")
    print("\nSample-adj = actual stat pulled toward the starter average based on number of starts (what gets scored).")
    print("Points = weighted contribution (x100). Cast grades give points when the help is BELOW average; defenses faced gives points when they were TOUGH.")


# ------------------------------------------------------------------ public API
# These are what the UI (app.py / pages/) and any future features build on.
DEFAULT_MIN_STARTS = 4
DEFAULT_PRIOR_STARTS = 3
# Everyone is graded against QBs with at least this many starts. The minimum-starts setting only decides
# who is SHOWN, so changing it never changes anyone's score or the order between them.
COMPARISON_MIN_STARTS = 4


def current_season() -> int:
    today = dt.date.today()
    return today.year if today.month >= 9 else today.year - 1


# ---- Madden-style OVR display (does NOT change any math: a monotonic relabel of the 50-point score)
# A score of 70 or more (two standard deviations above an average starter) is needed for 99.0 OVR.
# An average starter (score 50) = 78.0 OVR. Above average, OVR climbs 1.05 per score point to 99.0 at 70.
# Below average it falls more gently, so the worst starters land in the 60s. Shown to one decimal.
OVR_AVG = 78
OVR_ANCHOR_SCORE = 70        # minimum score for 99.0 OVR
OVR_DOWN_SLOPE = 0.6         # OVR per score point below average
OVR_UP_SLOPE = (99 - OVR_AVG) / (OVR_ANCHOR_SCORE - 50)


def to_ovr(score):
    """50-point score (50 = average, +10 = 1 std dev) -> Madden-style OVR (40.0-99.0, one decimal)."""
    s = pd.Series(score, dtype=float) if not np.isscalar(score) else pd.Series([score], dtype=float)
    o = np.where(s >= 50, OVR_AVG + OVR_UP_SLOPE * (s - 50), OVR_AVG + OVR_DOWN_SLOPE * (s - 50))
    o = pd.Series(np.clip(np.round(o, 1), 40, 99), index=s.index).where(s.notna())
    return o if not np.isscalar(score) else (None if pd.isna(o.iloc[0]) else float(o.iloc[0]))


def load_weights(path: Path | None = None) -> dict:
    return json.loads(Path(path or weights_path()).read_text(encoding="utf-8"))


def _ngs_totals(weekly: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Rebuild per-player Next Gen Stats season lines from weekly rows (so a week cutoff never sees
    later weeks). Weekly rows only exist for players who qualified that week."""
    if weekly is None or weekly.empty:
        return pd.DataFrame(columns=weekly.columns if weekly is not None else [])
    w = weekly.copy()
    if kind == "rec":
        avg_by = {"avg_separation": "targets", "avg_intended_air_yards": "targets", "avg_cushion": "targets",
                  "avg_yac_above_expectation": "receptions"}
        sums = ["targets", "receptions"]
    else:
        avg_by = {"rush_yards_over_expected_per_att": "rush_attempts"}
        sums = ["rush_attempts"]
    for c, by in avg_by.items():
        w[f"_{c}"] = w[c] * w[by]
    g = w.sort_values("week").groupby("player_gsis_id")
    out = g[sums].sum()
    for c, by in avg_by.items():
        out[c] = g[f"_{c}"].sum() / out[by].replace(0, np.nan)
    keep = ["team_abbr", "player_display_name"] + (["player_position"] if "player_position" in w else [])
    out = out.join(g[keep].last()).reset_index()
    out["week"] = 0
    return out


def filter_weeks(data: dict[str, pd.DataFrame], first: int, last: int) -> dict[str, pd.DataFrame]:
    """Restrict game-level data to weeks first..last (inclusive).

    Next Gen Stats season lines (separation, YAC over expected, RB rushing yards over expected) are
    rebuilt from the weekly rows in that range, so nothing from later weeks leaks in.
    """
    out = dict(data)
    if "ngs_rec_weekly" in data:
        out["ngs_rec"] = _ngs_totals(data["ngs_rec_weekly"][data["ngs_rec_weekly"]["week"].between(first, last)], "rec")
    if "ngs_rush_weekly" in data:
        out["ngs_rush"] = _ngs_totals(data["ngs_rush_weekly"][data["ngs_rush_weekly"]["week"].between(first, last)], "rush")
    out["weekly_all"] = data.get("weekly_all", data["weekly"])  # unfiltered, for "rating going into the week"
    out["weekly"] = data["weekly"][data["weekly"]["week"].between(first, last)]
    out["games"] = data["games"][data["games"]["week"].between(first, last)]
    out["ngs_rec_weekly"] = data["ngs_rec_weekly"][data["ngs_rec_weekly"]["week"].between(first, last)]
    for k in ("pfr_pass", "ngs_pass_weekly", "throws"):
        if len(data.get(k, [])):
            out[k] = data[k][data[k]["week"].between(first, last)]
    return out


def rank(data: dict[str, pd.DataFrame], cfg: dict, min_starts: int = DEFAULT_MIN_STARTS,
         prior_starts: float = DEFAULT_PRIOR_STARTS, weeks: tuple[int, int] | None = None,
         all_starters: bool = False):
    """Full pipeline. Returns (ranked QB table, {category: [metric keys]}, team context table).

    weeks=(n, n) ranks a single week; use min_starts=1, prior_starts=0 for that.
    """
    full = data
    if weeks:
        # Supporting cast = season to date through the end of the range (one game is too noisy, and no peeking ahead)
        ctx = team_context(filter_weeks(data, 1, weeks[1]), cfg.get("weapon_parts"), cfg.get("pass_pro_parts"))
        data = filter_weeks(data, *weeks)
    else:
        ctx = team_context(data, cfg.get("weapon_parts"), cfg.get("pass_pro_parts"))
    # A single week (or short range) gets its own group, since nobody has 4 starts in it
    if weeks and weeks[0] == weeks[1]:
        mw = missing_weapons(full, weeks[0], cfg.get("missing_players"), cfg.get("weapon_parts"))
        ctx = ctx.copy().join(mw, how="left")
        ctx["weapons_delta_z"] = ctx["weapons_delta_z"].astype(float).fillna(0)
        ctx["weapons_z"] = ctx["weapons_z"] + ctx["weapons_delta_z"]  # missing key weapons = less help that game
        ctx["weapons_grade"] = (50 + 10 * ctx["weapons_z"]).round(0).astype(int)
    pool_min = min_starts if weeks else COMPARISON_MIN_STARTS
    built = build_qbs(data, ctx, min(min_starts, pool_min), cfg.get("without_qb"), cfg.get("defense"), pool_min,
                      cfg.get("missing_players"), all_starters)
    if "missing_weapons" in ctx:
        built["missing_weapons"] = built["team"].map(ctx["missing_weapons"])
    q, cats = score(built, cfg, prior_starts, pool_min)
    q = q[q["starts"] >= min_starts].reset_index(drop=True)  # display filter only
    q["rank"] = q.index + 1
    return q, cats, ctx


def played_weeks(data: dict[str, pd.DataFrame]) -> list[int]:
    """Regular-season weeks with completed games."""
    return sorted(int(w) for w in data["games"]["week"].unique())


def week_rankings(data: dict[str, pd.DataFrame], cfg: dict, week: int):
    """Every QB who started in `week`, ranked on that game alone.

    His stats come only from that game (no small-sample adjustment), the defense he faced is rated
    going into that week, and his supporting cast is graded on the season through that week.
    50 = average starter that week.
    """
    return rank(data, cfg, min_starts=1, prior_starts=0, weeks=(week, week))


def best_by_week(data: dict[str, pd.DataFrame], cfg: dict) -> pd.DataFrame:
    """The top QB of every week, with the runner-up and his stat line."""
    rows = []
    for w in played_weeks(data):
        q = week_rankings(data, cfg, w)[0]
        if q.empty:
            continue
        b = q.iloc[0]
        second = q.iloc[1] if len(q) > 1 else None
        rows.append({
            "week": w, "name": b["name"], "team": b["team"], "opp": b.get("last_opp"),
            "opp_rank": b.get("last_opp_rank"), "result": b["record"], "qb_score": b["qb_score"], "ovr": b["ovr"],
            "pass_yds": b["pass_yds"], "pass_td": b["pass_td"], "ints": b["ints"], "rush_yds": b["rush_yds"],
            "rush_td": b["rush_td"], "passer_rating": b["passer_rating"], "epa_per_play": b["epa_per_play"],
            "missing_weapons": "" if pd.isna(b.get("missing_weapons")) else b.get("missing_weapons"),
            "opp_missing": "" if pd.isna(b.get("last_opp_missing")) else b.get("last_opp_missing"),
            "runner_up": None if second is None else f"{second['name']} ({second['ovr']:.1f})",
            # margin in the underlying score: single-game OVRs often both hit 99
            "margin": None if second is None else round(b["qb_score_exact"] - second["qb_score_exact"], 1),
            "starters": len(q),
        })
    return pd.DataFrame(rows)


def breakdown(q: pd.DataFrame, cfg: dict, player_id: str) -> pd.DataFrame:
    """Metric-by-metric contribution table for one QB (sorted by impact)."""
    r = q[q["player_id"] == player_id].iloc[0]
    rows = [{"Metric": s["label"], "Category": s["category"], "Actual": r[k], "Sample-adj": r[f"s_{k}"],
             "Starter avg": r.get(f"avg_{k}", q[k].mean()), "Points": r[f"c_{k}"] * 100} for k, s in cfg["metrics"].items()]
    rows += [{"Metric": s["label"] + " grade", "Category": "Supporting Cast", "Actual": r[f"{k}_grade"],
              "Sample-adj": r[f"{k}_grade"], "Starter avg": r.get(f"avg_{k}_grade", q[f"{k}_grade"].mean()),
              "Points": r[f"c_{k}"] * 100}
             for k, s in cfg.get("context", {}).items()]
    out = pd.DataFrame(rows)
    return out.reindex(out["Points"].abs().sort_values(ascending=False).index).reset_index(drop=True)


# ------------------------------------------------------------------------- CLI
def main():
    p = argparse.ArgumentParser(description="Rank NFL starting quarterbacks with a weighted, cast-adjusted model.")
    p.add_argument("--season", type=int, default=current_season())
    p.add_argument("--top", type=int, default=32)
    p.add_argument("--min-starts", type=int, default=DEFAULT_MIN_STARTS, help="only SHOW QBs with at least this many starts (default 4). Scores never change: everyone is "
                        "graded against QBs with 4+ starts")
    p.add_argument("--prior-starts", type=float, default=DEFAULT_PRIOR_STARTS,
                   help="small-sample shrinkage: how many 'average starter' games to blend into each QB's stats "
                        "(default 3; 0 = off; higher = trust small samples less)")
    p.add_argument("--weights", type=Path, default=HERE / "weights.json")
    p.add_argument("--explain", metavar="NAME", help="metric-by-metric breakdown for one QB")
    p.add_argument("--teams", action="store_true", help="also print every team's weapons / run game grades")
    p.add_argument("--week", type=int, help="rank one week: every QB who started that week, on that game alone")
    p.add_argument("--weekly", action="store_true", help="print the best QB of every week so far")
    p.add_argument("--csv", type=Path, help="write the full ranking table to CSV")
    p.add_argument("--refresh", action="store_true", help="force re-download of data")
    a = p.parse_args()

    cfg = load_weights(a.weights)
    data = load_data(a.season, a.refresh)
    if a.weekly:
        b = best_by_week(data, cfg)
        b["opp"] = [f"{o} (#{int(r)} pass D)" if pd.notna(r) else str(o) for o, r in zip(b["opp"], b["opp_rank"])]
        cols = {"week": "Wk", "name": "Best QB", "team": "Tm", "opp": "Opponent", "result": "W/L", "ovr": "OVR",
                "pass_yds": "PaYd", "pass_td": "PaTD", "ints": "INT", "rush_yds": "RuYd", "rush_td": "RuTD",
                "passer_rating": "Rate", "epa_per_play": "EPA/P", "runner_up": "Runner-up (OVR)", "margin": "Margin (score)",
                "missing_weapons": "His missing weapons", "opp_missing": "Opponent missing"}
        print(f"\n=== {a.season} Best QB of every week ===  (each week: that game only; OVR on the same scale as the season; 99.0 needs a score of 70+)\n")
        print(b[list(cols)].rename(columns=cols).to_string(index=False, float_format=fmt))
        return
    if a.week:
        q, cats, ctx = week_rankings(data, cfg, a.week)
        if q.empty:
            sys.exit(f"No completed games in week {a.week}.")
        print(f"\n(Week {a.week} only: stats from that game; defense rated going into the week; "
              f"supporting cast graded on the season through week {a.week}.)")
        print_table(q, cats, a.top, f"{a.season} Week {a.week}", 1)
        print("\nMissing key players that game:")
        for _, r in q.head(a.top).iterrows():
            mw = "" if pd.isna(r.get("missing_weapons")) else r.get("missing_weapons")
            om = "" if pd.isna(r.get("last_opp_missing")) else r.get("last_opp_missing")
            if mw or om:
                print(f"  #{r['rank']} {r['name']}: " + "; ".join(x for x in [
                    f"without {mw}" if mw else "", f"{r['last_opp']} without {om}" if om else ""] if x))
        if a.explain:
            explain(q, cfg, a.explain)
        return
    q, cats, ctx = rank(data, cfg, a.min_starts, a.prior_starts)

    print_table(q, cats, a.top, a.season, a.min_starts)
    if a.teams:
        print_teams(ctx)
    if a.explain:
        explain(q, cfg, a.explain)
    if a.csv:
        q.drop(columns=[c for c in q.columns if c.startswith(("z_", "c_", "s_"))]).to_csv(a.csv, index=False)
        print(f"\nWrote {a.csv}")


if __name__ == "__main__":
    main()
