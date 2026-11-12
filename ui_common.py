"""Shared helpers for every UI page: cached data loading, current settings, chart colors.

A page gets everything it needs with:

    from ui_common import settings, get_rankings
    s = settings()
    q, cats, ctx = get_rankings(**s)                     # season rankings
    q, cats, ctx = get_rankings(**s, weeks=(5, 5))       # just week 5
"""

from __future__ import annotations

import json

import streamlit as st

import qb_rank as core

FIRST_SEASON = 2016  # Next Gen Stats (used for the weapons/run-game grades) starts in 2016

# Chart colors (validated reference palette)
BLUE = "#2a78d6"   # primary series / "helps"
RED = "#e34948"    # "hurts"
GRAY = "#8a8984"   # reference lines


@st.cache_data(ttl=core.CACHE_HOURS * 3600, show_spinner="Loading NFL data ...")
def get_data(season: int) -> dict:
    return core.load_data(season, refresh=False)


@st.cache_data(show_spinner="Ranking quarterbacks ...")
def _rank(season: int, cfg_json: str, min_starts: int, prior_starts: float, weeks: tuple[int, int] | None):
    return core.rank(get_data(season), json.loads(cfg_json), min_starts, prior_starts, weeks)


def get_rankings(season: int, cfg: dict, min_starts: int, prior_starts: float,
                 weeks: tuple[int, int] | None = None):
    """(ranked QB table, categories, team context). Cached per unique settings."""
    return _rank(season, json.dumps(cfg), min_starts, prior_starts, weeks)


def init_state() -> None:
    """Defaults for settings shared across pages (called once from app.py)."""
    ss = st.session_state
    ss.setdefault("cfg", core.load_weights())
    ss.setdefault("season", core.current_season())
    ss.setdefault("min_starts", core.DEFAULT_MIN_STARTS)
    ss.setdefault("prior_starts", float(core.DEFAULT_PRIOR_STARTS))


def settings() -> dict:
    """The current sidebar + model settings, ready to pass to get_rankings(**settings())."""
    ss = st.session_state
    return {"season": ss["season"], "cfg": ss["cfg"], "min_starts": ss["min_starts"],
            "prior_starts": ss["prior_starts"]}


def available_weeks(season: int) -> list[int]:
    """Regular-season weeks that have completed games with stats."""
    return sorted(int(w) for w in get_data(season)["weekly"]["week"].unique())
