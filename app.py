"""
QB Ranker GUI - a desktop window for browsing the rankings.

    py app.py

* Rankings tab: every starter. Click a column header to sort, click a QB to see
  his metric-by-metric breakdown on the right.
* Supporting Cast tab: weapons / run game grades for all 32 teams.
* Change season, minimum starts or small-sample shrinkage in the top bar and
  hit "Update". "Re-download" forces fresh data from nflverse.

Built on the public API in qb_rank.py (rank, breakdown, load_data, load_weights).
"""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import messagebox, ttk

import pandas as pd

import qb_rank as qr
import predict as pr

# (column key, header, width, format)
BASE_COLS = [
    ("rank", "#", 36, "{:.0f}"),
    ("name", "Quarterback", 150, "{}"),
    ("team", "Team", 50, "{}"),
    ("record", "Record", 60, "{}"),
    ("starts", "GS", 36, "{:.0f}"),
    ("ovr", "OVR", 50, "{:.1f}"),
    ("raw_ovr", "Raw", 46, "{:.1f}"),
    ("ctx_ovr", "+/- Ctx", 60, "{:+.1f}"),
]
TAIL_COLS = [
    ("weapons_grade", "Weapons", 64, "{:.0f}"),
    ("run_game_grade", "Run Game", 70, "{:.0f}"),
    ("pass_pro_grade", "Pass Pro", 64, "{:.0f}"),
    ("pp_time_given", "Time Given", 76, "{:.2f}"),
    ("pp_ttt", "Time to Throw", 90, "{:.2f}"),
    ("pocket_margin", "Margin", 60, "{:+.2f}"),
    ("opp_defense_grade", "Opp. D", 56, "{:.0f}"),
    ("pass_yds", "Pass Yds", 66, "{:.0f}"),
    ("pass_td", "Pass TD", 60, "{:.0f}"),
    ("ints", "INT", 40, "{:.0f}"),
    ("int_worthy_rate", "INT-worthy %", 84, "{:.1f}"),
    ("cmp_pct", "Cmp%", 52, "{:.1f}"),
    ("rush_yds", "Rush Yds", 66, "{:.0f}"),
    ("rush_td", "Rush TD", 60, "{:.0f}"),
    ("passer_rating", "Rating", 56, "{:.1f}"),
    ("epa_per_play", "EPA/Play", 64, "{:+.2f}"),
]
TEAM_COLS = [
    ("weapons_rank", "#", 36, "{:.0f}"),
    ("team", "Team", 60, "{}"),
    ("weapons_grade", "Weapons", 70, "{:.0f}"),
    ("separation", "Separation (yd)", 110, "{:.2f}"),
    ("yac_oe", "YAC over exp.", 100, "{:+.2f}"),
    ("run_game_rank", "Run #", 50, "{:.0f}"),
    ("run_game_grade", "Run Game", 70, "{:.0f}"),
    ("rb_epa_per_carry", "RB EPA/carry", 100, "{:+.3f}"),
    ("rb_ypc", "RB YPC", 70, "{:.2f}"),
    ("ryoe_per_att", "RYOE/att", 80, "{:+.2f}"),
    ("pass_pro_rank", "Pass Pro #", 72, "{:.0f}"),
    ("pass_pro_grade", "Pass Pro", 64, "{:.0f}"),
    ("pressure_pct", "Pressure %", 76, "{:.1f}"),
    ("sack_pct", "Sack %", 60, "{:.1f}"),
    ("hit_pct", "QB Hit %", 66, "{:.1f}"),
    ("time_given", "Time Given", 76, "{:.2f}"),
    ("time_to_throw", "Time to Throw", 90, "{:.2f}"),
]
PLAYER_COLS = [
    ("name", "Player", 150, "{}"),
    ("position", "Pos", 40, "{}"),
    ("player_ovr", "OVR", 50, "{:.1f}"),
    ("targets", "Targets", 60, "{:.0f}"),
    ("target_share", "Tgt Share %", 80, "{:.1f}"),
    ("receptions", "Rec", 44, "{:.0f}"),
    ("rec_yds", "Yds", 50, "{:.0f}"),
    ("rec_td", "TD", 36, "{:.0f}"),
    ("ypg", "Yds/G", 56, "{:.1f}"),
    ("track_record_grade", "Track Record", 86, "{:.0f}"),
    ("track_efficiency_grade", "Efficiency", 72, "{:.0f}"),
    ("form_grade", "This Season", 82, "{:.0f}"),
    ("sep_adj_grade", "Separation", 76, "{:.0f}"),
    ("yac_oe_grade", "YAC", 44, "{:.0f}"),
    ("accolades_grade", "Honors", 56, "{:.0f}"),
    ("pedigree_grade", "Draft", 50, "{:.0f}"),
]
PRED_COLS = [
    ("kickoff_disp", "Kickoff", 130, "{}"),
    ("matchup", "Matchup", 110, "{}"),
    ("qbs", "QBs (OVR)", 280, "{}"),
    ("pick", "Pick", 50, "{}"),
    ("pick_pct", "Win %", 56, "{:.0f}"),
    ("vegas_pick", "Vegas Pick", 76, "{}"),
    ("score_disp", "Predicted Score", 130, "{}"),
    ("vegas_line", "Vegas Line", 80, "{}"),
    ("total_line", "O/U", 46, "{:g}"),
    ("edge_disp", "Stats vs Vegas", 130, "{}"),
    ("int_disp", "Expected INTs", 110, "{}"),
    ("weather", "Weather", 150, "{}"),
    ("result", "Result", 150, "{}"),
    ("verdict", "Pick Right?", 76, "{}"),
]
WEEK_BEST_COLS = [
    ("week", "Wk", 36, "{:.0f}"),
    ("name", "Best QB", 150, "{}"),
    ("team", "Team", 46, "{}"),
    ("opp_disp", "Opponent (pass D rank)", 150, "{}"),
    ("result", "W/L", 40, "{}"),
    ("ovr", "OVR", 50, "{:.1f}"),
    ("pass_yds", "Pass Yds", 64, "{:.0f}"),
    ("pass_td", "Pass TD", 56, "{:.0f}"),
    ("ints", "INT", 36, "{:.0f}"),
    ("rush_yds", "Rush Yds", 64, "{:.0f}"),
    ("passer_rating", "Rating", 56, "{:.1f}"),
    ("epa_per_play", "EPA/Play", 64, "{:+.2f}"),
    ("missing_weapons", "His Missing Weapons", 230, "{}"),
    ("opp_missing", "Opponent Missing", 200, "{}"),
    ("runner_up", "Runner-up (OVR)", 170, "{}"),
    ("margin", "Margin (score)", 96, "{:.1f}"),
]
WEEK_COLS = [
    ("rank", "#", 36, "{:.0f}"),
    ("name", "Quarterback", 150, "{}"),
    ("team", "Team", 46, "{}"),
    ("opp_disp", "Opponent (pass D rank)", 150, "{}"),
    ("record", "W/L", 40, "{}"),
    ("ovr", "OVR", 50, "{:.1f}"),
    ("raw_ovr", "Raw", 46, "{:.1f}"),
    ("ctx_ovr", "+/- Ctx", 60, "{:+.1f}"),
    ("pass_yds", "Pass Yds", 64, "{:.0f}"),
    ("pass_td", "Pass TD", 56, "{:.0f}"),
    ("ints", "INT", 36, "{:.0f}"),
    ("rush_yds", "Rush Yds", 64, "{:.0f}"),
    ("rush_td", "Rush TD", 56, "{:.0f}"),
    ("passer_rating", "Rating", 56, "{:.1f}"),
    ("epa_per_play", "EPA/Play", 64, "{:+.2f}"),
    ("weapons_grade", "Weapons", 60, "{:.0f}"),
    ("opp_defense_grade", "Opp. D", 52, "{:.0f}"),
    ("missing_weapons", "His Missing Weapons", 230, "{}"),
    ("last_opp_missing", "Opponent Missing", 200, "{}"),
]
BREAKDOWN_COLS = [
    ("Metric", "Metric", 130, "{}"),
    ("Category", "Category", 100, "{}"),
    ("Actual", "Actual", 70, "{:.2f}"),
    ("Sample-adj", "Sample-adj", 76, "{:.2f}"),
    ("Starter avg", "Starter avg", 80, "{:.2f}"),
    ("Points", "Points", 60, "{:+.1f}"),
]


def fmt(spec: str, v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "-"
    try:
        return spec.format(v)
    except (ValueError, TypeError):
        return str(v)


def ovr_tag(o) -> str:
    """Row color by OVR: 90+ elite, 82-89 above average, 74-81 average, below 74 poor."""
    if o is None or pd.isna(o):
        return "avg"
    return "elite" if o >= 90 else "good" if o >= 82 else "avg" if o >= 74 else "poor"


def score_tag(s: float) -> str:
    if s >= 60:
        return "elite"
    if s >= 52:
        return "good"
    if s >= 45:
        return "avg"
    return "poor"


class Table(ttk.Frame):
    """Treeview + scrollbars with click-to-sort headers."""

    def __init__(self, master, cols, on_select=None, height=20):
        super().__init__(master)
        self.cols = cols
        self.df: pd.DataFrame | None = None
        self.sort_key, self.sort_desc = None, False
        self.tag_fn = None
        self.tree = ttk.Treeview(self, columns=[c[0] for c in cols], show="headings", height=height)
        for key, head, width, _ in cols:
            self.tree.heading(key, text=head, command=lambda k=key: self.sort_by(k))
            anchor = "w" if key in ("name", "Metric", "Category") else "center"
            self.tree.column(key, width=width, minwidth=30, anchor=anchor, stretch=key in ("name", "Metric"))
        ys = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        xs = ttk.Scrollbar(self, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        ys.grid(row=0, column=1, sticky="ns")
        xs.grid(row=1, column=0, sticky="ew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        for tag, bg in [("elite", "#c8efd0"), ("good", "#e6f6e9"), ("avg", "#ffffff"), ("poor", "#fbe3e3"),
                        ("odd", "#f4f6f9"), ("even", "#ffffff"), ("pos", "#e6f6e9"), ("neg", "#fbe3e3")]:
            self.tree.tag_configure(tag, background=bg)
        if on_select:
            self.tree.bind("<<TreeviewSelect>>", lambda e: on_select(self.selected()))

    def load(self, df: pd.DataFrame, tag_fn=None):
        self.df, self.tag_fn = df.reset_index(drop=True), tag_fn
        self.sort_key = None
        self._draw()

    def _draw(self):
        self.tree.delete(*self.tree.get_children())
        for i, r in self.df.iterrows():
            vals = [fmt(f, r.get(k)) for k, _, _, f in self.cols]
            tag = self.tag_fn(r) if self.tag_fn else ("odd" if i % 2 else "even")
            self.tree.insert("", "end", iid=str(i), values=vals, tags=(tag,))
        for key, head, _, _ in self.cols:
            arrow = (" ▼" if self.sort_desc else " ▲") if key == self.sort_key else ""
            self.tree.heading(key, text=head + arrow)

    def sort_by(self, key):
        if self.df is None:
            return
        self.sort_desc = not self.sort_desc if key == self.sort_key else key not in ("rank", "name", "team", "weapons_rank", "run_game_rank", "pass_pro_rank")
        self.sort_key = key
        col = self.df[key]
        if key == "record":  # sort records by win%
            col = self.df["win_pct"] if "win_pct" in self.df else col
        order = col.sort_values(ascending=not self.sort_desc, kind="mergesort").index
        self.df = self.df.loc[order].reset_index(drop=True)
        self._draw()

    def selected(self):
        sel = self.tree.selection()
        return None if not sel or self.df is None else self.df.iloc[int(sel[0])]


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("QB Ranker")
        self.minsize(1000, 600)
        try:
            self.state("zoomed")  # open maximized on Windows
        except tk.TclError:
            self.geometry(f"{self.winfo_screenwidth() - 80}x{self.winfo_screenheight() - 120}+40+40")
        self.cfg = qr.load_weights()
        self.cache: dict[int, dict] = {}
        self.q = None
        self._style()
        self._build()
        self.after(100, self.update_rankings)

    # ------------------------------------------------------------------ layout
    def _style(self):
        s = ttk.Style(self)
        if "vista" in s.theme_names():
            s.theme_use("vista")
        s.configure("Treeview", rowheight=24, font=("Segoe UI", 10))
        s.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))
        s.configure("Title.TLabel", font=("Segoe UI", 16, "bold"))
        s.configure("Banner.TLabel", font=("Segoe UI", 12), foreground="#1d5a2e")
        s.configure("Detail.TLabel", font=("Segoe UI", 11, "bold"))

    def _build(self):
        top = ttk.Frame(self, padding=(12, 10, 12, 4))
        top.pack(fill="x")
        ttk.Label(top, text="NFL Starting QB Rankings", style="Title.TLabel").pack(side="left")

        ctl = ttk.Frame(top)
        ctl.pack(side="right")
        self.season = tk.IntVar(value=qr.current_season())
        self.min_starts = tk.IntVar(value=qr.DEFAULT_MIN_STARTS)
        self.prior = tk.DoubleVar(value=qr.DEFAULT_PRIOR_STARTS)
        for label, var, lo, hi, inc in [("Season", self.season, 2016, qr.current_season(), 1),
                                        ("Min starts", self.min_starts, 1, 17, 1),
                                        ("Shrinkage", self.prior, 0, 10, 1)]:
            ttk.Label(ctl, text=label).pack(side="left", padx=(10, 3))
            ttk.Spinbox(ctl, textvariable=var, from_=lo, to=hi, increment=inc, width=6).pack(side="left")
        self.btn = ttk.Button(ctl, text="Update", command=self.update_rankings)
        self.btn.pack(side="left", padx=(12, 4))
        ttk.Button(ctl, text="Re-download", command=lambda: self.update_rankings(refresh=True)).pack(side="left")

        self.banner = ttk.Label(self, text="", style="Banner.TLabel", padding=(12, 0, 12, 6))
        self.banner.pack(fill="x")

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=12, pady=(0, 6))

        # Rankings tab: table on the left, breakdown on the right
        rk = ttk.Panedwindow(nb, orient="horizontal")
        nb.add(rk, text="  Rankings  ")
        self.rank_holder = ttk.Frame(rk)
        rk.add(self.rank_holder, weight=5)
        self.rank_table = None

        side = ttk.Frame(rk, padding=(10, 0, 0, 0))
        rk.add(side, weight=1)
        self.detail_title = ttk.Label(side, text="Click a quarterback to see why he ranks where he does.",
                                      style="Detail.TLabel", wraplength=460, justify="left")
        self.detail_title.pack(fill="x", pady=(4, 6))
        side.configure(width=470)
        self.detail_table = Table(side, BREAKDOWN_COLS)
        self.detail_table.pack(fill="both", expand=True)
        # Plain-English notes: interceptions, protection, weapons
        self.detail_notes = ttk.Label(side, wraplength=460, justify="left", text="")
        self.detail_notes.pack(fill="x", pady=(6, 0))
        ttk.Label(side, wraplength=460, justify="left", foreground="#555",
                  text="Points = weighted contribution to his score. Sample-adj = stat pulled toward the "
                       "starter average for QBs with few starts. Cast grades give points when his help is "
                       "below average.").pack(fill="x", pady=(6, 0))

        # Supporting cast tab
        tm = ttk.Frame(nb, padding=(0, 6, 0, 0))
        nb.add(tm, text="  Supporting Cast  ")
        ttk.Label(tm, text="50 = league average; higher = better help for the QB. Click a team to see its weapons.",
                  foreground="#555").pack(anchor="w", pady=(0, 4))
        split = ttk.PanedWindow(tm, orient="vertical")
        split.pack(fill="both", expand=True)
        top_pane, bottom_pane = ttk.Frame(split), ttk.Frame(split)
        split.add(top_pane, weight=3)
        split.add(bottom_pane, weight=2)
        self.team_table = Table(top_pane, TEAM_COLS, on_select=self.show_team_weapons, height=12)
        self.team_table.pack(fill="both", expand=True)
        # A team's pass catchers (fills in when a team is clicked)
        self.weapons_title = ttk.Label(bottom_pane, text="Click a team above to see its weapons.",
                                       style="Detail.TLabel")
        self.weapons_title.pack(anchor="w", pady=(6, 2))
        self.weapons_table = Table(bottom_pane, PLAYER_COLS, height=8)
        self.weapons_table.pack(fill="both", expand=True)
        ttk.Label(bottom_pane, foreground="#555", wraplength=1400, justify="left",
                  text="Player grades compare each player with others at his position (WR vs WR, TE vs TE, RB vs RB): "
                       "50 = average, +10 = one standard deviation better. Separation and YAC are blank for players "
                       "Next Gen Stats doesn't track; their overall grade uses the other parts.").pack(anchor="w", pady=(4, 0))
        self.players = None

        # Weekly tab: best QB of every week, and any week's full ranking
        wkf = ttk.Frame(nb, padding=(0, 6, 0, 0))
        nb.add(wkf, text="  Weekly  ")
        ttk.Label(wkf, foreground="#555", wraplength=1400, justify="left",
                  text="Each week is ranked on that game alone, with OVR on the same scale as the season (99.0 needs a "
                       "score of 70+); Margin shows the gap between the top two in score points. The defense he faced is "
                       "rated going into that week and is weaker if it was missing key players; his supporting cast is "
                       "graded on the season so far and is weaker if he was missing key weapons. "
                       "Click a week to see everyone's ranking for it.").pack(anchor="w", pady=(0, 4))
        wsplit = ttk.PanedWindow(wkf, orient="vertical")
        wsplit.pack(fill="both", expand=True)
        wtop, wbot = ttk.Frame(wsplit), ttk.Frame(wsplit)
        wsplit.add(wtop, weight=2)
        wsplit.add(wbot, weight=3)
        ttk.Label(wtop, text="Best QB of every week", style="Detail.TLabel").pack(anchor="w", pady=(0, 2))
        self.week_best_table = Table(wtop, WEEK_BEST_COLS, on_select=self.show_week, height=10)
        self.week_best_table.pack(fill="both", expand=True)
        self.week_title = ttk.Label(wbot, text="Click a week above to see its full ranking.", style="Detail.TLabel")
        self.week_title.pack(anchor="w", pady=(6, 2))
        self.week_table = Table(wbot, WEEK_COLS, height=12)
        self.week_table.pack(fill="both", expand=True)
        self.week_cache: dict[tuple[int, int], pd.DataFrame] = {}

        # Predictions tab: win probability, predicted score and key factors for every game in a week
        pf = ttk.Frame(nb, padding=(0, 6, 0, 0))
        nb.add(pf, text="  Predictions  ")
        bar = ttk.Frame(pf)
        bar.pack(fill="x", pady=(0, 4))
        ttk.Label(bar, text="Week").pack(side="left")
        self.pred_week = tk.StringVar(value="")
        self.pred_week_box = ttk.Combobox(bar, textvariable=self.pred_week, width=6, state="readonly")
        self.pred_week_box.pack(side="left", padx=(4, 8))
        self.pred_btn = ttk.Button(bar, text="Predict", command=self.load_predictions)
        self.pred_btn.pack(side="left")
        self.pred_status = ttk.Label(bar, text="  Pick a week and click Predict.", foreground="#555")
        self.pred_status.pack(side="left", padx=8)
        self.pred_table = Table(pf, PRED_COLS, on_select=self.show_prediction, height=16)
        self.pred_table.pack(fill="both", expand=True)
        self.pred_detail = ttk.Label(pf, wraplength=1400, justify="left", text="Click a game to see the key factors behind the pick.")
        self.pred_detail.pack(fill="x", pady=(6, 0))
        self.pred_note = ttk.Label(pf, foreground="#555", wraplength=1400, justify="left", text="")
        self.pred_note.pack(fill="x", pady=(4, 0))
        self.pred_cache: dict[tuple[int, int], pd.DataFrame] = {}

        legend = ttk.Frame(self, padding=(12, 0, 12, 2))
        legend.pack(fill="x")
        for txt, bg in [("90+ elite", "#c8efd0"), ("82-89 above avg", "#e6f6e9"), ("74-81 average", "#ffffff"),
                        ("< 74 below avg", "#fbe3e3")]:
            tk.Label(legend, text=f"  {txt}  ", bg=bg, relief="solid", bd=1, font=("Segoe UI", 9)).pack(side="left", padx=3)
        ttk.Label(legend, text="   OVR: 99.0 needs a score of 70+, 78.0 = average starter. Click headers to sort.",
                  foreground="#555").pack(side="left")

        self.status = ttk.Label(self, text="", relief="sunken", anchor="w", padding=(8, 2))
        self.status.pack(fill="x", side="bottom")

    # ------------------------------------------------------------------ data
    def update_rankings(self, refresh=False):
        try:
            season, ms, ps = self.season.get(), self.min_starts.get(), self.prior.get()
        except tk.TclError:
            messagebox.showerror("QB Ranker", "Season, min starts and shrinkage must be numbers.")
            return
        self.btn.state(["disabled"])
        self.config(cursor="watch")
        self.status.config(text=f"Loading {season} data{' (downloading)' if refresh or season not in self.cache else ''}...")

        def work():
            try:
                if refresh or season not in self.cache:
                    self.cache[season] = qr.load_data(season, refresh)
                res = qr.rank(self.cache[season], self.cfg, ms, ps)
                self.after(0, lambda: self.show(res, season, ms))
            except Exception as e:  # surface any failure in the window
                msg = f"{type(e).__name__}: {e}"
                self.after(0, lambda: self.fail(msg))

        threading.Thread(target=work, daemon=True).start()

    def fail(self, msg):
        self.btn.state(["!disabled"])
        self.config(cursor="")
        self.status.config(text="Failed: " + msg)
        messagebox.showerror("QB Ranker", msg)

    def show(self, res, season, min_starts):
        q, cats, ctx = res
        self.q = q
        cat_cols = [(f"cat_{c}", c, 72, "{:.0f}") for c in cats]
        if self.rank_table:
            self.rank_table.destroy()
        self.rank_table = Table(self.rank_holder, BASE_COLS + cat_cols + TAIL_COLS, on_select=self.show_detail)
        self.rank_table.pack(fill="both", expand=True)
        self.rank_table.load(q, tag_fn=lambda r: ovr_tag(r["ovr"]))

        teams = ctx.reset_index().rename(columns={ctx.index.name or "index": "team"})
        # Rank numbers (1 = best help) so each team's standing is readable at a glance, whatever the sort
        teams["weapons_rank"] = teams["weapons_z"].rank(ascending=False, method="min")
        teams["run_game_rank"] = teams["run_game_z"].rank(ascending=False, method="min")
        teams["pass_pro_rank"] = teams["pass_pro_z"].rank(ascending=False, method="min")
        for src, dst in [("pressure_rate", "pressure_pct"), ("sack_rate", "sack_pct"), ("hit_rate", "hit_pct")]:
            teams[dst] = teams[src] * 100 if src in teams else float("nan")
        self.team_table.load(teams.sort_values("weapons_rank"))
        self.players = qr.player_weapons(self.cache[season], self.cfg.get("weapon_parts"))
        self.weapons_table.load(pd.DataFrame(columns=[c[0] for c in PLAYER_COLS]))
        self.weapons_title.config(text="Click a team above to see its weapons.")

        b = q.iloc[0]
        self.banner.config(text=f"★  The true best QB of {season}: {b['name']} ({b['team']}, {b['record']})"
                                f" — {b['ovr']:.1f} OVR")
        self.status.config(text=f"{season} season: showing {len(q)} QBs with >= {min_starts} starts "
                                f"(all graded vs. QBs with {qr.COMPARISON_MIN_STARTS}+ starts). "
                                f"Stats come only from games each QB started.")
        self.btn.state(["!disabled"])
        self.config(cursor="")
        self.rank_table.tree.selection_set("0")
        self.load_weekly(season)
        self.setup_predictions(season)

    # ------------------------------------------------------------------ weekly tab
    def load_weekly(self, season):
        """Build the best-QB-of-every-week table in the background (a few seconds)."""
        self.season_shown = season
        self.week_cache = {}
        self.week_best_table.load(pd.DataFrame(columns=[c[0] for c in WEEK_BEST_COLS]))
        self.week_table.load(pd.DataFrame(columns=[c[0] for c in WEEK_COLS]))
        self.week_title.config(text="Working out the best QB of every week...")

        def work():
            try:
                best = qr.best_by_week(self.cache[season], self.cfg)
                self.after(0, lambda: self.show_weekly(best, season))
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                self.after(0, lambda: self.week_title.config(text="Weekly rankings failed: " + msg))

        threading.Thread(target=work, daemon=True).start()

    def show_weekly(self, best, season):
        if season != getattr(self, "season_shown", season) or best.empty:
            return
        best = best.copy()
        best["opp_disp"] = [f"{o} (#{int(r)})" if pd.notna(r) else str(o) for o, r in zip(best["opp"], best["opp_rank"])]
        self.week_best_table.load(best, tag_fn=lambda r: ovr_tag(r["ovr"]))
        self.week_title.config(text="Click a week above to see its full ranking.")

    def show_week(self, row):
        """Weekly tab: full ranking for the clicked week."""
        if row is None:
            return
        season, week = self.season_shown, int(row["week"])
        if (season, week) in self.week_cache:
            return self._fill_week(season, week)
        self.week_title.config(text=f"Ranking week {week}...")

        def work():
            try:
                self.week_cache[(season, week)] = qr.week_rankings(self.cache[season], self.cfg, week)[0]
                self.after(0, lambda: self._fill_week(season, week))
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                self.after(0, lambda: self.week_title.config(text="Week ranking failed: " + msg))

        threading.Thread(target=work, daemon=True).start()

    def _fill_week(self, season, week):
        q = self.week_cache[(season, week)].copy()
        q["opp_disp"] = [f"{o} (#{int(r)})" if pd.notna(r) else str(o) for o, r in zip(q["last_opp"], q["last_opp_rank"])]
        self.week_table.load(q, tag_fn=lambda r: ovr_tag(r["ovr"]))
        b = q.iloc[0]
        self.week_title.config(text=f"Week {week}: {len(q)} starting QBs — best: {b['name']} ({b['team']}) "
                                    f"vs {b['opp_disp']}, {b['ovr']:.1f} OVR")

    def show_team_weapons(self, team_row):
        """Supporting Cast tab: list the clicked team's pass catchers with their grades."""
        if team_row is None or self.players is None:
            return
        team = team_row["team"]
        roster = qr.team_weapons(self.players, team)
        self.weapons_table.load(roster, tag_fn=lambda r: ovr_tag(r["player_ovr"]) if pd.notna(r["player_ovr"]) else "avg")
        self.weapons_title.config(
            text=f"{team} weapons — team grade {int(team_row['weapons_grade'])} (#{int(team_row['weapons_rank'])} of "
                 f"{len(self.team_table.df)}), {len(roster)} pass catchers")

    # ------------------------------------------------------------------ predictions tab
    def setup_predictions(self, season):
        """Fill the week list for the season and pre-select the next week to be played."""
        try:
            allg = pr.all_games()
            weeks = sorted(int(w) for w in allg.loc[allg["season"] == season, "week"].unique())
            nxt = pr.next_week(allg, season) or (weeks[-1] if weeks else None)
            self.pred_week_box["values"] = weeks
            if nxt:
                self.pred_week.set(str(nxt))
            m = pr.load_model()
            b = m.get("backtest", {})
            self.pred_note.config(text=(
                f"Model trained on {m['trained_on'][0]}-{m['trained_on'][-1]}. In a backtest on {b.get('games', 0):,} games "
                f"(each season predicted by a model that never saw it), it picked {b.get('model_accuracy', 0):.1%} of winners "
                f"vs {b.get('vegas_accuracy', 0):.1%} for the Vegas favorite. 'Stats vs Vegas' is how many points the football "
                f"stats alone (without the Vegas line) lean beyond the spread; when they disagree by 3+ points, that side "
                f"covered {b.get('ats', {}).get('edge_3', {}).get('pct', 0):.1%} of the time (52.4% breaks even). "
                f"Predictions made before kickoff are saved and graded automatically."))
        except FileNotFoundError:
            self.pred_note.config(text="No trained model found (model.json). Run: python predict.py --train")
        except Exception as e:
            self.pred_note.config(text=f"Predictions unavailable: {type(e).__name__}: {e}")
        self.pred_season = season

    def load_predictions(self):
        try:
            season, week = self.pred_season, int(self.pred_week.get())
        except (AttributeError, ValueError):
            return
        if (season, week) in self.pred_cache:
            return self._fill_predictions(season, week)
        self.pred_btn.state(["disabled"])
        self.pred_status.config(text=f"  Predicting week {week} (gathering QB ratings, injuries and weather)...")

        def work():
            try:
                self.pred_cache[(season, week)] = pr.predict_week(season, week)
                self.after(0, lambda: self._fill_predictions(season, week))
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                self.after(0, lambda: (self.pred_status.config(text="  Prediction failed: " + msg),
                                       self.pred_btn.state(["!disabled"])))

        threading.Thread(target=work, daemon=True).start()

    def _fill_predictions(self, season, week):
        self.pred_btn.state(["!disabled"])
        p = self.pred_cache[(season, week)]
        if p.empty:
            self.pred_status.config(text=f"  No games found for week {week}.")
            return
        p = p.copy()
        p["kickoff_disp"] = [k.strftime("%a %b %d %I:%M %p") if pd.notna(k) else "" for k in p["kickoff"]]
        p["matchup"] = p["away_team"] + " @ " + p["home_team"]
        p["qbs"] = [f"{a} ({ao:.1f}) vs {h} ({ho:.1f})" for a, ao, h, ho in
                    zip(p["away_qb_name"].fillna("?"), p["away_qb_ovr"].fillna(0), p["home_qb_name"].fillna("?"), p["home_qb_ovr"].fillna(0))]
        p["pick_pct"] = p["pick_prob"] * 100
        p["score_disp"] = [f"{a} {int(ap)} - {h} {int(hp)}" for a, ap, h, hp in
                           zip(p["away_team"], p["pred_away_pts"], p["home_team"], p["pred_home_pts"])]
        p["edge_disp"] = ["no line yet" if pd.isna(e) else f"{abs(e):.1f} pts toward {h if e > 0 else a}" if abs(e) >= 0.5
                          else "agrees" for e, h, a in zip(p["edge_vs_vegas"], p["home_team"], p["away_team"])]
        p["weather"] = ["Dome" if d else f"{t:.0f}F, wind {w:.0f} mph" + (f", rain {r:.2f} in" if r and r > 0.01 else "")
                        for d, t, w, r in zip(p["dome"], p["temp"], p["wind"], p["rain"])]
        p["verdict"] = ["" if c is None else ("Yes" if c else "No") for c in p["correct"]]
        p["int_disp"] = [f"{a} {ai:.1f} / {h} {hi:.1f}" for a, ai, h, hi in
                         zip(p["away_team"], p["away_exp_int"], p["home_team"], p["home_exp_int"])]
        self.pred_table.load(p, tag_fn=lambda r: "pos" if r["verdict"] == "Yes" else "neg" if r["verdict"] == "No" else
                             ("elite" if r["pick_prob"] >= 0.75 else "good" if r["pick_prob"] >= 0.6 else "avg"))
        played = p["correct"].notna()
        rec = f"  Picks so far this week: {int(p.loc[played, 'correct'].sum())}-{int((~p.loc[played, 'correct'].astype(bool)).sum())}" if played.any() else ""
        try:
            tr = pr.pick_record(season)
            if tr.get("games"):
                rec += (f"   Saved picks this season: {tr['wins']}-{tr['losses']}"
                        f" (vs spread when stats disagree by 3+: {tr['ats_edge3_wins']}-{tr['ats_edge3_losses']})")
        except Exception:
            pass
        lines = p["has_line"] if "has_line" in p else pd.Series(True, index=p.index)
        dis = p[~p["agrees_with_vegas"]] if "agrees_with_vegas" in p else p.iloc[0:0]
        if lines.any():
            rec = (f"  Disagrees with Vegas on {len(dis)}: " + ", ".join(dis["matchup"]) + "." if len(dis) else
                   "  Agrees with Vegas on every game.") + rec
        ahead = int(p["weeks_ahead"].max()) if "weeks_ahead" in p else 0
        if not lines.all():
            rec = (f"  {ahead} week{'s' if ahead != 1 else ''} ahead: no Vegas line"
                   f"{'s' if (~lines).sum() > 1 else ''} yet for {int((~lines).sum())} game(s), so they use the football "
                   f"stats only - in the backtest about 3 points less accurate than with a line, and a bit less for each "
                   f"week further out.") + rec
        self.pred_status.config(text=f"  Week {week}: {len(p)} games.{rec}")

    def show_prediction(self, r):
        if r is None:
            return
        fx = "; ".join(f"{lab}: {abs(v):.1f} pts toward {r['home_team'] if v > 0 else r['away_team']}" for lab, v in r["factors"])
        self.pred_detail.config(text=(
            f"{r['matchup']}: {r['pick']} wins {r['pick_prob']:.0%} of the time. Predicted {r['score_disp']} "
            f"(margin {abs(r['pred_margin']):.1f}, total {r['pred_total']:.0f}); "
            + ("no Vegas line yet, so this uses the football stats only" if pd.isna(r["total_line"]) else
               f"Vegas {r['vegas_line']}, O/U {r['total_line']:g}") + ".\n"
            f"Biggest factors from the football stats: {fx}.\n"
            f"Interception matchup: {r['away_qb_name']} expected ~{r['away_exp_int']:.1f} INTs (throws one on "
            f"{r['away_qb_int_rate']:.1%} of passes; {r['home_team']}'s defense picks off {r['home_def_int_rate']:.1%}). "
            f"{r['home_qb_name']} ~{r['home_exp_int']:.1f} (throws {r['home_qb_int_rate']:.1%}; {r['away_team']}'s "
            f"defense picks off {r['away_def_int_rate']:.1%})."))

    def show_detail(self, r):
        if r is None or self.q is None:
            return
        self.detail_title.config(
            text=f"#{r['rank']}  {r['name']} ({r['team']})  —  {r['record']} in {r['starts']} starts\n"
                 f"{r['ovr']:.1f} OVR   (raw {r['raw_ovr']:.1f}, context {r['ctx_ovr']:+.1f})")
        bd = qr.breakdown(self.q, self.cfg, r["player_id"])
        self.detail_table.load(bd, tag_fn=lambda x: "pos" if x["Points"] > 0.05 else "neg" if x["Points"] < -0.05 else "avg")
        self.detail_notes.config(text="\n\n".join([qr.int_note(r), qr.oline_note(r), qr.weapons_note(r)]))


if __name__ == "__main__":
    App().mainloop()
