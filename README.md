# BallKnowerAI — NFL QB Ranker & Game Predictor

A Windows desktop app that answers two questions with public NFL data:

1. **Who is the best starting quarterback in the NFL right now?** Every starter gets a Madden-style **OVR rating** built from his own production and adjusted for the help around him and the defenses he faced.
2. **Who wins this week's games?** A machine-learning model predicts each game's winner, win probability and score, and shows where it disagrees with Vegas and why.

All data is downloaded live from [nflverse](https://github.com/nflverse), so ratings and predictions update as the season goes on.

## Features

- **QB rankings:** 17 weighted stats (EPA per play, ANY/A, CPOE, interception-worthy throws, pocket timing and more), from games each QB **started** only.
- **Context adjustments:** grades for every team's **weapons**, **run game** and **pass protection**, plus the toughness of each **defense faced**, rated going into that game. Weak help earns credit, strong help costs points.
- **Missing players:** key receivers or defenders who didn't play (from snap counts) adjust that game's context.
- **Supporting Cast tab:** click any team to see its pass catchers, each with his own grade.
- **Weekly view:** the best QB of every week, with each week's full ranking.
- **Game predictions:** the model's pick and win %, Vegas's pick and line, predicted score, expected interceptions for each QB, weather forecast, and the key factors behind each pick.
- **Pick tracking:** predictions are saved before kickoff and graded automatically after the games.

## How it works

**Rankings.** Each stat is compared with every QB who has 4+ starts (as z-scores), small samples are pulled toward average (Bayesian shrinkage), extremes are capped at ±3σ, and the weighted total is shown on a 40–99 OVR scale. The comparison group is fixed, so changing the minimum-starts filter only hides QBs; it never reorders them.

**Team ratings.** Every week, a weighted ridge regression solves offense and defense ratings for all 32 teams at once (opponent-adjusted EPA, passing and rushing separately), with recent games weighted more and garbage time removed.

**Predictions.** About 40 pre-game features (Vegas line, rest, travel, weather, coaching, head-to-head, team ratings, QB rating, supporting cast, injuries) feed a ridge-regression model of the margin, with win % from a logistic link on that margin. It's implemented in NumPy, and its settings are tuned by leave-one-season-out cross-validation.

**No peeking.** Every input for a game uses only information from before that game. That was verified with a truncation test: grades computed with all later weeks deleted are identical.

## Results

Backtest on 2,629 regular-season games (2016–2025), each season predicted by a model trained only on the other seasons:

| | Model | Vegas favorite |
|---|---|---|
| Winners picked correctly | **66.9%** | 66.6% |
| Win-probability error (Brier, lower is better) | 0.2110 | 0.2105 |
| Football stats only, no Vegas line | 63.9% | — |

The win probabilities are well calibrated (for example, its 70–75% picks won 75% of the time). Against the spread there's no reliable edge, so treat the predictions as an explained second opinion, not a betting system.

Ideas that were tested and **left out** because they didn't improve the backtest: turnover-based features (giveaway/takeaway rates, fumble luck, turnover cost by field position and game time) and a QB-vs-defense interception matchup. The matchup is still shown on predictions as context.

## Getting started

**Easiest:** download `QB Ranker.exe` from the [Releases](../../releases) page and double-click it. No Python needed. (Windows may warn that it's unsigned: click *More info → Run anyway*.)

**From source** (Python 3.11+):

```bash
pip install -r requirements.txt
python app.py                       # the desktop app
python qb_rank.py --explain "Josh Allen"
python predict.py --week 10         # predictions for a week
python predict.py --train           # retrain the model (about 15 minutes the first time)
```

`build_exe.bat` rebuilds the standalone `.exe`. See **[README.txt](README.txt)** for the full manual: every weight, stat definition, data source and limitation.

## Project structure

| File | What it does |
|---|---|
| `app.py` | Tkinter desktop app (Rankings, Supporting Cast, Weekly, Predictions) |
| `qb_rank.py` | Ranking engine and command-line tool |
| `predict.py` | Game-prediction features, model, backtest and pick log |
| `stadiums.py` | Stadium locations and time zones (travel and weather) |
| `weights.json` | Every ranking weight, all adjustable |
| `model.json` | The trained prediction model |
| `build_exe.bat` | Builds the standalone Windows executable |

## Data & credits

NFL data from [nflverse](https://github.com/nflverse): play-by-play, player stats, Next Gen Stats, Pro Football Reference advanced stats, FTN charting, snap counts, injury reports, depth charts and schedules. Weather from [Open-Meteo](https://open-meteo.com). Built in Python (pandas, NumPy, Tkinter, PyInstaller) with help from Claude Code.
