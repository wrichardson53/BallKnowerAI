===============================================================================
                                 QB RANKER
            Who is the true best starting quarterback in the NFL?
===============================================================================

QB Ranker grades every NFL starting quarterback on 17 stats from his own play,
then adjusts for the situation around him: the receivers he throws to, his run
game, his offensive line, and the defenses he faced. Every grade is built from
real, public NFL data that the program downloads itself, so the rankings update
as the season goes on.

CONTENTS
  1.  Getting started
  2.  How a QB's score is calculated
  3.  The weights (how much each thing counts)
  4.  What each QB stat means
  5.  Context grades: weapons, run game, pass protection, defenses faced
  6.  Special adjustments
  7.  Weekly rankings
  8.  Game predictions
  9.  Where the data comes from
  10. Changing the weights
  11. Files in this folder
  12. Limitations
  13. Credits


-------------------------------------------------------------------------------
1. GETTING STARTED
-------------------------------------------------------------------------------

THE EASY WAY (no Python needed)
  Double-click  dist\QB Ranker.exe

  - The first launch downloads about 33 MB of stats, which takes about a minute.
  - If Windows shows "Windows protected your PC", click More info -> Run anyway.
    The program isn't digitally signed, so Windows warns about it.
  - The app has three tabs:
      Rankings        Every starting QB. Click a QB to see exactly which stats
                      raise and lower his score.
      Supporting Cast Every team's weapons, run game and pass protection grades.
                      Click a team to see its receivers, tight ends and backs,
                      each with his own grade.
      Weekly          The best QB of every week. Click a week to see every
                      starter's ranking for that week.
      Predictions     Pick a week and click Predict: each game's pick and
                      win probability, Vegas's pick and line, the predicted
                      score, expected interceptions for each QB, the
                      weather, and the key factors behind the pick. The
                      status line lists the games where the model and
                      Vegas disagree.
  - Change the season, minimum starts or small-sample adjustment in the top bar
    and click Update. Click Re-download to fetch the newest stats right away.

  The program can be shared: send someone the .exe and they double-click it.
  It runs on Windows only.

  Where the app keeps its files (on each computer it runs on):
    %LOCALAPPDATA%\QB Ranker\data          downloaded stats
    %LOCALAPPDATA%\QB Ranker\weights.json  that computer's editable weights

RUNNING FROM THE CODE (for developers)
  Needs Python 3.11+ with pandas and numpy.
    python app.py                         the desktop app
    python qb_rank.py                     season rankings in the terminal
    python qb_rank.py --explain "Josh Allen"   one QB's full breakdown
    python qb_rank.py --teams             every team's supporting-cast grades
    python qb_rank.py --weekly            the best QB of every week
    python qb_rank.py --week 9            everyone's ranking for week 9
    python qb_rank.py --season 2025       any season from 2016 on
    python qb_rank.py --min-starts 6      only show QBs with 6+ starts
    python qb_rank.py --csv out.csv       save the full table
    python predict.py                     predictions for the next week
    python predict.py --week 10           predictions for week 10
    python predict.py --train             retrain the prediction model

REBUILDING THE .EXE AFTER CHANGING THE CODE
  Double-click build_exe.bat. It creates a new dist\QB Ranker.exe in about a
  minute. The first time, it also sets up its own build environment in
  %LOCALAPPDATA%\QBRankerBuild.


-------------------------------------------------------------------------------
2. HOW A QB'S SCORE IS CALCULATED
-------------------------------------------------------------------------------

Step 1  Starters only, starts only.
        Only QBs who were their team's main starter, or started its latest game,
        are ranked. Only games a QB started count toward his stats and record.
        The starter of each game is whoever had the most dropbacks for his team,
        taken from the box score. (The schedule's listed starter is often wrong.)

Step 2  Small samples are pulled toward average.
        Each stat is blended with 3 games of average-starter play:
            adjusted = (starts x his stat + 3 x starter average) / (starts + 3)
        A 4-start QB keeps 57% of his own number, a 9-start QB keeps 75%.
        (Win % has its own adjustment instead; see section 4.)

Step 3  Every stat is compared with the other starters.
        Each stat becomes "standard deviations above or below the average
        starter," so yards, percentages and seconds can be added together.
        For stats where lower is better (interceptions, sacks), the sign flips.
        Each stat is capped at +/-3 standard deviations, so one extreme number
        can't take over a score.

        The comparison group is always QBs with 4+ starts. The "minimum starts"
        setting only decides who is SHOWN. Changing it never changes anyone's
        score or the order between QBs.

Step 4  Weights are applied.
        Each stat's comparison is multiplied by its weight (section 3). These
        are the "Points" shown in a QB's breakdown.

Step 5  Context is added.
        Weapons, run game, pass protection and defenses faced get points the
        same way. Weak help earns credit and strong help costs points. A tough
        schedule earns credit and an easy one costs points.

Step 6  The final scale.
        The points are added up and rescaled so that:
            50 = average starter
            60 = one standard deviation better (very good)
            70 = two standard deviations better (elite)
        "Raw" is the same calculation using only the QB's own stats, with no
        context.

Step 7  OVR (what the app shows).
        That score is shown as a Madden-style OVR rating. This only relabels
        the number: it doesn't change any math, and the ranking order is
        exactly the same.
          - OVR is shown to one decimal place (for example 97.7).
          - A score of 70 or more (two standard deviations above an average
            starter) is needed for 99.0 OVR. Anything higher also shows 99.0,
            and the rank number shows who's first.
          - Average starter (score 50) = 78.0 OVR.
          - Above average: +1.05 OVR per score point, reaching 99.0 at 70.
          - Below average: +0.6 OVR per score point, so the weakest starters
            land in the low-to-mid 60s instead of the 40s.
              OVR = 78 + 1.05 x (score - 50)   when score >= 50
              OVR = 78 + 0.6  x (score - 50)   when score <  50
              then rounded to one decimal and kept between 40.0 and 99.0
          - For reference: Matthew Stafford's 2025 season (score 63.7) is 92.4,
            Patrick Mahomes's 2022 season (74.8, the highest since 2016) is 99.0.
          - Raw OVR = the same conversion of his production-only score.
            +/- Ctx = OVR minus Raw OVR (what the context added or took away).
          - QBs who round to the same OVR stay ordered by their exact score.
          - Receivers' player grades on the Supporting Cast tab use the same
            OVR conversion.
          - Row colors: 90+ elite, 82-89 above average, 74-81 average,
            below 74 below average.
          - Weekly rankings use the same scale. The weekly "Margin" is shown in
            score points.
          - The 99.0 threshold (OVR_ANCHOR_SCORE = 70) is set in qb_rank.py.

The other grades in the app (category grades like Passing and Efficiency, team
grades like Weapons and Pass protection, and each part of them) stay on the
50-point scale: 50 = average, +10 = one standard deviation better.


-------------------------------------------------------------------------------
3. THE WEIGHTS
-------------------------------------------------------------------------------

Weights are relative: only their size compared with each other matters. They
add up to 143, and the share is each weight's part of that total.

  CATEGORY        STAT                      WEIGHT   SHARE   BETTER IS
  --------------  ------------------------  ------  ------   ---------
  Efficiency      EPA per play                  14    9.8%   higher
  (28.0%)         ANY/A                         10    7.0%   higher
                  Passer rating                  8    5.6%   higher
                  CPOE                           5    3.5%   higher
                  1st-down rate                  3    2.1%   higher

  Passing         Pass yards per game           10    7.0%   higher
  (18.2%)         Pass TDs per game             10    7.0%   higher
                  Completion %                   4    2.8%   higher
                  Attempts per game              2    1.4%   higher

  Ball security   INT-worthy throw %             6    4.2%   lower
  (12.6%)         Interception %                 5    3.5%   lower
                  Sack %                         4    2.8%   lower
                  Fumbles lost per game          3    2.1%   lower

  Winning (8.4%)  Win % as a starter            12    8.4%   higher

  Rushing         Rush yards per game            6    4.2%   higher
  (7.0%)          Rush TDs per game              4    2.8%   higher

  Pocket (3.5%)   Time margin                    5    3.5%   higher

  Context         Weapons                       10    7.0%   weak help = credit
  (22.4%)         Pass protection                8    5.6%   weak line = credit
                  Defenses faced                 8    5.6%   tough = credit
                  Run game                       6    4.2%   weak help = credit

  Totals: the QB's own production = 77.6%, context = 22.4%.


-------------------------------------------------------------------------------
4. WHAT EACH QB STAT MEANS
-------------------------------------------------------------------------------

EFFICIENCY
  EPA per play     Expected Points Added per play, on passes, scrambles and
                   designed runs. EPA measures how much each play changed the
                   team's expected points, based on down, distance and field
                   position. It's the best single measure of how much a QB
                   helps his offense score.
  ANY/A            Adjusted Net Yards per Attempt:
                   (pass yds + 20 x TD - 45 x INT - sack yds) / (attempts + sacks)
                   Yards per throw that rewards TDs and penalizes INTs and sacks.
  Passer rating    The official NFL passer rating, 0 to 158.3.
  CPOE             Completion % Over Expected. How much more often he completes
                   passes than expected, given how hard each throw was (depth,
                   coverage and so on).
  1st-down rate    Share of his plays (passes, sacks and runs) that gained a
                   first down.

PASSING
  Pass yds / game, pass TDs / game, attempts / game: volume per start.
  Completion %     Completions / attempts.

BALL SECURITY
  INT-worthy %     Throws a defender should have intercepted, per 100 charted
                   attempts, whether or not he did. Credited to the passer
                   using play-by-play data. This rewards good decisions over
                   luck: dropped interceptions still count against him, and
                   tipped passes that bounce off receivers into a defender's
                   hands do not.
  Interception %   Actual interceptions per attempt.
  Sack %           Sacks per dropback.
  Fumbles lost     Lost fumbles per game, on sacks and runs.

WINNING
  Win %            His record as a starter, pulled toward .500 by adding two
                   wins and two losses, so a 2-0 QB isn't treated as perfect:
                       (wins + 0.5 x ties + 2) / (starts + 4)

RUSHING
  Rush yds / game, rush TDs / game. Rushing also counts in EPA per play.

POCKET
  Time margin      Time his line gives him before pressure arrives, minus the
                   time he takes to throw. Positive = he gets the ball out
                   before his protection breaks down. Negative = he holds it
                   into pressure.
                   "Time given" is an estimate. Across every game, the program
                   measures how pressure rises the longer QBs hold the ball.
                   Time given is when his line's pressure would reach the
                   league-average rate (about 23% of dropbacks). An average line
                   gives about the league-average time to throw, roughly 2.8
                   seconds.


-------------------------------------------------------------------------------
5. CONTEXT GRADES
-------------------------------------------------------------------------------

Each one is a team grade (50 = league average). A QB who started for two
teams gets a blend weighted by his starts with each.

WEAPONS (his WRs, TEs and RBs as pass catchers)
  Each player counts in proportion to his targets this season, so injured or
  departed players drop out right away. Players are compared with others at
  their own position (100 = average WR / TE / RB), so a team that throws to its
  backs and tight ends isn't penalized for their naturally lower yardage.

    PART                          WEIGHT  WHAT IT MEASURES
    Track record                     25   Receiving yards per game: this season
                                          (counts 3x), last season (2x) and two
                                          seasons ago (1x), on any team with any
                                          QB. Rookies are rated by what players
                                          drafted in the same round have done.
    Separation (depth-adjusted)      15   How open receivers get when the ball
                                          arrives, compared with what's expected
                                          for how deep the routes are and how
                                          far off defenders play (Next Gen Stats).
    YAC over expected                15   Yards after the catch beyond what the
                                          tracking data predicts.
    Efficiency                       10   Receiving yards per target (same
                                          seasons as track record).
    Star power                       10   Track record of the team's best pass
                                          catcher.
    This season's form               10   This season's yards and EPA per
                                          target, after removing what the QB's
                                          accuracy (CPOE) explains, so the QB
                                          doesn't inflate it.
    Depth                             5   Track record of the third-best option.
    Pro Bowls / All-Pros              5   Career honors (All-Pro counts double).
    Draft pedigree                    5   How high the players were drafted.

  About 59% of the weapons grade comes from this season and 41% from past
  seasons. The share for this season grows as more games are played.

  Clicking a team on the Supporting Cast tab shows each player's own grade,
  built from the same parts (except star power and depth, which only apply
  to a team). Players with few targets are pulled toward 50.

RUN GAME (non-QB rushing)
  Equal parts: rushing EPA per carry, yards per carry, and running backs'
  rushing yards over expected (Next Gen Stats, 2018 and later).

PASS PROTECTION (his offensive line)
  Based on every dropback behind that line this season, by all of the team's
  QBs.
    Pressure rate allowed            40
    Sack rate allowed                20
    QB hit rate allowed              15
    Time given by the line           15   (see Time margin above)
    Hurry rate allowed               10
  Pressure, sack, hit and hurry rates are judged after removing what the QBs'
  time to throw and the blitzes they faced explain. A QB who holds the ball
  doesn't sink his line's grade, and a line that faces lots of blitzes isn't
  blamed for it.

DEFENSES FACED
  Each pass defense is rated on what opposing QBs did against it, using only
  the games BEFORE the week it was played:
    EPA per dropback allowed         50
    ANY/A allowed                    30
    Sack rate                        20
  Early in the season, last season's rating counts as 4 games and fades as
  real games are played. A QB's grade is the average toughness of the
  defenses in his starts. Example: if a defense was the #10 pass defense going
  into its game against him, it counts as #10, even if it falls apart later.


-------------------------------------------------------------------------------
6. SPECIAL ADJUSTMENTS
-------------------------------------------------------------------------------

GAMES THE QB MISSED
  If a QB missed games, his weapons grade is blended with how his receivers
  played without him. They're judged only on what receivers control, so an
  inaccurate backup doesn't count against them:
    Getting open (depth-adjusted separation)     40
    Production after removing the backup's
      accuracy (yards and EPA per target)        40
    YAC over expected                            20
  Receivers who held up = more credit goes to them. Receivers who fell off =
  he was elevating them. How much it counts depends on games missed:
      share = 50% x games missed / (games missed + 3)
  So 1 game missed = 12%, 3 games = 25%, 5 games = 31%.

MISSING KEY PLAYERS (from snap counts: 0 snaps = missed the game)
  Key weapon out (weekly rankings):
    A pass catcher with 10%+ of his team's targets who didn't play. His targets
    are treated as going to a typical backup (player grade 42), which lowers
    the QB's weapons grade for that game. The QB gets more credit.
  Key defender out (season and weekly rankings):
    Each defense's top 4 players, by production in earlier weeks (sacks, QB
    hits, interceptions, passes defended, tackles for loss, forced fumbles)
    plus Pro Bowls. Any who didn't play make that defense easier for the game,
    and the QB gets less credit. Losing all 4 makes a defense one standard
    deviation weaker. In a QB's schedule, a * marks games where the opponent
    was short-handed.


-------------------------------------------------------------------------------
7. WEEKLY RANKINGS
-------------------------------------------------------------------------------

  - Only that game counts, with no small-sample adjustment.
  - The defense he faced is rated going into that week, and is weaker if it was
    missing key players.
  - His supporting cast is graded on the season through that week (one game is
    too noisy, and a week never uses later games), and is weaker if he was
    missing key weapons.
  - Under the hood, a score of 50 = average starter that week (78.0 OVR).
  - Weekly OVR uses the same scale as the season (99.0 needs a score of 70+).
    The Weekly tab lists the best QB of each week with the runner-up, the
    margin between them (in score points), and who was missing on each side.


-------------------------------------------------------------------------------
8. GAME PREDICTIONS
-------------------------------------------------------------------------------

For every game, the model predicts each team's chance to win, the final margin
and the total points, then shows the predicted score. Every input uses only
what was known BEFORE kickoff: stats from earlier weeks, the Vegas line, that
week's injury report and the weather forecast.

WHAT GOES IN (home team minus away team, where it applies)
  Vegas            Point spread, over/under, and the win chance implied by
                   the moneylines.
  Situation        Home field (none at neutral sites), rest days, division
                   game, primetime, dome, artificial turf.
  Travel           Miles the visitor travels, time-zone change, and West Coast
                   teams playing early (1 PM Eastern) kickoffs.
  Weather          Temperature, wind and rain at kickoff for outdoor games.
                   Upcoming games use the Open-Meteo forecast; past games use
                   the recorded weather and Open-Meteo's rain history.
  Coaches          Each head coach's career win % before the game.
  Referee          The crew's home-team win %, points per game and penalties
                   per game in earlier games. Neutral until the crew is
                   announced, usually game week.
  Head-to-head     Results of the last 5 meetings, and revenge games (lost
                   the last meeting).
  Team strength    Opponent-adjusted offense and defense ratings (EPA per
                   play overall, passing, rushing, and success rate), plus
                   point differential. Each week every team's rating is
                   solved together with its opponents', so stats against
                   strong defenses count for more. Recent games count more
                   (a game 6 weeks ago counts half), last season fades in
                   early on, and garbage time (win probability under 10% or
                   over 90%) is left out.
  Recent form      Point margin and EPA over the last 3 games.
  QB               The expected starter's QB rating going into the game.
                   Upcoming games use the depth chart's QB1 unless he's
                   listed Out or Doubtful. A QB with no starts this season
                   uses last season's rating; one with none at all is rated
                   like a typical backup.
  Supporting cast  Weapons, run game and pass protection grades (season to
                   date), and each team's pass defense going into the week.
  Injuries         Share of each team's key weapons and key defenders listed
                   Out or Doubtful on that week's injury report.

HOW IT WORKS
  - The model learns how much each input matters from every regular-season
    game since 2016, using ridge regression (a linear model that avoids
    overfitting). It predicts the margin and the total; the win % comes
    from the predicted margin, so the pick and the predicted score always
    agree.
  - Its settings (how strongly it's regularized, how much older seasons
    count, and how sharply a margin turns into a win %) were chosen by the
    backtest below, not by hand.
  - It's trained twice:
      Full model      uses everything, including Vegas. This gives the pick,
                      the win %, and the predicted score.
      Football model  uses only the football inputs, with no Vegas. "Stats vs
                      Vegas" is how many points the football model leans past
                      the spread. Big disagreements are the interesting games.
  - Key factors: the inputs that moved the football model's margin the most,
    grouped (team strength, QB, supporting cast, home field & travel, and so
    on), in points toward the team they favor.
  - Weeks already played also show the actual result and whether the pick
    was right.

HOW GOOD IS IT
  Backtest: each season from 2016 to 2025 was predicted by a model trained only
  on the OTHER seasons, so it never saw the games it was predicting
  (2,629 games). Every input for a game only uses information from
  before that game, including Next Gen Stats, which are rebuilt from the
  weekly numbers for earlier weeks only:
    Picks correct                66.9%   (Vegas favorite: 66.6%)
    Football stats only          63.9%   (no Vegas line at all)
    Win-probability error        0.2110   (Vegas moneyline: 0.2105; lower is better)
    Log loss                     0.6100   (Vegas moneyline: 0.6086; lower is better)
    Average margin miss          9.89 pts   (Vegas spread: 9.81)
    By season: 2016 63%, 2017 69%, 2018 68%, 2019 66%, 2020 67%, 2021 65%, 2022 67%, 2023 68%, 2024 71%, 2025 65%

  Calibration (when it says X%, does the pick win X% of the time?):
        50%-55%: predicted 53%, actually won 54%  (380 games)
        55%-60%: predicted 58%, actually won 57%  (406 games)
        60%-65%: predicted 63%, actually won 63%  (458 games)
        65%-70%: predicted 67%, actually won 64%  (415 games)
        70%-75%: predicted 72%, actually won 75%  (359 games)
        75%-80%: predicted 77%, actually won 80%  (263 games)
        80%-90%: predicted 84%, actually won 83%  (316 games)
       90%-100%: predicted 92%, actually won 91%  (32 games)

  Against the spread, taking the side the football stats favor:
    when they disagree with Vegas by 3+ points: 448-409 (52.3%)
    when they disagree by 5+ points:            166-135 (55.1%)
    (52.4% is the break-even rate against standard betting odds.)

  In plain terms: it picks winners about as well as Vegas (slightly better in
  the backtest), and its percentages are honest. Against the spread there's
  no reliable edge: 3+ point disagreements are break-even, and the 5+ point
  ones did a bit better but on only about 300 games, which could be luck.
  The predictions' real value is a clear, explained second opinion, not a
  betting system.

  Remaining shortcuts in the backtest (live predictions don't have these):
  past games use the QB who actually started (live: the depth chart) and the
  recorded weather (live: a forecast), and the model's settings were chosen
  using the whole backtest. Each makes the backtest look slightly better than
  live use. The pick log below is the fully honest measure.

PREDICTING FURTHER AHEAD
  You can predict any week of the season, but predictions get less reliable
  the further ahead they are:
  - Vegas only posts lines about a week ahead. Without a line, a game is
    predicted by the football-stats model alone and marked "No line yet".
    In the backtest that's about 3 points less accurate (64% vs 67%).
  - The injury report and the weather forecast don't exist yet either, and
    the team ratings are older.
  Football-stats model on the same games (weeks 9-18), by how far ahead:
      1 week   66.3%  (Brier 0.2116)        4 weeks  65.5%  (0.2146)
      2 weeks  65.4%  (0.2134)              6 weeks  65.3%  (0.2160)
      3 weeks  65.9%  (0.2137)              8 weeks  65.1%  (0.2175)
  (QB ratings and supporting-cast grades weren't re-dated in this test, so the
  real drop is probably a little bigger.) For the best predictions, run each
  week once the line, the injury report and the forecast are out, usually
  Wednesday to Saturday.

TRACKING REAL PICKS
  Every time you predict a week before its games kick off, the predictions
  are saved (the latest one before kickoff counts). After the games, they're
  graded automatically. The Predictions tab shows this season's record, or
  run:  python predict.py --record
  The pick log is %LOCALAPPDATA%\QB Ranker\data\pick_log.csv in the app
  (data\pick_log.csv when running from code).

WHAT WAS TESTED AND LEFT OUT
  Turnovers: each team's giveaway and takeaway rates, fumble-recovery luck,
  how costly its turnovers have been (win probability lost, which reflects
  where on the field and when in the game they happened), how its defense
  does on the short field after a giveaway, and how its offense cashes in
  after a takeaway. Each one relates to winning on its own, but adding them
  made the backtest slightly worse (Brier 0.2107 -> 0.2114), because EPA
  already counts every turnover and the rest is mostly luck that doesn't
  carry into the next game. The code is still in predict.py
  (USE_TURNOVERS = False) for future use, such as live in-game predictions.

  Interception matchup: each QB's interception rate (per attempt, before the
  game) against the opposing defense's interception rate, giving expected
  interceptions for each QB in that game. It does predict interceptions
  (QBs expected under 0.5 averaged 0.67; those expected 0.9+ averaged 0.85),
  and the defense's rate adds to the QB's own. But it didn't improve the game
  predictions (picks 66.91% -> 66.95%, Brier unchanged), so it's shown on
  every prediction as context ("Expected INTs") but isn't part of the model's
  math (USE_INT_MATCHUP = False).

  Retrain after each season, or after changing the QB weights, with:
      python predict.py --train
  That rebuilds features for every season since 2016 (it takes about 15
  minutes the first time), saves model.json, and prints the backtest. Run
  build_exe.bat afterwards to put the new model in the .exe.


-------------------------------------------------------------------------------
9. WHERE THE DATA COMES FROM
-------------------------------------------------------------------------------

All NFL data comes from nflverse (https://github.com/nflverse), a free, public
collection of NFL data. Weather comes from Open-Meteo (https://open-meteo.com),
a free weather service that needs no account. Nothing is scraped from other
sites. Data is
downloaded while the app runs, saved for 12 hours, and fetched again after
that (or right away with Re-download). An internet connection is needed.

  DATA                       USED FOR                                YEARS
  -------------------------  --------------------------------------  ---------
  Weekly player stats        QB, receiver, rusher and defender box   all
   (stats_player_week)        scores; EPA; CPOE; defensive stats
  Season player stats        Receivers' track record (last 2 years)  all
   (stats_player_reg)
  Schedule / results         Wins, losses, opponents, weeks          all
   (nfldata games.csv)
  Next Gen Stats receiving   Separation, cushion, YAC over expected  2016+
  Next Gen Stats rushing     RB rushing yards over expected          2018+
  Next Gen Stats passing     QB time to throw                        2016+
  PFR advanced passing       Pressures, hits, hurries, sacks,        2018+
   (Pro Football Reference)   blitzes per QB per game
  FTN charting               Interception-worthy throws              2022+
  Play-by-play               Who threw each charted pass             all
  Snap counts                Who played each game (missing players)  2012+
  Draft picks                Draft position, Pro Bowls, All-Pros     all
  Injury reports             Out / Doubtful players (predictions)    2009+
  Depth charts               Expected starting QB (predictions)      current
  Open-Meteo (not nflverse)  Weather forecasts and rain history      all

  Download addresses:
    https://github.com/nflverse/nflverse-data/releases/download/<file>
      stats_player/stats_player_week_<season>.csv
      stats_player/stats_player_reg_<season>.csv
      nextgen_stats/ngs_receiving.csv.gz
      nextgen_stats/ngs_rushing.csv.gz
      nextgen_stats/ngs_passing.csv.gz
      pfr_advstats/advstats_week_pass_<season>.csv
      ftn_charting/ftn_charting_<season>.csv
      pbp/play_by_play_<season>.csv.gz
      snap_counts/snap_counts_<season>.csv
      draft_picks/draft_picks.csv
      injuries/injuries_<season>.csv
      depth_charts/depth_charts_<season>.csv
    https://api.open-meteo.com/v1/forecast          (weather forecasts)
    https://archive-api.open-meteo.com/v1/archive   (past rain)
    https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv

  Timing: box scores usually update the night of or the day after games.
  Charting and snap counts can lag a day or two behind. Seasons before a
  data source starts (for example, pressures before 2018) treat that stat as
  average for everyone.


-------------------------------------------------------------------------------
10. CHANGING THE WEIGHTS
-------------------------------------------------------------------------------

Edit weights.json (the app's copy is in %LOCALAPPDATA%\QB Ranker), then reopen
the app or click Update. Doubling a weight doubles its share. Set a weight to 0
to turn something off.

  "metrics"          the 16 QB stats (weight, and whether higher is better)
  "context"          weapons, run_game, pass_pro, opp_defense
  "weapon_parts"     how the weapons grade is built
  "pass_pro_parts"   how the pass-protection grade is built
  "defense"          how defenses are rated, and how many games last season
                     counts for early on
  "without_qb"       how much games the QB missed count ("max_share",
                     "prior_games"), and the parts used
  "missing_players"  key-weapon cutoff (% of targets), backup grade, number of
                     key defenders, and how much losing them matters

Other settings are in qb_rank.py:
  COMPARISON_MIN_STARTS = 4    the group everyone is graded against
  TRACK_SEASON_WEIGHTS         {this season: 3, last: 2, two ago: 1}
  CACHE_HOURS = 12             how long downloaded data is reused


-------------------------------------------------------------------------------
11. FILES IN THIS FOLDER
-------------------------------------------------------------------------------

  dist\QB Ranker.exe   The standalone app. Share this file.
  app.py               The desktop app (Rankings / Supporting Cast / Weekly).
  qb_rank.py           The ranking engine and command-line tool.
  predict.py           Game predictions (features, model, backtest).
  stadiums.py          Stadium locations and time zones (travel, weather).
  model.json           The trained prediction model (bundled into the .exe).
  weights.json         Default weights (bundled into the .exe).
  build_exe.bat        Rebuilds dist\QB Ranker.exe.
  requirements.txt     Python packages for running from code.
  data\                Downloaded stats, when run from code.
  views\, ui_common.py An earlier web version of the app (Streamlit pages).
  run_ui.bat           Launcher for that web version. It currently doesn't
                       start, because app.py is now the desktop app; the web
                       pages need their own launch file to run again.


-------------------------------------------------------------------------------
12. LIMITATIONS
-------------------------------------------------------------------------------

  - Stats measure results, not film. Things like reads, pre-snap work and
    throws the receiver should have caught are only partly captured.
  - Pressure is shared between a line and its QB. The adjustments remove what
    time to throw and blitzes explain, but not everything (a QB scrambling into
    pressure, for example). Lines aren't adjusted for the pass rushes they faced.
  - "Time given" is estimated from pressure rates, not measured play by play.
  - A receiver's track record still includes some of his past QBs' influence.
  - A defense's rating isn't adjusted for the QBs it has faced.
  - Players who left a game injured after a few snaps count as having played.
  - Next Gen Stats only tracks receivers above a target threshold, so depth
    players have no separation or YAC numbers.
  - The newest week's charting and snap counts can lag, so missing players and
    interception-worthy throws may not show up for a day or two.
  - The app depends on nflverse keeping its file names and columns. If they
    change, the code needs an update and the .exe a rebuild.


-------------------------------------------------------------------------------
13. CREDITS
-------------------------------------------------------------------------------

  Weather: Open-Meteo (https://open-meteo.com), weather data by Open-Meteo.com.
  Data: nflverse (https://github.com/nflverse), which collects NFL stats and
  republishes Next Gen Stats, Pro Football Reference advanced stats, FTN Data
  charting and snap counts. Thanks to the nflverse maintainers and to the
  original providers. See nflverse's pages for each source's license and
  attribution terms.

===============================================================================
