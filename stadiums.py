"""Stadium locations and time zones, for travel distance, time-zone changes and weather.

TEAM_HOME: each team's home stadium (latitude, longitude, hours from US Eastern time during the season,
whether it has a roof). Teams that moved have their old homes keyed by their old abbreviation
(SD = San Diego Chargers 2016, OAK = Oakland Raiders through 2019).

NEUTRAL: international / neutral-site stadiums, keyed by the schedule's stadium_id.
"""

from __future__ import annotations

import math

# team: (lat, lon, hours vs Eastern, indoor)
TEAM_HOME = {
    "ARI": (33.5276, -112.2626, -3, True),    # State Farm Stadium (retractable roof; Arizona doesn't use DST)
    "ATL": (33.7554, -84.4008, 0, True),      # Mercedes-Benz Stadium
    "BAL": (39.2780, -76.6227, 0, False),     # M&T Bank Stadium
    "BUF": (42.7738, -78.7870, 0, False),     # Highmark Stadium
    "CAR": (35.2258, -80.8528, 0, False),     # Bank of America Stadium
    "CHI": (41.8623, -87.6167, -1, False),    # Soldier Field
    "CIN": (39.0955, -84.5161, 0, False),     # Paycor Stadium
    "CLE": (41.5061, -81.6995, 0, False),     # Cleveland Browns Stadium
    "DAL": (32.7473, -97.0945, -1, True),     # AT&T Stadium
    "DEN": (39.7439, -105.0201, -2, False),   # Empower Field at Mile High
    "DET": (42.3400, -83.0456, 0, True),      # Ford Field
    "GB": (44.5013, -88.0622, -1, False),     # Lambeau Field
    "HOU": (29.6847, -95.4107, -1, True),     # NRG Stadium
    "IND": (39.7601, -86.1639, 0, True),      # Lucas Oil Stadium
    "JAX": (30.3239, -81.6373, 0, False),     # EverBank Stadium
    "KC": (39.0489, -94.4839, -1, False),     # GEHA Field at Arrowhead
    "LV": (36.0909, -115.1833, -3, True),     # Allegiant Stadium (2020+)
    "OAK": (37.7516, -122.2005, -3, False),   # Oakland Coliseum (through 2019)
    "LAC": (33.9535, -118.3392, -3, True),    # SoFi Stadium (2020+; Carson 2017-19 is nearby)
    "SD": (32.7831, -117.1196, -3, False),    # Qualcomm Stadium (2016)
    "LA": (33.9535, -118.3392, -3, True),     # SoFi Stadium (2020+; LA Coliseum 2016-19 is nearby)
    "MIA": (25.9580, -80.2389, 0, False),     # Hard Rock Stadium
    "MIN": (44.9737, -93.2581, -1, True),     # U.S. Bank Stadium
    "NE": (42.0909, -71.2643, 0, False),      # Gillette Stadium
    "NO": (29.9511, -90.0812, -1, True),      # Caesars Superdome
    "NYG": (40.8135, -74.0745, 0, False),     # MetLife Stadium
    "NYJ": (40.8135, -74.0745, 0, False),     # MetLife Stadium
    "PHI": (39.9008, -75.1675, 0, False),     # Lincoln Financial Field
    "PIT": (40.4468, -80.0158, 0, False),     # Acrisure Stadium
    "SF": (37.4030, -121.9700, -3, False),    # Levi's Stadium
    "SEA": (47.5952, -122.3316, -3, False),   # Lumen Field
    "TB": (27.9759, -82.5033, 0, False),      # Raymond James Stadium
    "TEN": (36.1665, -86.7713, -1, False),    # Nissan Stadium
    "WAS": (38.9078, -76.8645, 0, False),     # Northwest Stadium
}

# schedule stadium_id: (lat, lon, hours vs Eastern during the season, indoor, name)
NEUTRAL = {
    "LON00": (51.5560, -0.2796, 5, False, "Wembley Stadium, London"),
    "LON01": (51.4560, -0.3415, 5, False, "Twickenham Stadium, London"),
    "LON02": (51.6043, -0.0664, 5, False, "Tottenham Hotspur Stadium, London"),
    "FRA00": (50.0686, 8.6455, 6, True, "Deutsche Bank Park, Frankfurt"),
    "GER00": (48.2188, 11.6247, 6, False, "Allianz Arena, Munich"),
    "MUN01": (48.2188, 11.6247, 6, False, "Allianz Arena, Munich"),
    "MAD01": (40.4531, -3.6883, 6, True, "Santiago Bernabeu, Madrid"),
    "PAR00": (48.9245, 2.3601, 6, False, "Stade de France, Paris"),
    "MEX00": (19.3029, -99.1505, -1, False, "Estadio Azteca, Mexico City"),
    "SAO00": (-23.5453, -46.4742, 1, False, "Arena Corinthians, Sao Paulo"),
    "RIO00": (-22.9122, -43.2302, 1, False, "Maracana, Rio de Janeiro"),
    "MEL00": (-37.8200, 144.9834, 15, False, "Melbourne Cricket Ground"),
}
# US stadiums that sometimes host neutral-site games -> the team that plays there
US_STADIUM_TEAM = {
    "ATL97": "ATL", "CLE00": "CLE", "DET00": "DET", "HOU00": "HOU", "IND00": "IND", "JAX00": "JAX",
    "LAX01": "LA", "MIA00": "MIA", "MIN01": "MIN", "NOR00": "NO", "NYC01": "NYG", "PHO00": "ARI",
    "PIT00": "PIT", "SFO01": "SF", "TAM00": "TB", "VEG00": "LV",
}


def site(home_team: str, stadium_id: str | None = None, neutral: bool = False, name: str | None = None):
    """(lat, lon, tz, indoor) where the game is played."""
    if neutral and name:  # the schedule sometimes pairs an international stadium name with a US stadium id
        first = str(name).split()[0].lower()
        for v in NEUTRAL.values():
            if first in v[4].lower():
                return v[:4]
    if neutral and stadium_id in NEUTRAL:
        return NEUTRAL[stadium_id][:4]
    if neutral and stadium_id in US_STADIUM_TEAM:
        return TEAM_HOME[US_STADIUM_TEAM[stadium_id]]
    return TEAM_HOME.get(home_team, (39.0, -95.0, -1, False))


def home_of(team: str):
    return TEAM_HOME.get(team, (39.0, -95.0, -1, False))


def miles(a, b) -> float:
    """Great-circle distance in miles between two (lat, lon, ...) points."""
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 3958.8 * 2 * math.asin(math.sqrt(h))
