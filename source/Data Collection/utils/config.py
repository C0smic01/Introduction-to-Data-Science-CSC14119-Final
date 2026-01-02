HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


# ============================================================================
# DATA SCHEMA
# ============================================================================
BASE_SCHEMA = {
    # ========================
    # BASIC INFO
    # ========================
    "player_id": None,
    "player_name": None,
    "age": None,
    "nationality": None,
    "height": None,
    "foot": None,
    "position": None,
    "current_club": None,
    "league": None,
    # ========================
    # PLAYING TIME
    # ========================
    "appearances": 0,
    "minutes_played": 0,
    "minutes_per_game": None,
    # ========================
    # OUTFIELD - ATTACKING (13 fields)
    # ========================
    "goals": 0,
    "assists": 0,
    "goals_per_90": None,
    "assists_per_90": None,
    "npg_per90": None,
    "npxg_per90": None,
    "xag_per90": None,
    "npxg_xag_per90": None,
    "xg_per90": None,
    "shots_per90": None,
    "shots_on_target_per90": None,
    "shots_on_target_pct": None,
    "avg_shot_distance": None,
    "sca_per90": None,
    "gca_per90": None,
    # ========================
    # OUTFIELD - PLAYMAKING (8 fields)
    # ========================
    "key_passes_per90": None,
    "passes_completed_per90": None,
    "pass_completion_pct": None,
    "passes_into_final_third_per90": None,
    "passes_into_penalty_area_per90": None,
    "progressive_passes_per90": None,
    "progressive_passes_rec_per90": None,
    "progressive_carries_per90": None,
    # ========================
    # OUTFIELD - BALL PROGRESSION (8 fields)
    # ========================
    "take_ons_per90": None,
    "take_on_success_pct": None,
    "carries_per90": None,
    "carries_into_final_third_per90": None,
    "touches_per90": None,
    "touches_att_third_per90": None,
    "touches_att_pen_per90": None,
    "passes_received_per90": None,
    # ========================
    # OUTFIELD - DEFENSIVE (4 fields)
    # ========================
    "tackles_per90": None,
    "interceptions_per90": None,
    "blocks_per90": None,
    "ball_recoveries_per90": None,
    # ========================
    # OUTFIELD - PHYSICAL (2 fields)
    # ========================
    "aerials_won_per90": None,
    "aerial_win_pct": None,
    # ========================
    # OUTFIELD - DISCIPLINE (3 fields)
    # ========================
    "yellow_cards_per90": None,
    "red_cards_per90": None,
    "fouls_committed_per90": None,
    # ========================
    # GOALKEEPER - SHOT STOPPING (8 fields)
    # ========================
    "goals_against_per90": None,
    "shots_on_target_against_per90": None,
    "saves_per90": None,
    "save_percentage": None,
    "clean_sheet_pct": None,
    "psxg_per_shot": None,
    "psxg_ga_per90": None,
    "penalty_save_pct": None,
    # ========================
    # GOALKEEPER - DISTRIBUTION (3 fields)
    # ========================
    "passes_attempted_per90": None,
    "launch_pct": None,
    "avg_pass_length": None,
    # ========================
    # GOALKEEPER - SWEEPING (2 fields)
    # ========================
    "def_actions_outside_pen_per90": None,
    "avg_distance_def_actions": None,
    # ========================
    # GOALKEEPER - CROSS HANDLING (1 field)
    # ========================
    "crosses_stopped_pct": None,
    # ========================
    # GOALKEEPER - MATCH CONTEXT (3 fields)
    # ========================
    "wins_per90": None,
    "draws_per90": None,
    "losses_per90": None,
    # ========================
    # TARGET
    # ========================
    "market_value": None,
}

# ============================================================================
# STAT MAPPINGS
# ============================================================================
OUTFIELD_SCOUTING_MAPPING = {
    # Core attacking
    "Goals": "goals_per_90",
    "Assists": "assists_per_90",
    "Non-Penalty Goals": "npg_per90",
    "xG: Expected Goals": "xg_per90",
    "npxG: Non-Penalty xG": "npxg_per90",
    "xAG: Exp. Assisted Goals": "xag_per90",
    "npxG + xAG": "npxg_xag_per90",
}

OUTFIELD_DETAILED_MAPPING = {
    # Shooting
    "Shots Total": "shots_per90",
    "Shots on Target": "shots_on_target_per90",
    "Shots on Target %": "shots_on_target_pct",
    "Average Shot Distance": "avg_shot_distance",
    # Playmaking
    "Shot-Creating Actions": "sca_per90",
    "Goal-Creating Actions": "gca_per90",
    "Key Passes": "key_passes_per90",
    "Passes into Final Third": "passes_into_final_third_per90",
    "Passes into Penalty Area": "passes_into_penalty_area_per90",
    # Passing
    "Passes Completed": "passes_completed_per90",
    "Pass Completion %": "pass_completion_pct",
    "Progressive Passes": "progressive_passes_per90",
    # Progression
    "Progressive Carries": "progressive_carries_per90",
    "Progressive Passes Rec": "progressive_passes_rec_per90",
    "Successful Take-Ons": "take_ons_per90",
    "Successful Take-On %": "take_on_success_pct",
    "Carries": "carries_per90",
    "Carries into Final Third": "carries_into_final_third_per90",
    # Possession
    "Touches": "touches_per90",
    "Touches (Att 3rd)": "touches_att_third_per90",
    "Touches (Att Pen)": "touches_att_pen_per90",
    "Passes Received": "passes_received_per90",
    # Defense
    "Tackles": "tackles_per90",
    "Interceptions": "interceptions_per90",
    "Blocks": "blocks_per90",
    "Ball Recoveries": "ball_recoveries_per90",
    "Clearances": "clearances_per90",
    # Physical
    "Aerials Won": "aerials_won_per90",
    "% of Aerials Won": "aerial_win_pct",
    # Discipline
    "Yellow Cards": "yellow_cards_per90",
    "Red Cards": "red_cards_per90",
    "Fouls Committed": "fouls_committed_per90",
}

GOALKEEPER_SCOUTING_MAPPING = {
    "Goals Against": "goals_against_per90",
    "Shots on Target Against": "shots_on_target_against_per90",
    "Saves": "saves_per90",
    "Save Percentage": "save_percentage",
    "Clean Sheet Percentage": "clean_sheet_pct",
}

GOALKEEPER_DETAILED_MAPPING = {
    "PSxG/SoT": "psxg_per_shot",
    "PSxG-GA": "psxg_ga_per90",
    "Save% (Penalty Kicks)": "penalty_save_pct",
    "Passes Attempted (GK)": "passes_attempted_per90",
    "Launch %": "launch_pct",
    "Average Pass Length": "avg_pass_length",
    "Def. Actions Outside Pen. Area": "def_actions_outside_pen_per90",
    "Avg. Distance of Def. Actions": "avg_distance_def_actions",
    "Crosses Stopped %": "crosses_stopped_pct",
    "Wins": "wins_per90",
    "Draws": "draws_per90",
    "Losses": "losses_per90",
}

# ============================================================================
# FBREF DEFAULT LEAGUES
# ============================================================================
SEASON = "{SEASON}"

LEAGUE_CONFIG = {
    "La Liga": f"https://fbref.com/en/comps/12/{SEASON}/{SEASON}-La-Liga-Stats",
    "Premier League": f"https://fbref.com/en/comps/9/{SEASON}/{SEASON}-Premier-League-Stats",
    "Serie A": f"https://fbref.com/en/comps/11/{SEASON}/{SEASON}-Serie-A-Stats",
    "Bundesliga": f"https://fbref.com/en/comps/20/{SEASON}/{SEASON}-Bundesliga-Stats",
    "Ligue 1": f"https://fbref.com/en/comps/13/{SEASON}/{SEASON}-Ligue-1-Stats",
    "Eredivisie": f"https://fbref.com/en/comps/23/{SEASON}/{SEASON}-Eredivisie-Stats",
    "Primeira Liga": f"https://fbref.com/en/comps/32/{SEASON}/{SEASON}-Primeira-Liga-Stats",
    "Saudi Pro League": f"https://fbref.com/en/comps/70/{SEASON}/{SEASON}-Saudi-Pro-League-Stats",
    "Austrian Bundesliga": f"https://fbref.com/en/comps/56/{SEASON}/{SEASON}-Austrian-Bundesliga-Stats",
    "J1 League": "https://fbref.com/en/comps/25/J1-League-Stats",
    "MLS": "https://fbref.com/en/comps/22/Major-League-Soccer-Stats",
    "Belgian Pro League": f"https://fbref.com/en/comps/37/{SEASON}/{SEASON}-Belgian-Pro-League-Stats",
    "Süper Lig": f"https://fbref.com/en/comps/26/{SEASON}/{SEASON}-Super-Lig-Stats",
    "Scottish Premiership": f"https://fbref.com/en/comps/40/{SEASON}/{SEASON}-Scottish-Premiership-Stats",
    "Argentine Liga": f"https://fbref.com/en/comps/21/Liga-Profesional-Argentina-Stats",
    "Liga MX": f"https://fbref.com/en/comps/31/{SEASON}/{SEASON}-Liga-MX-Stats",
    "Eliteserien": f"https://fbref.com/en/comps/28/2024/2024-Eliteserien-Stats",
    "Serbian SuperLiga": f"https://fbref.com/en/comps/54/{SEASON}/{SEASON}-Serbian-SuperLiga-Stats",
    "Russian Premier League": f"https://fbref.com/en/comps/30/{SEASON}/{SEASON}-Russian-Premier-League-Stats",
    "Hrvatska NL": f"https://fbref.com/en/comps/63/{SEASON}/{SEASON}-Hrvatska-NL-Stats",
    "Czech First League": f"https://fbref.com/en/comps/66/{SEASON}/{SEASON}-Czech-First-League-Stats",
    "Chinese Super League": f"https://fbref.com/en/comps/62/Chinese-Super-League-Stats",
    "Allsvenskan": f"https://fbref.com/en/comps/29/{SEASON}/{SEASON}-Allsvenskan-Stats",
    "Ekstraklasa": f"https://fbref.com/en/comps/36/{SEASON}/{SEASON}-Ekstraklasa-Stats",
    "Swiss Super League": f"https://fbref.com/en/comps/57/{SEASON}/{SEASON}-Swiss-Super-League-Stats",
    "Liga 1": "https://fbref.com/en/comps/44/2024/2024-Liga-1-Stats",
    "Uruguayan Primera División": "https://fbref.com/en/comps/45/2024/2024-Uruguayan-Primera-Division-Stats",
    "Superettan": "https://fbref.com/en/comps/48/2024/2024-Superettan-Stats",
    "A-League Men": f"https://fbref.com/en/comps/65/{SEASON}/{SEASON}-A-League-Men-Stats",
    "Veikkausliiga": "https://fbref.com/en/comps/43/2024/2024-Veikkausliiga-Stats",
    "J2 League": "https://fbref.com/en/comps/49/2024/2024-J2-League-Stats",
    "South African Premiership": f"https://fbref.com/en/comps/52/{SEASON}/{SEASON}-South-African-Premiership-Stats",
    "Challenger Pro League": f"https://fbref.com/en/comps/69/{SEASON}/{SEASON}-Challenger-Pro-League-Stats",
    "Venezuelan Primera División": f"https://fbref.com/en/comps/105/2024/2024-Venezuelan-Primera-Division-Stats",
    "NB I": f"https://fbref.com/en/comps/46/{SEASON}/{SEASON}-NB-I-Stats",
    "Croatian Football League": f"https://fbref.com/en/comps/63/{SEASON}/{SEASON}-Hrvatska-NL-Stats",
    "Paraguayan Primera División": f"https://fbref.com/en/comps/61/2024/2024-Primera-Division-Stats",
    "Segunda División": f"https://fbref.com/en/comps/17/{SEASON}/{SEASON}-Segunda-Division-Stats",
    "Bolivian Primera División": f"https://fbref.com/en/comps/74/2024/2024-Bolivian-Primera-Division-Stats",
    "EFL League One": f"https://fbref.com/en/comps/15/{SEASON}/{SEASON}-League-One-Stats",
    "Ligue 2": f"https://fbref.com/en/comps/60/{SEASON}/{SEASON}-Ligue-2-Stats",
    "Liga Profesional Ecuador": f"https://fbref.com/en/comps/58/2024/2024-Serie-A-Stats",
    "National League": f"https://fbref.com/en/comps/34/{SEASON}/{SEASON}-National-League-Stats",
    "Danish Superliga": f"https://fbref.com/en/comps/50/{SEASON}/{SEASON}-Danish-Superliga-Stats",
    "League of Ireland Premier Division": f"https://fbref.com/en/comps/80/2024/2024-League-of-Ireland-Premier-Division-Stats",
    "Eerste Divisie": f"https://fbref.com/en/comps/51/{SEASON}/{SEASON}-Eerste-Divisie-Stats",
    "Liga I": f"https://fbref.com/en/comps/47/{SEASON}/{SEASON}-Liga-I-Stats",
    "Persian Gulf Pro League": f"https://fbref.com/en/comps/64/{SEASON}/stats/{SEASON}-Persian-Gulf-Pro-League-Stats",
    "Primera A": f"https://fbref.com/en/comps/41/2024/2024-Primera-A-Stats",
    "Canadian Premier League": f"https://fbref.com/en/comps/211/2024/2024-Canadian-Premier-League-Stats",
    "K League 1": f"https://fbref.com/en/comps/55/2024/2024-K-League-1-Stats",
    "Super League Greece": f"https://fbref.com/en/comps/27/{SEASON}/{SEASON}-Super-League-Greece-Stats",
    "Série B": f"https://fbref.com/en/comps/38/2024/2024-Serie-B-Stats",
    "Chilean Primera División": f"https://fbref.com/en/comps/35/2024/2024-Chilean-Primera-Division-Stats",
    "Ukrainian Premier League": f"https://fbref.com/en/comps/39/{SEASON}/{SEASON}-Ukrainian-Premier-League-Stats",
    "Indian Super League": f"https://fbref.com/en/comps/82/Indian-Super-League-Stats",
    "EFL League Two": f"https://fbref.com/en/comps/16/{SEASON}/{SEASON}-League-Two-Stats",
}

# URL
TRANSFERMARKT_PLAYER_SEACH_URL = "https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche?query={player_name}"

# Thread
MAX_THREADS = 5

# Delay
MIN_DELAY = 1.5
MAX_DELAY = 3.0

DATA_FOLDER = "data"
CSV_FOLDER = f"{DATA_FOLDER}/csv"
JSON_FOLDER = f"{DATA_FOLDER}/json"
HEADLESS = True
