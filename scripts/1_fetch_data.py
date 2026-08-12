"""
Script 1: Fetch data
- Loads historical season CSVs (static, stored in data/raw_historical/)
- Downloads the CURRENT season's results fresh from football-data.co.uk
- Pulls xG for all seasons from Understat
- Saves the combined, cleaned dataset to data/matches_with_xg.csv

This is the only script that talks to the internet. Splitting it out
means steps 2 and 3 can be re-run/debugged without re-fetching data.
"""
import pandas as pd
import requests
import io
import os
import time
import soccerdata as sd

RAW_HISTORICAL_DIR = 'data/raw_historical/'
OUTPUT_PATH = 'data/matches_with_xg.csv'

# --- Step 1: Load static historical season files ---
# These don't change once a season ends, so they're stored as plain
# files in the repo rather than re-downloaded every week.
season_files = sorted([f for f in os.listdir(RAW_HISTORICAL_DIR) if f.endswith('.csv')])
print(f"Found {len(season_files)} historical season files")

all_seasons = []
for file in season_files:
    df = pd.read_csv(os.path.join(RAW_HISTORICAL_DIR, file))
    keep_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
    all_seasons.append(df[[c for c in keep_cols if c in df.columns]])

# --- Step 2: Download the CURRENT season fresh (this DOES change weekly) ---
# football-data.co.uk uses a predictable URL pattern:
# mmz4281/{season_code}/E0.csv — e.g. 2526 for the 2025-26 season.
CURRENT_SEASON_CODE = '2627'  # 2026-27 season — update this each August when a new season starts
current_season_url = f'https://www.football-data.co.uk/mmz4281/{CURRENT_SEASON_CODE}/E0.csv'

print(f"Downloading current season from {current_season_url}")
response = requests.get(current_season_url)
if response.status_code == 200 and len(response.content) > 100:
    current_df = pd.read_csv(io.StringIO(response.text))
    keep_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
    current_df = current_df[[c for c in keep_cols if c in current_df.columns]]
    all_seasons.append(current_df)
    print(f"Current season: {len(current_df)} matches so far")
else:
    print("WARNING: current season file not available yet (season hasn't started, or URL changed) — proceeding with historical data only")

matches = pd.concat(all_seasons, ignore_index=True)

# --- Step 3: Clean (same logic as the Colab version) ---
matches['Date'] = pd.to_datetime(matches['Date'], dayfirst=True, errors='coerce')
matches = matches.dropna(subset=['Date', 'HomeTeam', 'AwayTeam'])  # drop any blank trailing rows

team_name_fixes = {
    'Man United': 'Manchester United', 'Man City': 'Manchester City',
    'Newcastle': 'Newcastle United', "Nott'm Forest": 'Nottingham Forest',
    'Spurs': 'Tottenham',
}
matches['HomeTeam'] = matches['HomeTeam'].replace(team_name_fixes)
matches['AwayTeam'] = matches['AwayTeam'].replace(team_name_fixes)
matches = matches.rename(columns={'FTHG': 'home_goals', 'FTAG': 'away_goals', 'FTR': 'result'})
matches = matches.dropna(subset=['home_goals', 'away_goals', 'result'])  # drop unplayed fixtures
matches = matches.sort_values('Date').reset_index(drop=True)

def season_start_year(date):
    return date.year if date.month >= 8 else date.year - 1

matches['season_start_year'] = matches['Date'].apply(season_start_year)

def to_understat_code(start_year):
    return str(start_year)[2:] + str(start_year + 1)[2:]

start_years = sorted(matches['season_start_year'].unique())
understat_seasons = [to_understat_code(y) for y in start_years]
print(f"Seasons in dataset: {understat_seasons}")

# --- Step 4: Pull xG from Understat, one season at a time ---
xg_pieces = []
for code in understat_seasons:
    try:
        print(f"Fetching Understat xG for {code}...")
        understat = sd.Understat(leagues='ENG-Premier League', seasons=[code])
        df = understat.read_team_match_stats().reset_index()
        xg_pieces.append(df)
    except Exception as e:
        print(f"  Could not fetch {code}: {e}")
    time.sleep(1)

xg_data = pd.concat(xg_pieces, ignore_index=True)
xg_matches = xg_data[['date', 'home_team', 'away_team', 'home_xg', 'away_xg']].rename(
    columns={'date': 'Date', 'home_team': 'HomeTeam', 'away_team': 'AwayTeam'})
xg_matches['Date'] = pd.to_datetime(xg_matches['Date']).dt.tz_localize(None).dt.normalize()

understat_name_fixes = {
    'Wolverhampton Wanderers': 'Wolves',
    'West Bromwich Albion': 'West Brom',
    'Newcastle': 'Newcastle United',
}
xg_matches['HomeTeam'] = xg_matches['HomeTeam'].replace(understat_name_fixes)
xg_matches['AwayTeam'] = xg_matches['AwayTeam'].replace(understat_name_fixes)

matches = matches.merge(xg_matches, on=['Date', 'HomeTeam', 'AwayTeam'], how='left')
missing_xg = matches['home_xg'].isna().sum()
print(f"Matches missing xG: {missing_xg} / {len(matches)}")

# --- Step 5: Save ---
os.makedirs('data', exist_ok=True)
matches.to_csv(OUTPUT_PATH, index=False)
print(f"Saved {len(matches)} matches to {OUTPUT_PATH}")
