"""
Script 2: Build features
- Reads data/matches_with_xg.csv (from Script 1)
- Builds rolling form/xG features (leak-safe)
- Builds sequential Elo ratings
- Applies the promoted-team baseline + manual transfer adjustments
- Saves data/matches_featured_v2.csv (training data)
  and data/starting_elo_2026_27.csv (current team strength for simulation)
"""
import pandas as pd
import os

INPUT_PATH = 'data/matches_with_xg.csv'

matches = pd.read_csv(INPUT_PATH, parse_dates=['Date'])
print(f"Loaded {len(matches)} matches")

def season_start_year(date):
    return date.year if date.month >= 8 else date.year - 1

# ============================================================
# Rolling form features
# ============================================================
home_rows = matches[['Date', 'HomeTeam', 'AwayTeam', 'home_goals', 'away_goals', 'home_xg', 'away_xg', 'result']].copy()
home_rows = home_rows.rename(columns={'HomeTeam': 'team', 'AwayTeam': 'opponent',
    'home_goals': 'goals_for', 'away_goals': 'goals_against', 'home_xg': 'xg_for', 'away_xg': 'xg_against'})
home_rows['is_home'] = 1
home_rows['points'] = home_rows['result'].map({'H': 3, 'D': 1, 'A': 0})

away_rows = matches[['Date', 'HomeTeam', 'AwayTeam', 'home_goals', 'away_goals', 'home_xg', 'away_xg', 'result']].copy()
away_rows = away_rows.rename(columns={'AwayTeam': 'team', 'HomeTeam': 'opponent',
    'away_goals': 'goals_for', 'home_goals': 'goals_against', 'away_xg': 'xg_for', 'home_xg': 'xg_against'})
away_rows['is_home'] = 0
away_rows['points'] = away_rows['result'].map({'A': 3, 'D': 1, 'H': 0})

team_matches = pd.concat([home_rows, away_rows], ignore_index=True).sort_values(['team', 'Date']).reset_index(drop=True)

def add_rolling_features(df, window):
    grouped = df.groupby('team')
    suffix = f'_last{window}'
    for col, out in [('points', 'form_points'), ('goals_for', 'goals_for'),
                      ('goals_against', 'goals_against'), ('xg_for', 'xg_for'), ('xg_against', 'xg_against')]:
        # shift(1) BEFORE rolling — prevents leakage (a match's features
        # must only reflect what happened BEFORE it, never itself)
        df[f'{out}{suffix}'] = grouped[col].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    return df

team_matches = add_rolling_features(team_matches, window=5)
team_matches = add_rolling_features(team_matches, window=10)

rolling_cols = [c for c in team_matches.columns if 'last' in c]
merge_cols = ['Date', 'team', 'opponent'] + rolling_cols

home_features = team_matches[team_matches['is_home'] == 1][merge_cols].rename(
    columns={'team': 'HomeTeam', 'opponent': 'AwayTeam', **{c: f'home_{c}' for c in rolling_cols}})
away_features = team_matches[team_matches['is_home'] == 0][merge_cols].rename(
    columns={'team': 'AwayTeam', 'opponent': 'HomeTeam', **{c: f'away_{c}' for c in rolling_cols}})

matches_featured = matches.merge(home_features, on=['Date', 'HomeTeam', 'AwayTeam'], how='left')
matches_featured = matches_featured.merge(away_features, on=['Date', 'HomeTeam', 'AwayTeam'], how='left')

# ============================================================
# Elo ratings
# ============================================================
def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

HOME_ADVANTAGE = 60
STARTING_ELO = 1500
K_FACTOR = 20

matches_sorted = matches_featured.sort_values('Date').reset_index(drop=True)
current_elo = {}
home_elo_pre, away_elo_pre = [], []

for idx, row in matches_sorted.iterrows():
    home, away = row['HomeTeam'], row['AwayTeam']
    rating_home = current_elo.get(home, STARTING_ELO)
    rating_away = current_elo.get(away, STARTING_ELO)
    home_elo_pre.append(rating_home)
    away_elo_pre.append(rating_away)

    actual_home = {'H': 1.0, 'D': 0.5, 'A': 0.0}[row['result']]
    actual_away = 1.0 - actual_home
    exp_home = expected_score(rating_home + HOME_ADVANTAGE, rating_away)
    exp_away = 1 - exp_home

    current_elo[home] = rating_home + K_FACTOR * (actual_home - exp_home)
    current_elo[away] = rating_away + K_FACTOR * (actual_away - exp_away)

matches_sorted['home_elo_pre'] = home_elo_pre
matches_sorted['away_elo_pre'] = away_elo_pre
matches_sorted['elo_diff'] = matches_sorted['home_elo_pre'] - matches_sorted['away_elo_pre']

print("Current Elo (top 10):")
print(pd.Series(current_elo).sort_values(ascending=False).head(10))

# --- Promoted-team baseline (recency-weighted) ---
home_elo_rows = matches_sorted[['Date', 'HomeTeam', 'AwayTeam', 'home_elo_pre']].rename(
    columns={'HomeTeam': 'team', 'AwayTeam': 'opponent', 'home_elo_pre': 'elo_pre'})
away_elo_rows = matches_sorted[['Date', 'HomeTeam', 'AwayTeam', 'away_elo_pre']].rename(
    columns={'AwayTeam': 'team', 'HomeTeam': 'opponent', 'away_elo_pre': 'elo_pre'})
elo_long = pd.concat([home_elo_rows, away_elo_rows], ignore_index=True).sort_values(['team', 'Date']).reset_index(drop=True)
elo_long['game_number'] = elo_long.groupby('team').cumcount() + 1
elo_long['season_start_year'] = elo_long['Date'].apply(season_start_year)

first_season = matches_sorted['Date'].apply(season_start_year).min()
first_appearance_season = elo_long.groupby('team')['season_start_year'].min()
promoted_historical = first_appearance_season[first_appearance_season > first_season].index.tolist()

settled = elo_long[(elo_long['team'].isin(promoted_historical)) & (elo_long['game_number'] == 11)][
    ['team', 'elo_pre', 'season_start_year']].copy()
if len(settled) > 0:
    most_recent = settled['season_start_year'].max()
    settled['weight'] = 1 / (1 + (most_recent - settled['season_start_year']))
    baseline_promoted_elo = (settled['elo_pre'] * settled['weight']).sum() / settled['weight'].sum()
else:
    baseline_promoted_elo = STARTING_ELO - 100  # fallback if no historical promoted teams found

print(f"Promoted-team baseline Elo: {baseline_promoted_elo:.1f}")

# ============================================================
# 2026-27 team list, transfer adjustments
# ============================================================
teams_2026_27 = ['Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton', 'Chelsea',
    'Crystal Palace', 'Everton', 'Fulham', 'Leeds', 'Liverpool', 'Manchester City',
    'Manchester United', 'Newcastle United', 'Nottingham Forest', 'Sunderland',
    'Tottenham', 'Coventry City', 'Ipswich', 'Hull City']

for team in ['Coventry City', 'Hull City']:
    if team not in current_elo:
        current_elo[team] = baseline_promoted_elo

missing = [t for t in teams_2026_27 if t not in current_elo]
if missing:
    print(f"WARNING — team name mismatch, not found in Elo: {missing}")

elo_2026_27 = pd.Series({t: current_elo[t] for t in teams_2026_27 if t in current_elo})

# Update this dictionary as the transfer window develops — this is the
# manual, editable layer we discussed. Re-running this script picks up
# whatever values are here at the time.
manual_elo_adjustments = {
    'Tottenham': 38, 'Chelsea': 35, 'Manchester City': 25, 'Manchester United': 20,
    'Nottingham Forest': -15, 'Newcastle United': -15, 'Bournemouth': -10,
    'Arsenal': 15, 'Everton': 10, 'Leeds': 10, 'Brighton': 5, 'Brentford': 5,
    'Ipswich': 5, 'Aston Villa': -5, 'Crystal Palace': -5, 'Sunderland': -10,
    'Liverpool': 0, 'Fulham': 0, 'Hull City': -10, 'Coventry City': 0,
}
elo_2026_27_adjusted = elo_2026_27.copy()
for team, delta in manual_elo_adjustments.items():
    if team in elo_2026_27_adjusted.index:
        elo_2026_27_adjusted[team] += delta
elo_2026_27_adjusted = elo_2026_27_adjusted.sort_values(ascending=False)

print("\n2026-27 starting Elo:")
print(elo_2026_27_adjusted)

# ============================================================
# Save outputs
# ============================================================
rolling_cols_v2 = [c for c in matches_sorted.columns if 'last' in c]
matches_featured_v2 = matches_sorted.dropna(subset=rolling_cols_v2).reset_index(drop=True)

os.makedirs('data', exist_ok=True)
matches_featured_v2.to_csv('data/matches_featured_v2.csv', index=False)
elo_2026_27_adjusted.rename('starting_elo').to_csv('data/starting_elo_2026_27.csv')

print(f"\nSaved data/matches_featured_v2.csv ({len(matches_featured_v2)} rows)")
print("Saved data/starting_elo_2026_27.csv")
