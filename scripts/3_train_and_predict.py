"""
Script 3: Train and predict
- Reads data/matches_featured_v2.csv and data/starting_elo_2026_27.csv
- Fits the Poisson goals model FRESH (retrains every run, per your choice)
- Predicts every 2026-27 fixture
- Runs a 10,000-trial Monte Carlo season simulation
- Saves data/season_simulation_2026_27.csv and a dashboard-ready JSON
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import poisson as poisson_dist
from itertools import permutations
import json
import os
from datetime import datetime, timezone

matches_featured_v2 = pd.read_csv('data/matches_featured_v2.csv', parse_dates=['Date'])
elo_2026_27_adjusted = pd.read_csv('data/starting_elo_2026_27.csv', index_col=0)['starting_elo']

teams_2026_27 = list(elo_2026_27_adjusted.index)
print(f"Training on {len(matches_featured_v2)} matches, predicting for {len(teams_2026_27)} teams")

# ============================================================
# Fit Poisson model
# ============================================================
home_side = matches_featured_v2[['Date', 'HomeTeam', 'AwayTeam', 'home_goals',
    'home_xg_for_last5', 'home_xg_against_last5', 'away_xg_for_last5', 'away_xg_against_last5',
    'home_form_points_last5', 'away_form_points_last5', 'home_elo_pre', 'away_elo_pre']].rename(columns={
    'HomeTeam': 'attack_team', 'AwayTeam': 'defense_team', 'home_goals': 'goals_for',
    'home_xg_for_last5': 'attack_xg_for_last5', 'home_xg_against_last5': 'attack_xg_against_last5',
    'away_xg_for_last5': 'defense_xg_for_last5', 'away_xg_against_last5': 'defense_xg_against_last5',
    'home_form_points_last5': 'attack_form', 'away_form_points_last5': 'defense_form',
    'home_elo_pre': 'attack_elo', 'away_elo_pre': 'defense_elo'})
home_side['is_home'] = 1

away_side = matches_featured_v2[['Date', 'HomeTeam', 'AwayTeam', 'away_goals',
    'away_xg_for_last5', 'away_xg_against_last5', 'home_xg_for_last5', 'home_xg_against_last5',
    'away_form_points_last5', 'home_form_points_last5', 'away_elo_pre', 'home_elo_pre']].rename(columns={
    'AwayTeam': 'attack_team', 'HomeTeam': 'defense_team', 'away_goals': 'goals_for',
    'away_xg_for_last5': 'attack_xg_for_last5', 'away_xg_against_last5': 'attack_xg_against_last5',
    'home_xg_for_last5': 'defense_xg_for_last5', 'home_xg_against_last5': 'defense_xg_against_last5',
    'away_form_points_last5': 'attack_form', 'home_form_points_last5': 'defense_form',
    'away_elo_pre': 'attack_elo', 'home_elo_pre': 'defense_elo'})
away_side['is_home'] = 0

poisson_train = pd.concat([home_side, away_side], ignore_index=True)
poisson_train['elo_diff'] = poisson_train['attack_elo'] - poisson_train['defense_elo']

poisson_model = smf.glm(
    formula='goals_for ~ attack_xg_for_last5 + attack_xg_against_last5 + '
            'defense_xg_for_last5 + defense_xg_against_last5 + '
            'attack_form + defense_form + elo_diff + is_home',
    data=poisson_train, family=sm.families.Poisson()
).fit()
print("Poisson model fit complete.")

# ============================================================
# Predict fixtures — snapshot + fallback for no-history teams
# ============================================================
def get_latest_team_snapshot_raw(team, matches_df):
    team_home = matches_df[matches_df['HomeTeam'] == team][
        ['Date', 'home_xg_for_last5', 'home_xg_against_last5', 'home_form_points_last5']
    ].rename(columns={'home_xg_for_last5': 'xg_for_last5', 'home_xg_against_last5': 'xg_against_last5',
                       'home_form_points_last5': 'form_last5'})
    team_away = matches_df[matches_df['AwayTeam'] == team][
        ['Date', 'away_xg_for_last5', 'away_xg_against_last5', 'away_form_points_last5']
    ].rename(columns={'away_xg_for_last5': 'xg_for_last5', 'away_xg_against_last5': 'xg_against_last5',
                       'away_form_points_last5': 'form_last5'})
    return pd.concat([team_home, team_away]).sort_values('Date').iloc[-1]

def get_league_average_snapshot(matches_df):
    all_teams = pd.concat([matches_df['HomeTeam'], matches_df['AwayTeam']]).unique()
    snapshots = []
    for t in all_teams:
        try:
            snapshots.append(get_latest_team_snapshot_raw(t, matches_df))
        except IndexError:
            continue
    return pd.DataFrame(snapshots).mean()

league_avg_snapshot = get_league_average_snapshot(matches_featured_v2)

def get_latest_team_snapshot(team, matches_df):
    try:
        return get_latest_team_snapshot_raw(team, matches_df)
    except IndexError:
        return league_avg_snapshot

def predict_fixture(home_team, away_team, matches_df, elo_series, model):
    h, a = get_latest_team_snapshot(home_team, matches_df), get_latest_team_snapshot(away_team, matches_df)
    home_elo, away_elo = elo_series[home_team], elo_series[away_team]

    home_row = pd.DataFrame([{'attack_xg_for_last5': h['xg_for_last5'], 'attack_xg_against_last5': h['xg_against_last5'],
        'defense_xg_for_last5': a['xg_for_last5'], 'defense_xg_against_last5': a['xg_against_last5'],
        'attack_form': h['form_last5'], 'defense_form': a['form_last5'],
        'elo_diff': home_elo - away_elo, 'is_home': 1}])
    away_row = pd.DataFrame([{'attack_xg_for_last5': a['xg_for_last5'], 'attack_xg_against_last5': a['xg_against_last5'],
        'defense_xg_for_last5': h['xg_for_last5'], 'defense_xg_against_last5': h['xg_against_last5'],
        'attack_form': a['form_last5'], 'defense_form': h['form_last5'],
        'elo_diff': away_elo - home_elo, 'is_home': 0}])
    return model.predict(home_row).iloc[0], model.predict(away_row).iloc[0]

# ============================================================
# Monte Carlo season simulation
# ============================================================
fixtures = list(permutations(teams_2026_27, 2))
fixture_lambdas = []
for home_team, away_team in fixtures:
    lam_h, lam_a = predict_fixture(home_team, away_team, matches_featured_v2, elo_2026_27_adjusted, poisson_model)
    fixture_lambdas.append((home_team, away_team, lam_h, lam_a))
fixture_df = pd.DataFrame(fixture_lambdas, columns=['HomeTeam', 'AwayTeam', 'lambda_home', 'lambda_away'])
print(f"Predicted {len(fixture_df)} fixtures")

N_TRIALS = 10000
n_fixtures = len(fixture_df)
lambda_home_arr = fixture_df['lambda_home'].values
lambda_away_arr = fixture_df['lambda_away'].values

home_goals_sim = np.random.poisson(lam=lambda_home_arr, size=(N_TRIALS, n_fixtures))
away_goals_sim = np.random.poisson(lam=lambda_away_arr, size=(N_TRIALS, n_fixtures))

home_points_sim = np.where(home_goals_sim > away_goals_sim, 3, np.where(home_goals_sim == away_goals_sim, 1, 0))
away_points_sim = np.where(away_goals_sim > home_goals_sim, 3, np.where(home_goals_sim == away_goals_sim, 1, 0))

team_list = teams_2026_27
team_idx = {t: i for i, t in enumerate(team_list)}
n_teams = len(team_list)

home_indicator = np.zeros((n_fixtures, n_teams))
away_indicator = np.zeros((n_fixtures, n_teams))
for i, (h, a) in enumerate(zip(fixture_df['HomeTeam'], fixture_df['AwayTeam'])):
    home_indicator[i, team_idx[h]] = 1
    away_indicator[i, team_idx[a]] = 1

team_points = home_points_sim @ home_indicator + away_points_sim @ away_indicator
home_gd_sim = home_goals_sim - away_goals_sim
away_gd_sim = away_goals_sim - home_goals_sim
team_gd = home_gd_sim @ home_indicator + away_gd_sim @ away_indicator

combined_score = team_points + team_gd / 1000.0
ranks = np.argsort(-combined_score, axis=1)

results = []
for i, team in enumerate(team_list):
    team_ranks = np.where(ranks == i)[1]
    results.append({
        'team': team,
        'avg_points': round(float(team_points[:, i].mean()), 1),
        'title_pct': round(float((team_ranks == 0).mean() * 100), 2),
        'top4_pct': round(float((team_ranks < 4).mean() * 100), 2),
        'relegation_pct': round(float((team_ranks >= 17).mean() * 100), 2),
    })

season_summary = pd.DataFrame(results).sort_values('avg_points', ascending=False).reset_index(drop=True)
print(season_summary)

# ============================================================
# Save outputs — CSV for you, JSON for the dashboard
# ============================================================
os.makedirs('data', exist_ok=True)
os.makedirs('docs', exist_ok=True)

season_summary.to_csv('data/season_simulation_2026_27.csv', index=False)

dashboard_data = {
    'last_updated': datetime.now(timezone.utc).isoformat(),
    'n_trials': N_TRIALS,
    'standings': results,
}
with open('docs/predictions.json', 'w') as f:
    json.dump(dashboard_data, f, indent=2)

print("\nSaved data/season_simulation_2026_27.csv and docs/predictions.json")
