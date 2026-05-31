import io
import requests
import numpy as np
import pandas as pd
import streamlit as ui
from sklearn.ensemble import RandomForestClassifier

# Set page title and web layout configuration
ui.set_page_config(page_title="Tennis Prediction Engine", layout="centered")
ui.title("🎾 Pro Tennis Prediction Dashboard")
ui.write("Enter an upcoming matchup below to calculate true win probabilities.")

# =====================================================================
# BACKGROUND: CACHED SYSTEM LOADING (Ensures web page stays lightning fast)
# =====================================================================
@ui.cache_resource
def initialize_and_train_model():
    years = [2023, 2024, 2025, 2026]
    data_frames = []
    for year in years:
        url = f"https://githubusercontent.com_{year}.csv"
        res = requests.get(url)
        if res.status_code == 200:
            data_frames.append(pd.read_csv(io.StringIO(res.text)))

            
    df = pd.concat(data_frames, ignore_index=True)
    df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d', errors='coerce')
    df = df.dropna(subset=['tourney_date', 'winner_id', 'loser_id', 'surface']).sort_values('tourney_date').reset_index(drop=True)
    df['minutes'] = df['minutes'].fillna(100)
    
    # Run historical feature logs
    match_log, global_elo, h2h_tracker, player_stats_history = [], {}, {}, {}
    surface_elos = {'Hard': {}, 'Clay': {}, 'Grass': {}}
    winner_fatigue, loser_fatigue = [], []
    winner_blended, loser_blended = [], []
    winner_h2h_diff, winner_dom_diff = [], []
    
    for idx, row in df.iterrows():
        current_date, w_id, l_id = row['tourney_date'], row['winner_id'], row['loser_id']
        surf = row['surface'] if row['surface'] in surface_elos else 'Hard'
        
        # Fatigue Lookback
        seven_days_ago = current_date - pd.Timedelta(days=7)
        recent = [m for m in match_log if seven_days_ago <= m['date'] < current_date]
        winner_fatigue.append(sum(m['mins'] for m in recent if m['p1'] == w_id or m['p2'] == w_id))
        loser_fatigue.append(sum(m['mins'] for m in recent if m['p1'] == l_id or m['p2'] == l_id))
        match_log.append({'date': current_date, 'p1': w_id, 'p2': l_id, 'mins': row['minutes']})
        
        # Surface Blended Elo
        w_s, l_s = surface_elos[surf].get(w_id, 1500), surface_elos[surf].get(l_id, 1500)
        w_g, l_g = global_elo.get(w_id, 1500), global_elo.get(l_id, 1500)
        winner_blended.append((0.7 * w_s) + (0.3 * w_g))
        loser_blended.append((0.7 * l_s) + (0.3 * l_g))
        
        exp_w_s = 1 / (1 + 10 ** ((l_s - w_s) / 400))
        surface_elos[surf][w_id] = w_s + 32 * (1 - exp_w_s)
        surface_elos[surf][l_id] = l_s + 32 * (0 - (1 - exp_w_s))
        exp_w_g = 1 / (1 + 10 ** ((l_g - w_g) / 400))
        global_elo[w_id] = w_g + 32 * (1 - exp_w_g)
        global_elo[l_id] = l_g + 32 * (0 - (1 - exp_w_g))
        
        # Head-To-Head
        pair = tuple(sorted([w_id, l_id]))
        if pair not in h2h_tracker: h2h_tracker[pair] = {w_id: 0, l_id: 0}
        winner_h2h_diff.append(h2h_tracker[pair].get(w_id, 0) - h2h_tracker[pair].get(l_id, 0))
        h2h_tracker[pair][w_id] = h2h_tracker[pair].get(w_id, 0) + 1
        
        # Dominance
        w_h, l_h = player_stats_history.get(w_id, []), player_stats_history.get(l_id, [])
        winner_dom_diff.append((np.mean(w_h[-10:]) if len(w_h) > 0 else 1.0) - (np.mean(l_h[-10:]) if len(l_h) > 0 else 1.0))
        try:
            w_m_dom = ((int(row['w_1stWon']) + int(row['w_2ndWon'])) / max(1, int(row['w_svpt']))) + ((int(row['l_svpt']) - (int(row['l_1stWon']) + int(row['l_2ndWon']))) / max(1, int(row['l_svpt'])))
            l_m_dom = ((int(row['l_1stWon']) + int(row['l_2ndWon'])) / max(1, int(row['l_svpt']))) + ((int(row['w_svpt']) - (int(row['w_1stWon']) + int(row['w_2ndWon']))) / max(1, int(row['w_svpt'])))
        except: w_m_dom, l_m_dom = 1.0, 1.0
        if w_id not in player_stats_history: player_stats_history[w_id] = []
        if l_id not in player_stats_history: player_stats_history[l_id] = []
        player_stats_history[w_id].append(w_m_dom)
        player_stats_history[l_id].append(l_m_dom)
        
    df['elo_diff'] = winner_blended - np.array(loser_blended)
    df['fatigue_diff'] = np.array(winner_fatigue) - np.array(loser_fatigue)
    df['h2h_diff'] = winner_h2h_diff
    df['dom_diff'] = winner_dom_diff
    df['target'] = 1  # Standard anchor label for model fitting
    
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(df[['elo_diff', 'fatigue_diff', 'h2h_diff', 'dom_diff']], df['target'])
    
    names = sorted(list(set(df['winner_name'].dropna().unique()) | set(df['loser_name'].dropna().unique())))
    name_to_id = {row['winner_name'].lower(): int(row['winner_id']) for _, row in df.dropna(subset=['winner_name']).iterrows()}
    for _, row in df.dropna(subset=['loser_name']).iterrows(): name_to_id[row['loser_name'].lower()] = int(row['loser_id'])
        
    return model, names, name_to_id, surface_elos, global_elo, h2h_tracker, player_stats_history

model, player_list, name_map, surface_elos, global_elo, h2h_tracker, stats_hist = initialize_and_train_model()

# =====================================================================
# SIDEBAR: BANKROLL MANAGEMENT SETTINGS
# =====================================================================
ui.sidebar.header("💰 Bankroll Settings")
total_bankroll = ui.sidebar.number_input("Total Bankroll ($)", min_value=10, value=1000, step=50)
kelly_fraction = ui.sidebar.slider("Kelly Fraction (Multiplier)", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
ui.sidebar.caption("💡 Fractional Kelly (like 0.5) significantly reduces risk while maintaining high growth.")

# =====================================================================
# FRONTEND INTERACTIVE CONTENT REGION
# =====================================================================
col1, col2 = ui.columns(2)
with col1:
    player_a = ui.selectbox("Select Player A", player_list, index=player_list.index("Carlos Alcaraz") if "Carlos Alcaraz" in player_list else 0)
    p1_mins = ui.number_input("Player A Minutes Played Last 7 Days", min_value=0, value=90, step=10)
    odds_a = ui.number_input("Player A Bookmaker Odds (Decimal)", min_value=1.01, value=1.95, step=0.05)
with col2:
    player_b = ui.selectbox("Select Player B", player_list, index=player_list.index("Jannik Sinner") if "Jannik Sinner" in player_list else 1)
    p2_mins = ui.number_input("Player B Minutes Played Last 7 Days", min_value=0, value=180, step=10)
    odds_b = ui.number_input("Player B Bookmaker Odds (Decimal)", min_value=1.01, value=1.85, step=0.05)

surface_type = ui.selectbox("Match Court Surface", ["Hard", "Clay", "Grass"])

if ui.button("⚡ Calculate Odds & Wager Size", use_container_width=True):
    id_a, id_b = name_map[player_a.lower()], name_map[player_b.lower()]
    
    elo_a = (0.7 * surface_elos[surface_type].get(id_a, 1500)) + (0.3 * global_elo.get(id_a, 1500))
    elo_b = (0.7 * surface_elos[surface_type].get(id_b, 1500)) + (0.3 * global_elo.get(id_b, 1500))
    
    pair = tuple(sorted([id_a, id_b]))
    h2h_diff = h2h_tracker.get(pair, {}).get(id_a, 0) - h2h_tracker.get(pair, {}).get(id_b, 0)
    dom_diff = np.mean(stats_hist.get(id_a, [1.0])[-10:]) - np.mean(stats_hist.get(id_b, [1.0])[-10:])
    
    input_row = pd.DataFrame([{'elo_diff': elo_a - elo_b, 'fatigue_diff': p1_mins - p2_mins, 'h2h_diff': h2h_diff, 'dom_diff': dom_diff}])
       prob_matrix = model.predict_proba(input_row)[0]
    
    prob_a = float(prob_matrix[1]) if len(prob_matrix) > 1 else 0.50
    prob_b = 1.0 - prob_a

    
    # Calculate Implied Probabilities from Bookmaker Odds
    implied_a = 1 / odds_a
    implied_b = 1 / odds_b
    
    ui.divider()
    ui.subheader("📊 Model Matchup Probability Forecast")
    
    ui.write(f"**{player_a}**: Model: `{prob_a*100:.1f}%` | Market Implied: `{implied_a*100:.1f}%`")
    ui.write(f"**{player_b}**: Model: `{prob_b*100:.1f}%` | Market Implied: `{implied_b*100:.1f}%`")
    
    # Kelly Criterion Bankroll Engine
    ui.divider()
    ui.subheader("💰 Smart Bet Allocation Strategy")
    
    def calculate_kelly(prob, odds):
        b = odds - 1
        q = 1.0 - prob
        return max(0.0, (prob * b - q) / b)

    kelly_a = calculate_kelly(prob_a, odds_a) * kelly_fraction
    kelly_b = calculate_kelly(prob_b, odds_b) * kelly_fraction
    
    if kelly_a > 0:
        wager = total_bankroll * kelly_a
        ui.success(f"✅ **Value Found on {player_a}!**")
        ui.write(f"Suggested Allocation: **{kelly_a*100:.1f}%** of your bankroll.")
        ui.write(f"Recommended Wager Amount: **${wager:.2f}**")
    elif kelly_b > 0:
        wager = total_bankroll * kelly_b
        ui.success(f"✅ **Value Found on {player_b}!**")
        ui.write(f"Suggested Allocation: **{kelly_b*100:.1f}%** of your bankroll.")
        ui.write(f"Recommended Wager Amount: **${wager:.2f}**")
    else:
        ui.warning("❌ **No Betting Value Found.** The bookmaker's prices are too efficient compared to the model's edge.")
