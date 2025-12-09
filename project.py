# %% Download the dataset
import numpy as np
import kagglehub
import ast
import os
import sqlite3
import pandas as pd
import warnings

# Suppress sklearn and other annoying warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', module='sklearn')
# Also suppress deprecation warnings from sklearn
import sklearn
warnings.filterwarnings('ignore', category=DeprecationWarning, module='sklearn')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import torch.nn as nn
import torch
from torch.distributions import Categorical
from typing import List

# %% Read the dataset
from inspect_db import load_all_data, inspect_database

# Load all data from database
games, rounds, scoreboard, matches = load_all_data()

# Inspect the loaded data
inspect_database(games, rounds, scoreboard, matches)

# %% Clean rounds data
from clean_rounds import clean_rounds_data, show_sample_cleaned_data

# Clean the rounds data - extract RoundHistory into separate rows (one row per round)
rounds_cleaned = clean_rounds_data(rounds, games)

# Show sample of cleaned data
show_sample_cleaned_data(rounds_cleaned, n_games=1)

# %% Compute score difference
# Calculate score difference: Team1Score - Team2Score
rounds_cleaned['ScoreDifference'] = rounds_cleaned['Team1Score'] - rounds_cleaned['Team2Score']

# %% Prepare features for logistic regression (round-level prediction)
# Goal: Predict if Team1 wins the round

# Create additional features
# 1. Bank difference (economic advantage)
rounds_cleaned['BankDifference'] = rounds_cleaned['Team1Bank'] - rounds_cleaned['Team2Bank']

# 2. Encode buy types as binary indicators (one-hot style)
# Team1 buy types
rounds_cleaned['Team1_IsEco'] = rounds_cleaned['Team1BuyType'].str.lower().str.contains('eco', na=False).astype(int)
rounds_cleaned['Team1_IsSemi'] = rounds_cleaned['Team1BuyType'].str.lower().str.contains('semi', na=False).astype(int)
rounds_cleaned['Team1_IsFull'] = rounds_cleaned['Team1BuyType'].str.lower().str.contains('full', na=False).astype(int)

# Team2 buy types
rounds_cleaned['Team2_IsEco'] = rounds_cleaned['Team2BuyType'].str.lower().str.contains('eco', na=False).astype(int)
rounds_cleaned['Team2_IsSemi'] = rounds_cleaned['Team2BuyType'].str.lower().str.contains('semi', na=False).astype(int)
rounds_cleaned['Team2_IsFull'] = rounds_cleaned['Team2BuyType'].str.lower().str.contains('full', na=False).astype(int)

# 3. Game phase indicators
rounds_cleaned['IsEarlyGame'] = (rounds_cleaned['RoundNumber'] <= 6).astype(int)
rounds_cleaned['IsMidGame'] = ((rounds_cleaned['RoundNumber'] > 6) & (rounds_cleaned['RoundNumber'] <= 12)).astype(int)
rounds_cleaned['IsLateGame'] = (rounds_cleaned['RoundNumber'] > 12).astype(int)

# 4. Create target variable: 1 if Team1 wins the round, 0 if Team2 wins
# We need to determine which team won from RoundWinner
def get_team1_win(row):
    """
    Convert RoundWinner abbreviation to binary target.
    
    RoundWinner is like "BOOS" or "PHO " (team abbreviations)
    We need to match it to Team1's name to create binary: 1 = Team1 won, 0 = Team2 won
    """
    if pd.isna(row['RoundWinner']):
        return None
    round_winner = str(row['RoundWinner']).strip().upper()
    team1_name = str(row.get('Team1', '')).strip() if pd.notna(row.get('Team1')) else ''
    
    if team1_name:
        # Create abbreviation from team name (first letters of each word)
        team1_abbrev = ''.join([w[0] for w in team1_name.split()]).upper()[:4]
        if round_winner.startswith(team1_abbrev) or team1_abbrev.startswith(round_winner):
            return 1
    return 0

rounds_cleaned['Team1_WinsRound'] = rounds_cleaned.apply(get_team1_win, axis=1)

# Recommended feature columns for round-level prediction:
round_feature_cols = [
    # Score information (most important)
    'ScoreDifference',           # Current score advantage
    
    # Economic information
    'BankDifference',            # Economic advantage (Team1Bank - Team2Bank)

    
    # Buy type indicators
    'Team1_IsEco',              # Team1 eco round
    'Team1_IsFull',             # Team1 full buy
    'Team2_IsEco',              # Team2 eco round
    'Team2_IsFull',             # Team2 full buy
    
    # Game phase
    'RoundNumber',              # Round number
    'IsEarlyGame',              # Early game indicator
    'IsMidGame',                # Mid game indicator
    'IsLateGame',               # Late game indicator
]

print("\n=== Recommended Features for Round-Level Logistic Regression ===")
print("Features to predict: Team1_WinsRound (1 = Team1 wins, 0 = Team2 wins)")
print(f"\nSelected {len(round_feature_cols)} features:")
for i, feat in enumerate(round_feature_cols, 1):
    print(f"  {i:2d}. {feat}")

# %% Create model for round-level prediction
# Use cleaned rounds data with round-level features
rounds_model = rounds_cleaned.copy()

# Prepare features and target
# Drop rows with missing values in features or target
print("\n=== Preparing Data for Model ===")
print(f"Initial rows: {len(rounds_model)}")

# Check which features exist and drop rows with missing values
available_features = [f for f in round_feature_cols if f in rounds_model.columns]
missing_features = [f for f in round_feature_cols if f not in rounds_model.columns]

if missing_features:
    print(f"Warning: Missing features: {missing_features}")
    print(f"Using available features: {available_features}")

rounds_model = rounds_model.dropna(subset=available_features + ['Team1_WinsRound'])

print(f"Rows after dropping missing values: {len(rounds_model)}")
print(f"Target distribution:")
print(rounds_model['Team1_WinsRound'].value_counts())

# Split into features and target
X = rounds_model[available_features]
y = rounds_model['Team1_WinsRound']

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")

# %% Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# %% Train model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %% Run Logistic regression
print("\n=== Training Logistic Regression Model ===")
logit = LogisticRegression(max_iter=2000, random_state=42)
logit.fit(X_train_scaled, y_train)

# Store feature names for environment
LOGIT_FEATURE_NAMES = available_features.copy()

# %% Create prediction and measure accuracy
y_pred = logit.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

print("\n=== Model Results ===")
print(f"Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Show feature importance (coefficients)
print("\n=== Feature Importance (Coefficients) ===")
feature_importance = pd.DataFrame({
    'Feature': available_features,
    'Coefficient': logit.coef_[0]
}).sort_values('Coefficient', ascending=False)

print(feature_importance.to_string(index=False))

#%% Team 2 policy
ACTION_ID_TO_NAME = {
    0: "eco",
    1: "semi",
    2: "full",
}

def sample_action_from_probs(action_ids, probs, rng=None):
    """
    action_ids: list like [0,1,2]
    probs: list/array of same length, sum to 1
    rng: optional np.random.Generator
    """
    if rng is None:
        rng = np.random.default_rng()
    return int(rng.choice(action_ids, p=probs))

def team2_policy_aggressive(team2_bank, rng=None):
    """
    Aggressive:
      - If bank >= 21000 (5 * 4200): always FULL
      - Else if bank >= 10000: SEMI (still spending)
      - Else: ECO
    """
    if team2_bank >= 5 * 4200:  # 21000
        action_id = 2  # full
    elif team2_bank >= 10000:
        action_id = 1  # semi
    else:
        action_id = 0  # eco

    return action_id, ACTION_ID_TO_NAME[action_id]

def team2_policy_middle(team2_bank, rng=None):
    """
    Middle / balanced strategy:
      - If bank >= 21000 (5 * 4200): [eco 0.3, semi 0.4, full 0.3]
      - If 10000 <= bank < 21000: [eco 0.5, semi 0.5, full 0]
      - If bank < 10000: [eco 1, semi 0, full 0]
    """
    action_ids = [0, 1, 2]

    if team2_bank >= 5 * 4200:
        probs = [0.3, 0.4, 0.3]
    elif team2_bank >= 10000:
        probs = [0.5, 0.5, 0]
    else:
        probs = [1, 0, 0]

    action_id = sample_action_from_probs(action_ids, probs, rng)
    return action_id, ACTION_ID_TO_NAME[action_id]

def team2_policy_conservative(team2_bank, rng=None):
    """
    Conservative:
      - If bank >= 30000: FULL
      - If 21000 (5 * 4200) <= bank < 30000: SEMI
      - If bank < 21000: ECO
    """
    if team2_bank >= 30000:
        action_id = 2  # full
    elif team2_bank >= 5 * 4200:
        action_id = 1  # semi
    else:
        action_id = 0  # eco

    return action_id, ACTION_ID_TO_NAME[action_id]


# %% Econ Rule
# Core Valorant economy rules (standard comp/unrated) :contentReference[oaicite:0]{index=0}
# Team-level constants (5 players)
TEAM_START_CREDITS = 5 * 800      # pistol start
TEAM_MAX_CREDITS   = 5 * 9000     # soft cap

# Team-level win/loss bonus (5 players * per-player bonus)
TEAM_WIN_REWARD    = 5 * 3000     # 15000
TEAM_LOSS_BONUS_1  = 5 * 1900     #  9500
TEAM_LOSS_BONUS_2  = 5 * 2400     # 12000
TEAM_LOSS_BONUS_3P = 5 * 2900     # 14500

TEAM_LOSS_BONUSES = [TEAM_LOSS_BONUS_1, TEAM_LOSS_BONUS_2, TEAM_LOSS_BONUS_3P]

# Team-level spend for each buy type (very rough, just pick something consistent)
BUY_COST_TEAM = {
    0: 5 * 800,   # eco: everyone does cheap buy
    1: 5 * 2000,  # semi: everyone spends mid
    2: 5 * 4200,  # full: everyone fully buys
}

def team_income(outcome: str, loss_streak_before: int):
    """
    outcome: "win" or "loss"
    loss_streak_before: how many losses in a row before this round

    Returns:
      income_team (credits), loss_streak_after
    """
    outcome = outcome.lower()
    if outcome == "win":
        return TEAM_WIN_REWARD, 0

    # loss
    new_streak = loss_streak_before + 1
    idx = min(new_streak - 1, 2)  # 0,1,2
    income = TEAM_LOSS_BONUSES[idx]
    return income, new_streak

# %% Helper functions
def check_game_end(team1_score, team2_score, win_threshold=13):
    """
    Returns True if either team has reached the win threshold.
    """
    if team1_score >= win_threshold:
        return True, "team1"
    if team2_score >= win_threshold:
        return True, "team2"
    return False, None

def update_team_bank_simple(team_bank_before, action_id, outcome, loss_streak_before):
    """Simplified team credit system with spend + win/loss bonus."""
    # 1. Spend credits
    spend = BUY_COST_TEAM[action_id]
    spend = min(spend, team_bank_before)  # can't spend more than bank
    bank_after = team_bank_before - spend

    # 2. Add income from win/loss (via your function)
    income, loss_streak_after = team_income(outcome, loss_streak_before)
    bank_after += income

    # 3. Cap at team max
    bank_after = min(bank_after, TEAM_MAX_CREDITS)

    return bank_after, loss_streak_after
# %% Create Valorant Environment
class ValorantEnv:
    def __init__(
        self,
        team1_bank_init=4000.0,
        team2_bank_init=4000.0,
        team2_policy=team2_policy_aggressive,  # default
        max_rounds=24,
        rng_seed=10086,
        logit_model=None,  # Trained logistic regression model
        scaler=None,       # Scaler used for features
        feature_names=None,  # List of feature names in order
        econ_weight=0.0,     # Weight for economic discount (0 = no discount, higher = more penalty for spending)
        round_weight=1.0,    # Weight for round number importance (1.0 = all rounds equal, >1.0 = later rounds matter more)
    ):
        self.team1_bank_init = team1_bank_init
        self.team2_bank_init = team2_bank_init
        self.team2_policy    = team2_policy
        self.max_rounds      = max_rounds
        self.logit_model     = logit_model
        self.scaler          = scaler
        self.feature_names   = feature_names
        self.econ_weight     = econ_weight  # Controls penalty for spending money
        self.round_weight    = round_weight  # Controls importance of later rounds

        self.rng = np.random.default_rng(rng_seed)

        self.reset()

    def reset(self):
        """Reset environment to start of game."""
        self.round_number      = 1
        self.team1_bank        = self.team1_bank_init
        self.team2_bank        = self.team2_bank_init
        self.team1_losestreak  = 0
        self.team2_losestreak  = 0
        self.team1_score       = 0
        self.team2_score       = 0

        state = {
            "round_number": self.round_number,
            "team1_bank": self.team1_bank,
            "team2_bank": self.team2_bank,
            "team1_losestreak": self.team1_losestreak,
            "team2_losestreak": self.team2_losestreak,
            "team1_score": self.team1_score,
            "team2_score": self.team2_score,
        }
        return state


    def _state_to_features(self, state, action1, action2):
        """
        Convert current state and actions to feature vector for logistic regression.
        
        Args:
            state: Current state dictionary
            action1: Team1's action (0=eco, 1=semi, 2=full)
            action2: Team2's action (0=eco, 1=semi, 2=full)
        
        Returns:
            Feature vector as numpy array
        """
        features = {}
        
        # Score information
        features['ScoreDifference'] = state['team1_score'] - state['team2_score']
        
        # Economic information
        features['BankDifference'] = state['team1_bank'] - state['team2_bank']
        
        # Buy type indicators
        features['Team1_IsEco'] = 1 if action1 == 0 else 0
        features['Team1_IsFull'] = 1 if action1 == 2 else 0
        features['Team2_IsEco'] = 1 if action2 == 0 else 0
        features['Team2_IsFull'] = 1 if action2 == 2 else 0
        
        # Game phase
        features['RoundNumber'] = state['round_number']
        features['IsEarlyGame'] = 1 if state['round_number'] <= 6 else 0
        features['IsMidGame'] = 1 if 6 < state['round_number'] <= 12 else 0
        features['IsLateGame'] = 1 if state['round_number'] > 12 else 0
        
        # Convert to array in the same order as feature_names
        if self.feature_names:
            feature_vector = np.array([features.get(feat, 0) for feat in self.feature_names])
        else:
            # Fallback: use order from available_features
            feature_vector = np.array([
                features['ScoreDifference'],
                features['BankDifference'],
                features['Team1_IsEco'],
                features['Team1_IsFull'],
                features['Team2_IsEco'],
                features['Team2_IsFull'],
                features['RoundNumber'],
                features['IsEarlyGame'],
                features['IsMidGame'],
                features['IsLateGame'],
            ])
        
        return feature_vector

    def step(self, action1):

        # -------------------------
        # 1. Team 2 action
        # -------------------------
        action2, action2_name = self.team2_policy(self.team2_bank, rng=self.rng)

        # -------------------------
        # 2. Determine round outcome using logistic regression
        # -------------------------
        if self.logit_model is not None and self.scaler is not None:
            # Get current state
            current_state = {
                'round_number': self.round_number,
                'team1_bank': self.team1_bank,
                'team2_bank': self.team2_bank,
                'team1_score': self.team1_score,
                'team2_score': self.team2_score,
            }
            
            # Convert state to features
            feature_vector = self._state_to_features(current_state, action1, action2)
            feature_vector = feature_vector.reshape(1, -1)
            
            # Scale features
            feature_vector_scaled = self.scaler.transform(feature_vector)
            
            # Predict probability that Team1 wins
            prob_team1_wins = self.logit_model.predict_proba(feature_vector_scaled)[0, 1]
            
            # Sample outcome based on probability
            team1_wins = self.rng.random() < prob_team1_wins
        else:
            team1_wins = self.rng.random() < 0.5

        outcome1 = "win"  if team1_wins else "loss"
        outcome2 = "loss" if team1_wins else "win"

        # Calculate RL reward with economic and round number weighting
        base_reward = 1.0 if outcome1 == "win" else 0.0
        
        # Economic discount: penalize spending money
        # Higher econ_weight means spending more reduces reward
        team1_spend = BUY_COST_TEAM[action1]
        # Normalize spend by max possible spend (full buy = 21000)
        normalized_spend = team1_spend / BUY_COST_TEAM[2]  # Divide by full buy cost
        econ_penalty = self.econ_weight * normalized_spend
        
        # Round number weighting: later rounds matter more
        # Normalize round number (1 to max_rounds) to [0, 1] range
        normalized_round = (self.round_number - 1) / (self.max_rounds - 1) if self.max_rounds > 1 else 0
        # Apply round weight: 1.0 = all rounds equal, >1.0 = later rounds amplified
        round_multiplier = 1.0 + (self.round_weight - 1.0) * normalized_round
        
        # Final reward: base reward * round multiplier - economic penalty
        reward = base_reward * round_multiplier - econ_penalty
        
        # Ensure reward is non-negative (optional, can remove if negative rewards are desired)
        reward = max(reward, 0.0)

        # -------------------------
        # 3. Update score
        # -------------------------
        if outcome1 == "win":
            self.team1_score += 1
        else:
            self.team2_score += 1

        # -------------------------
        # 4. Update banks
        # -------------------------
        self.team1_bank, self.team1_losestreak = update_team_bank_simple(
            team_bank_before   = self.team1_bank,
            action_id          = action1,
            outcome            = outcome1,
            loss_streak_before = self.team1_losestreak
        )

        self.team2_bank, self.team2_losestreak = update_team_bank_simple(
            team_bank_before   = self.team2_bank,
            action_id          = action2,
            outcome            = outcome2,
            loss_streak_before = self.team2_losestreak
        )

        # -------------------------
        # 5. Game-end check
        # -------------------------
        ended, winner = check_game_end(self.team1_score, self.team2_score)
        if ended:
            done = True
        else:
            self.round_number += 1
            done = (self.round_number > self.max_rounds)

        # -------------------------
        # 6. Build next state
        # -------------------------
        next_state = {
            "round_number": self.round_number,
            "team1_bank": self.team1_bank,
            "team2_bank": self.team2_bank,
            "team1_losestreak": self.team1_losestreak,
            "team2_losestreak": self.team2_losestreak,
            "team1_score": self.team1_score,
            "team2_score": self.team2_score,
            "team2_action": action2,
            "team2_action_name": action2_name,
        }

        info = {
            "team1_wins_round": team1_wins,
            "team2_action": action2,
            "team2_action_name": action2_name,
            "game_winner": winner if done else None,
        }

        return next_state, reward, done, info
# %% ============================================================
# REINFORCEMENT LEARNING SETUP
# ============================================================
print("\n" + "="*60)
print("REINFORCEMENT LEARNING SETUP")
print("="*60)

# %% Step 1: Create Environment with Trained Logistic Regression Model
print("\n--- Step 1: Creating Environment ---")

# Create environment with the trained logistic regression model
env = ValorantEnv(
    team1_bank_init=4000.0,
    team2_bank_init=4000.0,
    team2_policy=team2_policy_aggressive,  # Fixed Team2 policy
    max_rounds=24,
    rng_seed=42,
    logit_model=logit,                    # Use trained model for round prediction
    scaler=scaler,                        # Use same scaler
    feature_names=LOGIT_FEATURE_NAMES,    # Use exact feature names from training
    econ_weight=0.0,                      # Economic penalty weight (0 = no penalty)
    round_weight=1.0,                     # Round importance weight (1.0 = all equal)
)

print("✓ Environment created with logistic regression model")
print(f"  - Team2 Policy: aggressive")
print(f"  - Economic weight: {env.econ_weight}")
print(f"  - Round weight: {env.round_weight}")

# Test environment reset
test_state = env.reset()
print(f"\n✓ Environment reset successful")
print(f"  Initial state keys: {list(test_state.keys())}")
print(f"  Initial state: round={test_state['round_number']}, "
      f"Team1 bank={test_state['team1_bank']}, Team2 bank={test_state['team2_bank']}")


# %% Step 2: Convert State to Tensor
STATE_KEYS = [
    "round_number",
    "team1_bank",
    "team2_bank",
    "team1_losestreak",
    "team2_losestreak",
    "team1_score",
    "team2_score"
]

def state_to_tensor(state: dict) -> torch.Tensor:
    """
    Convert the environment state dict into a 1D float tensor.
    Only includes variables listed in STATE_KEYS.
    """
    vals = []
    for key in STATE_KEYS:
        vals.append(float(state[key]))   # ensure numeric
    return torch.tensor(vals, dtype=torch.float32)

# Test state_to_tensor function (commented out)
# print("\n--- Step 2: Testing State to Tensor Conversion ---")

# Test with the actual state from environment
# test_state = env.reset()
# print(f"\nTest state from environment:")
# for key, value in test_state.items():
#     if key in STATE_KEYS:
#         print(f"  {key}: {value}")

# Convert to tensor
# test_tensor = state_to_tensor(test_state)
# print(f"\n✓ Converted to tensor:")
# print(f"  Tensor shape: {test_tensor.shape}")
# print(f"  Tensor values: {test_tensor}")
# print(f"  Tensor dtype: {test_tensor.dtype}")

# Test with a different state
# test_state2 = {
#     "round_number": 4,
#     "team1_bank": 8000,
#     "team2_bank": 12000,
#     "team1_losestreak": 1,
#     "team2_losestreak": 0,
#     "team1_score": 3,
#     "team2_score": 2,
#     "team2_action": 2,  # Extra key (should be ignored)
#     "team2_action_name": "full"  # Extra key (should be ignored)
# }

# test_tensor2 = state_to_tensor(test_state2)
# print(f"\n✓ Test with different state:")
# print(f"  State: round={test_state2['round_number']}, Team1 bank={test_state2['team1_bank']}")
# print(f"  Tensor: {test_tensor2}")

# print(f"\n✓ State to tensor conversion working correctly!")
# print(f"  State dimension: {len(STATE_KEYS)} values")
# %% Step 3: Create Policy Network
print("\n--- Step 3: Creating Policy Network ---")

class PolicyNet(nn.Module):
    """
    Neural network policy that outputs action probabilities.
    
    Takes game state (7 values) and outputs probabilities for 3 actions:
    - Action 0: Eco
    - Action 1: Semi
    - Action 2: Full
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1),  # Ensures probabilities sum to 1
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: state -> action probabilities
        
        Args:
            state: shape [state_dim] or [batch_size, state_dim]
        
        Returns:
            action probabilities: shape [action_dim] or [batch_size, action_dim]
            Probabilities sum to 1.0
        """
        return self.net(state)

# Create policy network
state_dim = len(STATE_KEYS)  # 7: round, banks, scores, loss streaks
action_dim = 3  # eco, semi, full

policy = PolicyNet(state_dim=state_dim, action_dim=action_dim)

print(f"✓ Policy network created")
print(f"  - Input dimension: {state_dim} (state features)")
print(f"  - Output dimension: {action_dim} (action probabilities)")
print(f"  - Architecture: {state_dim} -> 128 -> {action_dim}")

# Test the policy with a sample state
print("\n  Testing policy network...")
test_state_dict = env.reset()
test_state_tensor = state_to_tensor(test_state_dict)
test_probs = policy(test_state_tensor)

print(f"  Sample state: {test_state_tensor}")
print(f"  Action probabilities: {test_probs}")
print(f"  Probabilities sum: {test_probs.sum().item():.4f} (should be 1.0)")
print(f"  Most likely action: {test_probs.argmax().item()} ({ACTION_ID_TO_NAME[test_probs.argmax().item()]})")

print("\n✓ Policy network is working correctly!")


# %% Step 4: Test Running One Episode
print("\n--- Step 4: Testing One Episode ---")

def run_one_episode(env: ValorantEnv, policy: PolicyNet):
    """
    Play one full game (episode) using the current policy.
    
    Args:
        env: ValorantEnv environment
        policy: PolicyNet neural network
    
    Returns:
        rewards:  list of rewards per step (round)
        logps:    list of log π(a_t|s_t) per step (PyTorch tensors)
        total_return: sum of rewards in the episode
    """
    state = env.reset()
    done = False

    rewards = []
    logps   = []

    while not done:
        # 1. Convert state dict -> tensor
        s_t = state_to_tensor(state)     # shape [state_dim]

        # 2. Get action probs from policy
        probs = policy(s_t)              # shape [3] = [p_eco, p_semi, p_full]

        # 3. Define a categorical distribution and sample an action
        dist = Categorical(probs)
        action = dist.sample()           # scalar {0,1,2}
        logp = dist.log_prob(action)     # log π(a_t|s_t), scalar tensor

        # 4. Step the environment with Team1's action
        next_state, reward, done, info = env.step(action.item())

        # 5. Store reward and log-prob
        rewards.append(float(reward))
        logps.append(logp)

        # 6. Move to next state
        state = next_state

    total_return = sum(rewards)
    return rewards, logps, total_return

# Test running one episode
print("Running one test episode...")
rewards, logps, total_return = run_one_episode(env, policy)

print(f"\n✓ Episode completed successfully!")
print(f"  - Number of rounds: {len(rewards)}")
print(f"  - Total return (sum of rewards): {total_return:.2f}")
print(f"  - Average reward per round: {total_return/len(rewards):.4f}")
print(f"  - First 5 rewards: {rewards[:5]}")
print(f"  - Last 5 rewards: {rewards[-5:]}")
print(f"  - Number of log-probabilities stored: {len(logps)}")

# %% Step 5: Discounted Returns Function
print("\n--- Step 5: Discounted Returns Function ---")

def discounted_returns(rewards: List[float], gamma: float = 0.99) -> torch.Tensor:
    """
    Compute discounted returns G_t for a single episode:
        G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...
    
    This gives the total future reward from each time step.
    
    Args:
        rewards: list of rewards [r_0, ..., r_{T-1}]
        gamma: discount factor (0.99 = future rewards matter almost as much as current)
    
    Returns:
        torch.Tensor of shape [T] with discounted returns for each time step
    """
    G = []
    running = 0.0
    # Work backwards: last reward has no future rewards, first reward sees all future
    for r in reversed(rewards):
        running = r + gamma * running
        G.append(running)
    G.reverse()
    return torch.tensor(G, dtype=torch.float32)

# Test discounted returns
print("Testing discounted returns function...")
test_rewards = [1.0, 0.0, 1.0, 0.0, 1.0]
test_returns = discounted_returns(test_rewards, gamma=0.99)
print(f"  Example rewards: {test_rewards}")
print(f"  Discounted returns: {test_returns}")
print(f"  ✓ Discounted returns working correctly!")

# %% Step 6: Training Function (REINFORCE Algorithm)
print("\n--- Step 6: REINFORCE Training Function ---")

def train_valorant_agent(
    env: ValorantEnv,
    policy: PolicyNet,
    episodes: int = 300,
    gamma: float = 0.99,
    lr: float = 1e-3,
    baseline_mode: str = "mean",  # 'none' or 'mean'
    seed: int = 0,
):
    """
    Train a policy with REINFORCE algorithm.
    
    Args:
        env: ValorantEnv environment (already created with model, etc.)
        policy: PolicyNet to train
        episodes: Number of episodes to train for
        gamma: Discount factor for future rewards
        lr: Learning rate for optimizer
        baseline_mode: 'none' or 'mean' (mean reduces variance)
        seed: Random seed
    
    Returns:
        episode_returns: list of total rewards per episode
        policy: trained PolicyNet (same object, modified in place)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build optimizer for the policy
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    episode_returns = []

    for ep in range(episodes):
        # ---- Roll out one episode ----
        rewards, logps, total_return = run_one_episode(env, policy)

        # ---- Convert to tensors ----
        logps_t = torch.stack(logps)             # shape [T]
        G = discounted_returns(rewards, gamma)   # shape [T]

        # ---- Baseline (variance reduction) ----
        if baseline_mode == "none":
            advantages = G
        elif baseline_mode == "mean":
            advantages = G - G.mean()
        else:
            raise ValueError("baseline_mode must be 'none' or 'mean'.")

        # ---- REINFORCE loss: -sum_t A_t * log π(a_t|s_t) ----
        loss = -(logps_t * advantages).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_returns.append(total_return)

        if (ep + 1) % 10 == 0:
            recent_mean = np.mean(episode_returns[-10:])
            print(
                f"[{baseline_mode}] Episode {ep+1:4d} | "
                f"mean return (last 10) = {recent_mean:6.3f}"
            )

    return episode_returns, policy

# Training code will be added in next step
# You can train the agent by calling:
# returns, trained_policy = train_valorant_agent(env, policy, episodes=300, ...)

# %%
returns, trained_policy = train_valorant_agent(
    env, 
    policy, 
    episodes=300, 
    gamma=0.99, 
    lr=1e-3,
    baseline_mode="mean"
)