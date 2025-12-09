# %% Download the dataset
import numpy as np
import kagglehub
import ast
import os
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import torch.nn as nn
import torch
from torch.distributions import Categorical
from typing import List

# Download latest version
path = kagglehub.dataset_download("visualize25/valorant-pro-matches-full-data")

print("Path to dataset files:", path)
# %% Read the dataset

# Use the path from kagglehub
db_path = os.path.join(path, "valorant.sqlite")
print("DB path:", db_path)

conn = sqlite3.connect(db_path)

games      = pd.read_sql_query("SELECT * FROM Games", conn)
rounds     = pd.read_sql_query("SELECT * FROM Game_Rounds", conn)
scoreboard = pd.read_sql_query("SELECT * FROM Game_Scoreboard", conn)
matches    = pd.read_sql_query("SELECT * FROM Matches", conn)

conn.close()

print("Games columns:", games.columns.tolist())
# %% Create model
games_model = games.copy()

# y = 1 if Team1 wins, else 0
games_model["y"] = (games_model["Winner"] == games_model["Team1"]).astype(int)
# %% Create feature
feature_cols = [
    "Team1_TotalRounds",
    "Team2_TotalRounds",
    "Team1_PistolWon",
    "Team2_PistolWon",
    "Team1_Eco",
    "Team2_Eco",
    "Team1_SemiBuy",
    "Team2_SemiBuy",
    "Team1_FullBuy",
    "Team2_FullBuy",
]

# Drop rows with missing values in these features
games_model = games_model.dropna(subset=feature_cols + ["y"])

X = games_model[feature_cols]
y = games_model["y"]
# %% Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# %% Train model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
# %% Run Logistic regression
logit = LogisticRegression(max_iter=2000)
logit.fit(X_train_scaled, y_train)
# %% Create prediction and measure accuracy
y_pred = logit.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

print("Logistic Regression Accuracy:", acc)
print(classification_report(y_test, y_pred))
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
      - If bank >= 20000: always FULL
      - Else if bank >= 10000: SEMI (still spending)
      - Else: ECO
    """
    if team2_bank >= 5 * 4200:
        action_id = 2  # full
    elif team2_bank >= 10000:
        action_id = 1  # semi
    else:
        action_id = 0  # eco

    return action_id, ACTION_ID_TO_NAME[action_id]

def team2_policy_middle(team2_bank, rng=None):
    """
    Middle / balanced strategy:
      - If bank >= 20000: [eco 0.3, semi 0.4, full 0.3]
      - If 10000 <= bank < 20000: [eco 0.5, semi 0.5, full 0]
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
      - If bank >= 25000: FULL
      - If 15000 <= bank < 25000: SEMI
      - If bank < 15000: ECO
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
    ):
        self.team1_bank_init = team1_bank_init
        self.team2_bank_init = team2_bank_init
        self.team2_policy    = team2_policy
        self.max_rounds      = max_rounds

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


    def step(self, action1):

        # -------------------------
        # 1. Team 2 action
        # -------------------------
        action2, action2_name = self.team2_policy(self.team2_bank, rng=self.rng)

        # -------------------------
        # 2. Determine round outcome
        # TODO: replace with logistic model later
        # -------------------------
        team1_wins = self.rng.random() < 0.5

        outcome1 = "win"  if team1_wins else "loss"
        outcome2 = "loss" if team1_wins else "win"

        # RL reward
        reward = 1 if outcome1 == "win" else 0

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

# %% Convet State to Tensor
import torch
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

# Test
# state = {
#     "round_number": 4,
#     "team1_bank": 8000,
#     "team2_bank": 12000,
#     "team1_losestreak": 1,
#     "team2_losestreak": 0,
#     "team1_score": 3,
#     "team2_score": 2,
# }
# print(state_to_tensor(state))
# %% Create Policy
class PolicyNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Input:
            state: shape [state_dim]

        Output:
            action probabilities: shape [action_dim], sum to 1
        """
        return self.net(state)

#state_dim = len(STATE_KEYS)
#action_dim = 3

#policy = PolicyNet(state_dim, action_dim)

#dummy_state = torch.tensor([1.0, 40000.0, 40000.0, 0.0, 1.0, 3.0, 4.0])

# print(policy(dummy_state))


# %% Run on episode
def run_one_episode(env: ValorantEnv, policy: PolicyNet):
    """
    Play one full game (episode) using the current policy.
    
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

if __name__ == "__main__":
    # 1. Build environment vs some fixed Team2 strategy
    env = ValorantEnv(team2_policy=team2_policy_aggressive)

    # 2. Build randomly initialized policy
    state_dim = len(STATE_KEYS)
    action_dim = 3
    policy = PolicyNet(state_dim, action_dim)

    # 3. Run one episode
    rewards, logps, total_return = run_one_episode(env, policy)

    print("Number of rounds played:", len(rewards))
    print("Total return (sum of rewards):", total_return)
    print("First few rewards:", rewards[:10])

# %% Discounted Return
def discounted_returns(rewards: List[float], gamma: float = 0.99) -> torch.Tensor:
    """
    Compute discounted returns G_t for a single episode:
        G_t = r_t + gamma * r_{t+1} + ...
    Input:
        rewards: list of rewards [r_0, ..., r_{T-1}]
    Output:
        torch.Tensor of shape [T]
    """
    G = []
    running = 0.0
    for r in reversed(rewards):
        running = r + gamma * running
        G.append(running)
    G.reverse()
    return torch.tensor(G, dtype=torch.float32)

# %% Train Model
def run_one_episode(env: ValorantEnv, policy: PolicyNet):
    """
    Play one full game (episode) using the current policy.
    
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

def train_valorant_agent(
    team2_policy_fn,
    episodes: int = 300,
    gamma: float = 0.99,
    lr: float = 1e-3,
    baseline_mode: str = "mean",  # 'none' or 'mean'
    seed: int = 0,
):
    """
    Train a policy with REINFORCE against a fixed Team2 policy.
    Returns:
        episode_returns: list of total rewards per episode
        policy: trained PolicyNet
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Create environment vs chosen Team2 strategy
    env = ValorantEnv(team2_policy=team2_policy_fn)

    # 2. Build policy & optimizer
    state_dim = len(STATE_KEYS)
    action_dim = 3
    policy = PolicyNet(state_dim, action_dim)
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

if __name__ == "__main__":
    EPISODES = 200
    GAMMA = 0.99
    LR = 1e-3

    # Train vs aggressive Team2 as a first test
    returns_aggr, policy_aggr = train_valorant_agent(
        team2_policy_fn=team2_policy_aggressive,
        episodes=EPISODES,
        gamma=GAMMA,
        lr=LR,
        baseline_mode="mean",
        seed=42,
    )

    print("\nTraining finished.")
    print("First 10 episode returns:", returns_aggr[:10])
    print("Last 10 episode returns:", returns_aggr[-10:])

# %%
