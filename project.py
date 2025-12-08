# %% Download the dataset
import kagglehub
import ast
import os
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

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
# %%
games_model = games.copy()

# y = 1 if Team1 wins, else 0
games_model["y"] = (games_model["Winner"] == games_model["Team1"]).astype(int)
# %%
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
# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# %%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
# %%
logit = LogisticRegression(max_iter=2000)
logit.fit(X_train_scaled, y_train)
# %%
y_pred = logit.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

print("Logistic Regression Accuracy:", acc)
print(classification_report(y_test, y_pred))
# %%
