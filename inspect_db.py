"""
Database inspection utilities for Valorant dataset.
Contains functions to load and inspect data from the SQLite database.
"""

import kagglehub
import os
import sqlite3
import pandas as pd


def load_all_data():
    """
    Load all data from the Valorant SQLite database.
    
    Returns:
        tuple: (games, rounds, scoreboard, matches) - DataFrames containing all tables
    """
    # Download latest version
    path = kagglehub.dataset_download("visualize25/valorant-pro-matches-full-data")
    print("Path to dataset files:", path)
    
    # Use the path from kagglehub
    db_path = os.path.join(path, "valorant.sqlite")
    print("DB path:", db_path)
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    
    # Load all tables
    games = pd.read_sql_query("SELECT * FROM Games", conn)
    rounds = pd.read_sql_query("SELECT * FROM Game_Rounds", conn)
    scoreboard = pd.read_sql_query("SELECT * FROM Game_Scoreboard", conn)
    matches = pd.read_sql_query("SELECT * FROM Matches", conn)
    
    conn.close()
    
    return games, rounds, scoreboard, matches


def inspect_database(games=None, rounds=None, scoreboard=None, matches=None):
    """
    Inspect and display information about all database tables.
    
    Args:
        games: DataFrame for Games table (optional, will load if not provided)
        rounds: DataFrame for Game_Rounds table (optional, will load if not provided)
        scoreboard: DataFrame for Game_Scoreboard table (optional, will load if not provided)
        matches: DataFrame for Matches table (optional, will load if not provided)
    """
    print("\n" + "="*60)
    print("INSPECTING DATABASE")
    print("="*60)
    
    # Load data if not provided
    if games is None or rounds is None or scoreboard is None or matches is None:
        print("\nLoading data from database...")
        games, rounds, scoreboard, matches = load_all_data()
    
    print("\n=== GAMES TABLE ===")
    print(f"Shape: {games.shape} (rows, columns)")
    print(f"Columns ({len(games.columns)}): {games.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(games.head())
    print(f"\nData types:")
    print(games.dtypes)
    
    print("\n=== ROUNDS TABLE ===")
    print(f"Shape: {rounds.shape} (rows, columns)")
    print(f"Columns ({len(rounds.columns)}): {rounds.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(rounds.head())
    print(f"\nData types:")
    print(rounds.dtypes)
    
    print("\n=== SCOREBOARD TABLE ===")
    print(f"Shape: {scoreboard.shape} (rows, columns)")
    print(f"Columns ({len(scoreboard.columns)}): {scoreboard.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(scoreboard.head())
    print(f"\nData types:")
    print(scoreboard.dtypes)
    
    print("\n=== MATCHES TABLE ===")
    print(f"Shape: {matches.shape} (rows, columns)")
    print(f"Columns ({len(matches.columns)}): {matches.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(matches.head())
    print(f"\nData types:")
    print(matches.dtypes)
    
    print("\n" + "="*60)
    print("INSPECTION COMPLETE")
    print("="*60)
    
    return games, rounds, scoreboard, matches


if __name__ == "__main__":
    # If run directly, just inspect the database
    inspect_database()
