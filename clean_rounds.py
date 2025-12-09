"""
Data cleaning utilities for Valorant rounds data.
Extracts round history information into separate rows (one row per round).
"""

import pandas as pd
import ast
import numpy as np


def clean_rounds_data(rounds_df, games_df=None):
    """
    Clean the rounds dataframe by extracting RoundHistory into separate rows.
    Each round in a game becomes its own row.
    
    Args:
        rounds_df: DataFrame from Game_Rounds table with RoundHistory column
        games_df: Optional DataFrame from Games table to merge team names
    
    Returns:
        cleaned_df: DataFrame with one row per round, containing:
            - GameID (repeated for each round in that game)
            - RoundNumber (1, 2, 3, ...)
            - RoundWinner
            - ScoreAfterRound
            - WinType
            - Team1Bank
            - Team2Bank
            - Team1BuyType
            - Team2BuyType
            - Other original columns from rounds_df
    """
    print("\n" + "="*60)
    print("CLEANING ROUNDS DATA")
    print("="*60)
    print(f"Input: {len(rounds_df)} rows")
    
    # Parse RoundHistory column (stored as string dict)
    def parse_round_history(rh_val):
        """Parse round history string into dict"""
        if pd.isna(rh_val):
            return {}
        if isinstance(rh_val, dict):
            return rh_val
        if isinstance(rh_val, str):
            try:
                parsed = ast.literal_eval(rh_val)
                return parsed if isinstance(parsed, dict) else {}
            except (ValueError, SyntaxError) as e:
                return {}
        return {}
    
    print("\nParsing RoundHistory strings...")
    parsed_histories = rounds_df['RoundHistory'].apply(parse_round_history)
    valid_parses = parsed_histories.apply(lambda x: isinstance(x, dict) and len(x) > 0).sum()
    print(f"  Successfully parsed {valid_parses} out of {len(rounds_df)} round histories")
    
    # Create list to store all round rows
    all_round_rows = []
    
    print("\nExtracting rounds into separate rows...")
    
    # Iterate through each game
    for idx, row in rounds_df.iterrows():
        game_id = row.get('GameID')
        round_history = parsed_histories.iloc[idx]
        
        # Skip if no valid round history
        if not isinstance(round_history, dict) or len(round_history) == 0:
            continue
        
        # For each round in this game, create a new row
        for round_num, round_data in sorted(round_history.items()):
            if not isinstance(round_data, dict):
                continue
            
            # Start with original row data (excluding RoundHistory)
            new_row = row.drop('RoundHistory').to_dict()
            
            # Add round-specific information
            new_row['RoundNumber'] = round_num
            new_row['RoundWinner'] = round_data.get('RoundWinner', None)
            new_row['ScoreAfterRound'] = round_data.get('ScoreAfterRound', None)
            new_row['WinType'] = round_data.get('WinType', None)
            new_row['Team1Bank'] = round_data.get('Team1Bank', None)
            new_row['Team2Bank'] = round_data.get('Team2Bank', None)
            new_row['Team1BuyType'] = round_data.get('Team1BuyType', None)
            new_row['Team2BuyType'] = round_data.get('Team2BuyType', None)
            
            all_round_rows.append(new_row)
    
    # Create DataFrame from all round rows
    cleaned_df = pd.DataFrame(all_round_rows)
    
    print(f"  Created {len(cleaned_df)} rows (one per round)")
    print(f"  Average rounds per game: {len(cleaned_df) / valid_parses:.2f}")
    
    # Convert numeric columns
    print("\nConverting data types...")
    numeric_cols = ['RoundNumber', 'Team1Bank', 'Team2Bank']
    for col in numeric_cols:
        if col in cleaned_df.columns:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
    
    # Clean up string columns (remove extra spaces)
    string_cols = ['RoundWinner', 'Team1BuyType', 'Team2BuyType', 'WinType']
    for col in string_cols:
        if col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].astype(str).str.strip() if cleaned_df[col].dtype == 'object' else cleaned_df[col]
            # Replace 'nan' strings with actual NaN
            cleaned_df[col] = cleaned_df[col].replace('nan', np.nan)
    
    # Parse ScoreAfterRound into separate team scores if needed
    if 'ScoreAfterRound' in cleaned_df.columns:
        def parse_score(score_str):
            """Parse score string like '1-0' into team1_score, team2_score"""
            if pd.isna(score_str) or score_str is None:
                return None, None
            try:
                parts = str(score_str).split('-')
                if len(parts) == 2:
                    return int(parts[0]), int(parts[1])
            except:
                pass
            return None, None
        
        scores = cleaned_df['ScoreAfterRound'].apply(parse_score)
        cleaned_df['Team1Score'] = scores.apply(lambda x: x[0] if x[0] is not None else None)
        cleaned_df['Team2Score'] = scores.apply(lambda x: x[1] if x[1] is not None else None)
        cleaned_df['Team1Score'] = pd.to_numeric(cleaned_df['Team1Score'], errors='coerce')
        cleaned_df['Team2Score'] = pd.to_numeric(cleaned_df['Team2Score'], errors='coerce')
    
    # Merge with games data if provided
    if games_df is not None and 'GameID' in cleaned_df.columns and 'GameID' in games_df.columns:
        print("\nMerging with games data...")
        game_cols = ['GameID', 'Team1', 'Team2', 'Winner', 'Map']
        game_cols = [col for col in game_cols if col in games_df.columns]
        
        if game_cols:
            games_subset = games_df[game_cols]
            cleaned_df = cleaned_df.merge(games_subset, on='GameID', how='left', suffixes=('', '_game'))
            print(f"  Merged {len(game_cols)-1} columns from games table")
    
    # Sort by GameID and RoundNumber
    if 'GameID' in cleaned_df.columns and 'RoundNumber' in cleaned_df.columns:
        cleaned_df = cleaned_df.sort_values(['GameID', 'RoundNumber']).reset_index(drop=True)
    
    print(f"\nFinal output: {len(cleaned_df)} rows, {len(cleaned_df.columns)} columns")
    print(f"Columns: {cleaned_df.columns.tolist()}")
    
    print("\n" + "="*60)
    print("CLEANING COMPLETE")
    print("="*60)
    
    return cleaned_df


def show_sample_cleaned_data(cleaned_df, n_games=3):
    """
    Display sample of cleaned data for a few games.
    
    Args:
        cleaned_df: Cleaned DataFrame from clean_rounds_data()
        n_games: Number of games to show
    """
    if len(cleaned_df) == 0:
        print("No data to display")
        return
    
    print(f"\n=== Sample Cleaned Data (first {n_games} games) ===")
    
    if 'GameID' in cleaned_df.columns:
        unique_games = cleaned_df['GameID'].unique()[:n_games]
        for game_id in unique_games:
            game_rounds = cleaned_df[cleaned_df['GameID'] == game_id]
            print(f"\nGameID: {game_id} ({len(game_rounds)} rounds)")
            print(game_rounds[['RoundNumber', 'RoundWinner', 'ScoreAfterRound', 
                              'Team1Bank', 'Team2Bank', 'Team1BuyType', 'Team2BuyType']].to_string())
    else:
        print(cleaned_df.head(n_games * 13))  # Show roughly n games worth of rounds


if __name__ == "__main__":
    # Test the cleaning function
    from inspect_db import load_all_data
    
    print("Loading data...")
    games, rounds, scoreboard, matches = load_all_data()
    
    print("\nCleaning rounds data...")
    cleaned_rounds = clean_rounds_data(rounds, games)
    
    print("\nShowing sample...")
    show_sample_cleaned_data(cleaned_rounds, n_games=2)
