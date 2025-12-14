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
    
    # Create duplicates with Team1 and Team2 swapped
    print("\nCreating duplicates with Team1/Team2 swapped...")
    
    # Helper function to get team abbreviation (defined at module level for reuse)
    def get_team_abbrev(team_str):
        """Get abbreviation from team name (first letters of each word)"""
        if not team_str:
            return ''
        return ''.join([w[0] for w in str(team_str).split()]).upper()[:4]
    
    # Helper function to check if a winner string matches a team (defined at module level for reuse)
    def winner_matches_team(winner_str, team_str):
        """Check if winner string matches team name (using abbreviation logic)"""
        if not winner_str or not team_str:
            return False
        winner_upper = str(winner_str).strip().upper()
        team_upper = str(team_str).strip().upper()
        # Try exact match
        if winner_upper == team_upper:
            return True
        # Try abbreviation match
        team_abbrev = get_team_abbrev(team_str)
        if team_abbrev and (winner_upper.startswith(team_abbrev) or team_abbrev.startswith(winner_upper)):
            return True
        # Try substring match
        if winner_upper in team_upper or team_upper in winner_upper:
            return True
        return False
    
    # First, determine which team won each round BEFORE swapping
    def determine_original_winner(row):
        """Determine which team originally won: 'team1', 'team2', or None"""
        if 'RoundWinner' not in row or pd.isna(row.get('RoundWinner')):
            return None
        round_winner = str(row['RoundWinner']).strip()
        team1_str = str(row.get('Team1', '')).strip() if pd.notna(row.get('Team1')) else ''
        team2_str = str(row.get('Team2', '')).strip() if pd.notna(row.get('Team2')) else ''
        
        if team1_str and winner_matches_team(round_winner, team1_str):
            return 'team1'
        elif team2_str and winner_matches_team(round_winner, team2_str):
            return 'team2'
        return None
    
    # Add a column to track original winner
    cleaned_df['_OriginalWinner'] = cleaned_df.apply(determine_original_winner, axis=1)
    original_winner_counts = cleaned_df['_OriginalWinner'].value_counts()
    print(f"  Original winner distribution: {original_winner_counts.to_dict()}")
    none_count = cleaned_df['_OriginalWinner'].isna().sum()
    if none_count > 0:
        print(f"  Rows with undetermined winner: {none_count}")
        # Show sample of rows where winner couldn't be determined
        sample_none = cleaned_df[cleaned_df['_OriginalWinner'].isna()][['Team1', 'Team2', 'RoundWinner']].head(3)
        print(f"  Sample rows with None winner:")
        for idx, row in sample_none.iterrows():
            print(f"    Team1={row['Team1']}, Team2={row['Team2']}, RoundWinner={row['RoundWinner']}")
    
    # Calculate Team1 wins for original rows BEFORE swapping
    def calculate_team1_win_original(row):
        """Determine if Team1 won the round based on RoundWinner (for original rows)"""
        if pd.isna(row.get('RoundWinner', None)) or pd.isna(row.get('Team1', None)):
            return None
        round_winner = str(row['RoundWinner']).strip()
        team1_name = str(row.get('Team1', '')).strip() if pd.notna(row.get('Team1')) else ''
        
        if team1_name and winner_matches_team(round_winner, team1_name):
            return 1
        return 0
    
    cleaned_df['_Team1Wins'] = cleaned_df.apply(calculate_team1_win_original, axis=1)
    original_win_counts = cleaned_df['_Team1Wins'].value_counts()
    print(f"  Original Team1 wins distribution: {original_win_counts.to_dict()}")
    
    def swap_teams_in_row(row):
        """Create a new row with ALL Team1 and Team2 information swapped"""
        new_row = row.copy()
        
        # Swap team names (must be done first, before RoundWinner update)
        original_team1 = new_row.get('Team1')
        original_team2 = new_row.get('Team2')
        if 'Team1' in new_row and 'Team2' in new_row:
            new_row['Team1'] = original_team2
            new_row['Team2'] = original_team1
        
        # Swap team banks
        if 'Team1Bank' in new_row and 'Team2Bank' in new_row:
            new_row['Team1Bank'], new_row['Team2Bank'] = new_row['Team2Bank'], new_row['Team1Bank']
        
        # Swap team buy types
        if 'Team1BuyType' in new_row and 'Team2BuyType' in new_row:
            new_row['Team1BuyType'], new_row['Team2BuyType'] = new_row['Team2BuyType'], new_row['Team1BuyType']
        
        # Swap team scores
        if 'Team1Score' in new_row and 'Team2Score' in new_row:
            new_row['Team1Score'], new_row['Team2Score'] = new_row['Team2Score'], new_row['Team1Score']
        
        # Swap ScoreAfterRound (swap the scores in the string)
        if 'ScoreAfterRound' in new_row and pd.notna(new_row['ScoreAfterRound']):
            try:
                score_str = str(new_row['ScoreAfterRound'])
                if '-' in score_str:
                    parts = score_str.split('-')
                    if len(parts) == 2:
                        new_row['ScoreAfterRound'] = f"{parts[1].strip()}-{parts[0].strip()}"
            except:
                pass
        
        original_team1_str = str(original_team1).strip() if pd.notna(original_team1) else ''
        original_team2_str = str(original_team2).strip() if pd.notna(original_team2) else ''
        
        # Update RoundWinner based on original winner
        # If original Team1 won, after swap new Team1 (old Team2) should have lost
        # So RoundWinner should be the abbreviation of old Team1 (now Team2)
        original_winner = new_row.get('_OriginalWinner')
        if original_winner == 'team1' and original_team1_str:
            # Original Team1 won, so new Team1 (old Team2) should have lost
            # Set RoundWinner to abbreviation of old Team1 (now Team2), so it won't match new Team1
            new_row['RoundWinner'] = get_team_abbrev(original_team1_str)
        elif original_winner == 'team2' and original_team2_str:
            # Original Team2 won, so new Team1 (old Team1) should have won
            # Set RoundWinner to abbreviation of old Team2 (now Team1), so it will match new Team1
            new_row['RoundWinner'] = get_team_abbrev(original_team2_str)
        elif original_winner is None:
            # If we couldn't determine original winner, try to infer from RoundWinner
            # This handles edge cases where the matching logic failed
            if 'RoundWinner' in new_row and pd.notna(new_row['RoundWinner']) and original_team1_str and original_team2_str:
                round_winner_orig = str(new_row['RoundWinner']).strip()
                # Check which team it matched originally
                if winner_matches_team(round_winner_orig, original_team1_str):
                    # It matched original Team1, so after swap new Team1 should lose
                    new_row['RoundWinner'] = get_team_abbrev(original_team1_str)
                elif winner_matches_team(round_winner_orig, original_team2_str):
                    # It matched original Team2, so after swap new Team1 should win
                    new_row['RoundWinner'] = get_team_abbrev(original_team2_str)
                # If it doesn't match either, leave as-is (will be counted as Team1 loss)
        
        # Update Winner (game winner): same logic
        if 'Winner' in new_row and pd.notna(new_row['Winner']) and original_team1_str and original_team2_str:
            game_winner_orig = str(new_row['Winner']).strip()
            if winner_matches_team(game_winner_orig, original_team1_str):
                new_row['Winner'] = get_team_abbrev(original_team1_str)
            elif winner_matches_team(game_winner_orig, original_team2_str):
                new_row['Winner'] = get_team_abbrev(original_team2_str)
        
        # Flip _Team1Wins: if original Team1 won (1), new Team1 (old Team2) should lose (0)
        # If original Team1 lost (0), new Team1 (old Team2) should win (1)
        if '_Team1Wins' in new_row:
            original_win = new_row.get('_Team1Wins')
            if pd.notna(original_win):
                # Flip: 1 becomes 0, 0 becomes 1
                new_row['_Team1Wins'] = 1 - original_win
            # If original_win is None, leave it as None
        
        return new_row
    
    # Create swapped duplicates
    swapped_rows = []
    for idx, row in cleaned_df.iterrows():
        swapped_row = swap_teams_in_row(row)
        swapped_rows.append(swapped_row)
    
    # Convert to DataFrame and concatenate with original
    swapped_df = pd.DataFrame(swapped_rows)
    
    cleaned_df = pd.concat([cleaned_df, swapped_df], ignore_index=True)
    
    print(f"  Created {len(swapped_df)} duplicate rows with swapped teams")
    print(f"  Total rows after duplication: {len(cleaned_df)}")
    
    # Calculate and check final distribution of Team1 wins
    # Use the same matching logic as determine_original_winner for consistency
    def calculate_team1_win(row):
        """Determine if Team1 won the round based on RoundWinner"""
        if pd.isna(row.get('RoundWinner', None)) or pd.isna(row.get('Team1', None)):
            return None
        round_winner = str(row['RoundWinner']).strip()
        team1_name = str(row.get('Team1', '')).strip() if pd.notna(row.get('Team1')) else ''
        
        if team1_name and winner_matches_team(round_winner, team1_name):
            return 1
        return 0
    
    # Drop the helper column
    if '_OriginalWinner' in cleaned_df.columns:
        cleaned_df = cleaned_df.drop(columns=['_OriginalWinner'])
    
    # Check final distribution (using the _Team1Wins we already calculated and flipped)
    if '_Team1Wins' in cleaned_df.columns:
        print(f"  Final Team1 wins distribution (should be balanced ~50/50):")
        win_counts = cleaned_df['_Team1Wins'].value_counts().to_dict()
        print(f"    {win_counts}")
        none_wins = cleaned_df['_Team1Wins'].isna().sum()
        if none_wins > 0:
            print(f"    Rows with None (undetermined): {none_wins}")
        if 1 in win_counts and 0 in win_counts:
            ratio = win_counts[1] / (win_counts[1] + win_counts[0])
            print(f"    Team1 win rate: {ratio:.2%} (should be close to 50%)")
            if abs(ratio - 0.5) > 0.05:
                print(f"    WARNING: Distribution is not balanced! Difference from 50%: {abs(ratio - 0.5):.2%}")
                # Show sample of rows where Team1 won vs lost
                sample_wins = cleaned_df[cleaned_df['_Team1Wins'] == 1][['Team1', 'Team2', 'RoundWinner']].head(5)
                sample_losses = cleaned_df[cleaned_df['_Team1Wins'] == 0][['Team1', 'Team2', 'RoundWinner']].head(5)
                print(f"    Sample rows where Team1 won (first 5):")
                for idx, row in sample_wins.iterrows():
                    team1_abbrev = ''.join([w[0] for w in str(row['Team1']).split()]).upper()[:4] if pd.notna(row['Team1']) else 'N/A'
                    print(f"      Team1={row['Team1']} (abbrev: {team1_abbrev}), RoundWinner={row['RoundWinner']}")
                print(f"    Sample rows where Team1 lost (first 5):")
                for idx, row in sample_losses.iterrows():
                    team1_abbrev = ''.join([w[0] for w in str(row['Team1']).split()]).upper()[:4] if pd.notna(row['Team1']) else 'N/A'
                    print(f"      Team1={row['Team1']} (abbrev: {team1_abbrev}), RoundWinner={row['RoundWinner']}")
        else:
            print(f"    WARNING: Missing win counts (1 or 0 not found)")
    
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
