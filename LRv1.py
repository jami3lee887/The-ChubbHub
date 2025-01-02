# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 18:50:00 2025

Activern Logistic Regression Mode !!

@author: jsham
"""

import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def ML():
    # Example data (replace with your real data)
    data = pd.DataFrame({
        'team_avg_points': [30, 30, 25, 18, 24, 20, 20, 20, 20, 20],
        'opponent_avg_points': [22, 28, 27, 23, 20, 20, 20, 20, 20, 20],
        'over_under_line': [45.5, 50.5, 48.5, 42.5, 47.5, 40.5, 39.5, 40.5, 39.5, 38.5],
        'over_line_hit': [1, 1, 0, 0, 0, 0, 1, 0, 1, 1]  # 1 = over_line_hit, 0 = Not over_line_hit
    })
    
    # Features and target
    X = data[['team_avg_points', 'opponent_avg_points', 'over_under_line']]
    y = data['over_line_hit']
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    
    # Prepare the input as a DataFrame
    future_game = pd.DataFrame({
        'team_avg_points': [40],
        'opponent_avg_points': [20],
        'over_under_line': [58.5]
    })
    
    # Predict the probability that the OVER line will hit
    probability = model.predict_proba(future_game)[0][1]  # Get the probability for class 1 (over_line_hit)
    
    # Output the result
    print(f"\nThe model predicts a {probability:.2f} confidence that the OVER line will hit.")


def GetGameLog(team):
    
    # URL of the page to scrape
    url = 'https://www.teamrankings.com/nfl/team/detroit-lions/game-log'
    
    # Send a GET request to fetch the page content
    headers = {'User-Agent': 'Mozilla/5.0'}  # Set a user-agent to avoid blocking
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Raise an error for bad status codes
    
    # Use pandas to read the HTML tables from the page content
    tables = pd.read_html(response.text)
    
    # Display the number of tables found
    print(f"Number of tables found: {len(tables)}")
    
    # Assuming the game log is the first table; adjust the index if necessary
    game_log = tables[0]
    
    # Display the first few rows of the game log
    #print(game_log)
    return game_log


def CompileTeamLog():
    game_log = GetGameLog('detroit-lions')
    game_log[['Pts', 'Opp Pts']] = game_log['Score'].str.split('-', expand=True)

    # Step 1: Extract the "W" or "L" into a new column
    game_log['W/L'] = game_log['Score'].str[0]  # First character of the "Score" column (W or L)
    
    # Step 2: Remove the "W " or "L " prefix from the "Score" column
    game_log['Score'] = game_log['Score'].str[2:]  # Remove the first two characters
    
    # Step 3: Split the "Score" column into two separate columns
    game_log[['Pts', 'Opp Pts']] = game_log['Score'].str.split('-', expand=True)
    
    # Step 4: Convert the new columns to integers
    game_log['Pts'] = game_log['Pts'].astype(int)
    game_log['Opp Pts'] = game_log['Opp Pts'].astype(int)
    game_log = game_log.drop(columns=['Score'])
    
    print(game_log)
    

    
    

CompileTeamLog()