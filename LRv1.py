# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 18:50:00 2025

Activern Logistic Regression Mode !!

@author: jsham
"""

import numpy as np
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def GetGameLog(team):
    
    # URL of the page to scrape
    url = 'https://www.teamrankings.com/college-football/team/mississippi-rebels/game-log'
    
    # Send a GET request to fetch the page content
    headers = {'User-Agent': 'Mozilla/5.0'}  # Set a user-agent to avoid blocking
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Raise an error for bad status codes
    
    # Use pandas to read the HTML tables from the page content
    tables = pd.read_html(response.text)
    
    # Display the number of tables found
    #print(f"Number of tables found: {len(tables)}")
    
    # Assuming the game log is the first table; adjust the index if necessary
    game_log = tables[0]
    
    # Display the first few rows of the game log
    #print(game_log)
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
    #game_log = game_log.drop(columns=['Score'])
    return game_log

def GetOverUnder(team):
    # URL of the page to scrape
    url = 'https://www.teamrankings.com/college-football/team/mississippi-rebels/over-under-results'
    
    # Send a GET request to fetch the page content
    headers = {'User-Agent': 'Mozilla/5.0'}  # Set a user-agent to avoid blocking
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Raise an error for bad status codes
    
    # Use pandas to read the HTML tables from the page content
    tables = pd.read_html(response.text)
    
    # Display the number of tables found
    #print(f"Number of tables found: {len(tables)}")
    
    # Assuming the game log is the first table; adjust the index if necessary
    game_log = tables[0]
    return game_log
    
#merging all the dashboard 
def TeamData(team):
    global game_log
    game_log = GetGameLog(team)
    over_under_log = GetOverUnder(team)
    
    game_log = game_log.merge(over_under_log[['Score','Total','Result','Diff']],on="Score",how="left")
    
    game_log['Over_Under'] = np.where(game_log['Result'] == 'Over', 1, 0)
    
    return game_log
    #print(game_log)
    #print(over_under_log)
    
#where da machine learning happens mode
def ML(team):

    data = TeamData(team)
    print(data)
    
    # Features and target
    X = data[['Pts', 'Opp Pts', 'Total']]
    y = data['Over_Under']
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

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
        'Pts': [34],
        'Opp Pts': [26.9],
        'Total': [51.5]
    })
    
    # Predict the probability that the OVER line will hit
    probability = model.predict_proba(future_game)[0][1]  # Get the probability for class 1 (over_line_hit)
    
    # Output the result
    print(f"\nThe model predicts a {probability:.2f} confidence that the OVER line will hit.")
    
ML('mississipi-rebels')

