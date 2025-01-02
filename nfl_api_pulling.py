# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 22:25:02 2024

@author: jsham
"""

#print("hello world")

import pandas as pd
#import datetime
#import nflscraPy
#import numpy as np
#from scipy.stats import poisson
#from scipy.stats import norm

# season_gamelogs = nflscraPy._gamelogs(
#     2024
# )

# #gamelog_metadata = nflscraPy._gamelog_metadata(
# #    'https://www.pro-football-reference.com/boxscores/202212180jax.htm'
# #)

# five_thirty_eight = nflscraPy._five_thirty_eight()

# from sportsreference.nfl.teams import Teams

# teams = Teams(year=2024)
# for team in teams:
#     print(team.name, team.defensive_simple_rating_system)

import requests

# url = 'https://api.sportsdata.io/v3/nfl/scores/json/Standings/2024REG'
# headers = {
#     'Ocp-Apim-Subscription-Key': '03aea0cd0d7447e697111d46afb8cb91'
# }

# response = requests.get(url, headers=headers)
# data = response.json()  # This will be a list of standings data

url = 'https://api.sportsdata.io/v3/nfl/scores/json/TeamTrends'
headers = {
    'Ocp-Apim-Subscription-Key': '03aea0cd0d7447e697111d46afb8cb91'
}

response1 = requests.get(url, headers=headers)
data1 = response1.json()  # This will be a list of standings data
