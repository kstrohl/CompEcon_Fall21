### Kaleigh Strohl | Problem Set 5: Matching

import numpy as np
import pandas as pd
import geopy
from geopy import Point
import scipy.optimize as opt

df = pd.read_csv("radio_merger_data.csv")
df.head(n=10)

# Re-scale the variables as suggested
df['new_pop'] = df['population_target']/1000000
df['new_prices'] = df['price']/1000000

# create distance variable
df['buyer_loc'] = df.apply(lambda row: Point(latitude=row['buyer_lat'], longitude=row['buyer_long']), axis=1)
df['target_loc'] = df.apply(lambda row: Point(latitude=row['target_lat'], longitude=row['target_long']), axis=1)
df['distance'] = df.apply(lambda row: geopy.distance.geodesic(row['buyer_loc'], row['target_loc']).miles, axis=1)

# create counterfactuals and create distance variable for counterfacts 
buyer = ['year', 'buyer_id', 'buyer_lat', 'buyer_long', 'buyer_loc', 'num_stations_buyer', 'corp_owner_buyer']
target = ['target_id', 'target_lat', 'target_long', 'target_loc', 'new_prices', 'hhi_target', 'new_pop']

years = [(df.loc[df['year'] == 2007]), (df.loc[df['year'] == 2008])]

counter = [x[buyer].iloc[i].values.tolist() + x[target].iloc[j].values.tolist()
             for x in years for i in range(len(x) - 1)
             for j in range(i + 1, len(x))]
counterfacts = pd.DataFrame(counter, columns = buyer + target)

counterfacts['distance'] = counterfacts.apply(lambda row: geopy.distance.geodesic(row['buyer_loc'], row['target_loc']).miles, axis=1)

# payoff function: f_m(b,t) = x1_bm*y1_tm + alpha*x2_bm*y1_tm + beta*distance_btm + epsilon_btm
def payoff_fn(dataframe, params):
    '''
    Args:
        dataframe: dataframe of interest for payoff calculation
        params: initial guesses for model parameters
    Returns:
        payoff: the payoff to the merger between radio station b and target t in market m
    '''
    alpha, beta = params

    payoff = dataframe['num_stations_buyer'] * dataframe['new_pop'] + alpha * dataframe['corp_owner_buyer'] * dataframe['new_pop'] + beta * dataframe['distance']
    return(payoff)

params = (0.5, 0.5)

# Calculate observed payoffs
df['payoff'] = payoff_fn(dataframe=df, params=params)

# Calculate counterfactual payoffs
counterfacts['payoff'] = payoff_fn(dataframe=counterfacts, params=params)

# Create MSE objective function
# Payoff function: f_m(b,t) = x1_bm*y1_tm + alpha*x2_bm*y1_tm + beta*distance_btm + epsilon_btm

def objective_1(params): 
    '''
    Args:
        params: initial guesses for model parameters (alpha and beta)
    Returns:
        payoff: the payoff to the merger between radio station b and target t in market m
        score: maximum score estimation
    '''
    alpha, beta = params
    
    count = 0
    score = 0
    
    
    def payoff_fn(dataframe, index):
        '''
        Args:
            dataframe: dataframe of interest for payoff calculation
            index: identification numbers for radio station b and target t
        Returns:
            payoff: the payoff to the merger between radio station b and target t in market m
        '''
        b, t = index

        payoff = dataframe.num_stations_buyer[b] * dataframe.new_pop[t] + alpha * dataframe.corp_owner_buyer[b] * dataframe.new_pop[t] + beta * dataframe.distance[(dataframe['buyer_id']==1) & (dataframe['target_id']==1)]
        return(payoff)
    
    
    for x in years:
        for i in range(len(df.buyer_id)):
            for j in range(i+1, len(df.buyer_id)):
                count += 1
                if payoff_fn(dataframe=df, index=(i, i)) + payoff_fn(dataframe=df, index=(j, j)) > payoff_fn(dataframe=counterfacts, index=(i, j)) + payoff_fn(dataframe=counterfacts, index=(j, i)): 
                    score += 1
    return(x, score)

guess = (0.5, 0.5)
result = opt.minimize(objective_1, guess, method='Nelder-Mead', tol=1e-15)
print(result)