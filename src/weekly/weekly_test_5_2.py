import pandas as pd
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import random
import src.utils
import src.weekly

def change_price_to_float(input_df):
    df = input_df.copy()
    df['item_price'] = df['item_price'].str.replace('$', '').astype(float)
    return df

def number_of_observations(input_df):
    return len(input_df)

def items_and_prices(input_df):
    return input_df[['item_name', 'item_price']]

def sorted_by_price(input_df):
    return input_df.sort_values(by='item_price', ascending=False)

def avg_price(input_df):
    return input_df['item_price'].mean()

def unique_items_over_ten_dollars(input_df):
    unique_items = input_df.drop_duplicates(subset=['item_name', 'choice_description', 'item_price'])
    high_cost_items = unique_items[unique_items['item_price'] > 10]
    return high_cost_items[['item_name', 'choice_description', 'item_price']]


def items_starting_with_s(input_df):
    filtered_items = input_df[input_df['item_name'].str.startswith('S')]
    result = filtered_items['item_name']
    return result

def first_three_columns(input_df):
    return input_df.iloc[:, :3]

def every_column_except_last_two(input_df):
    return input_df.iloc[:, :-2]

def slaced_view(input_df, columns_to_keep, column_to_filter, rows_to_keep):
    if column_to_filter in input_df.columns:
        sliced_df = input_df[columns_to_keep]
        return sliced_df[sliced_df[column_to_filter].isin(rows_to_keep)]
    else:
        return None

def generate_quartile(input_df):
    input_df['Quartile'] = pd.cut(
        input_df['item_price'],
        bins=[-1, 9.99, 19.99, 29.99, float('inf')],
        labels=['low-cost', 'medium-cost', 'high-cost', 'premium']
    )
    return input_df

def average_price_in_quartiles(input_df):
    quartile_averages = input_df.groupby('Quartile')['item_price'].mean()
    return quartile_averages

def minmaxmean_price_in_quartile(input_df):
    quartile_stats = input_df.groupby('Quartile')['item_price'].agg(['min', 'max', 'mean'])
    return quartile_stats

import random
import numpy as np

def gen_uniform_mean_trajectories(distribution, number_of_trajectories, length_of_trajectory):
    random.seed(42)
    result = []
    for _ in range(number_of_trajectories):
        trajectory = []
        for _ in range(length_of_trajectory):
            trajectory.append(distribution.rvs(size=1).mean())
        result.append(trajectory)
    return result

from scipy.stats import logistic

def gen_logistic_mean_trajectories(distribution, number_of_trajectories, length_of_trajectory):
    np.random.seed(42)
    result = []
    for _ in range(number_of_trajectories):
        trajectory = []
        for _ in range(length_of_trajectory):
            sample = logistic.rvs(loc=1, scale=3.3, size=1)
            trajectory.append(np.mean(sample))
        result.append(trajectory)
    return result

from scipy.stats import laplace

def gen_laplace_mean_trajectories(distribution, number_of_trajectories, length_of_trajectory):
    np.random.seed(42)
    result = []
    for _ in range(number_of_trajectories):
        trajectory = []
        for _ in range(length_of_trajectory):
            sample = laplace.rvs(loc=1, scale=3.3, size=1)
            trajectory.append(np.mean(sample))
        result.append(trajectory)
    return result

from scipy.stats import cauchy

def gen_cauchy_mean_trajectories(distribution, number_of_trajectories, length_of_trajectory):
    np.random.seed(42)
    result = []
    for _ in range(number_of_trajectories):
        trajectory = []
        for _ in range(length_of_trajectory):
            sample = cauchy.rvs(loc=2, scale=4, size=1)
            trajectory.append(np.mean(sample))
        result.append(trajectory)
    return result

from scipy.stats import chi2

def gen_chi2_mean_trajectories(distribution, number_of_trajectories, length_of_trajectory):
    np.random.seed(42)
    result = []
    for _ in range(number_of_trajectories):
        trajectory = []
        for _ in range(length_of_trajectory):
            sample = chi2.rvs(3, size=1)
            trajectory.append(np.mean(sample))
        result.append(trajectory)
    return result