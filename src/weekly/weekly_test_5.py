import pandas as pd
from pathlib import Path
import random
import numpy as np
import typing


file_to_load = Path.cwd().parent.joinpath('data').joinpath('chipotle.tsv')

food = pd.read_csv(file_to_load, sep='\t')


#2

def change_price_to_float(input_df):

    df_copy = input_df.copy()

    df_copy['item_price'] = df_copy['item_price'].str.replace('$', '').astype(float)

    return df_copy

#3

def number_of_observations(input_df):
    return len(input_df)

#4
def items_and_prices(input_df):
    result_df = input_df[['item_name', 'item_price']]

    return result_df

#5
def sorted_by_price(input_df):
    sorted_df = input_df.sort_values(by='item_price', ascending=False)

    return sorted_df

#6
def avg_price(input_df):
    average_price = input_df['item_price'].mean()

    return average_price

#7
def unique_items_over_ten_dollars(input_df):

    filtered_df = input_df[input_df['item_price'] > 10]

    unique_filtered_df = filtered_df.drop_duplicates(subset=['item_name', 'choice_description', 'item_price'])

    return unique_filtered_df

#8
def items_starting_with_s(input_df):
    filtered_df = input_df[input_df['item_name'].str.startswith('S')]

    result_df = filtered_df[['item_name']]

    return result_df

#9
def first_three_columns(input_df):

    result_df = input_df.iloc[:, :3]

    return result_df

#10
def every_column_except_last_two(input_df):
    result_df = input_df.iloc[:, :-2]

    return result_df

#11
def sliced_view(input_df, columns_to_keep, column_to_filter, rows_to_keep):

    filtered_columns_df = input_df[columns_to_keep]

    filtered_rows_df = filtered_columns_df[filtered_columns_df[column_to_filter].isin(rows_to_keep)]

    return filtered_rows_df

#12
def generate_quartile(input_df):

    input_df['Quartile'] = ''

    quartiles = [0, 10, 20, 30, float('inf')]
    quartile_labels = ['low-cost', 'medium-cost', 'high-cost', 'premium']

    input_df['Quartile'] = pd.cut(input_df['item_price'], bins=quartiles, labels=quartile_labels, right=False)

    return input_df

#13
def average_price_in_quartiles(input_df):

    avg_prices_df = input_df.groupby('Quartile')['item_price'].mean().reset_index()

    return avg_prices_df

#14
def minmaxmean_price_in_quartile(input_df):

    minmaxmean_prices_df = input_df.groupby('Quartile')['item_price'].agg(['min', 'max', 'mean']).reset_index()

    return minmaxmean_prices_df

#15
def gen_uniform_mean_trajectories(distribution, number_of_trajectories, length_of_trajectory):
    random.seed(42)
    np.random.seed(42)

    trajectories = []

    for _ in range(number_of_trajectories):

        random_numbers = distribution.random(length_of_trajectory)

        cumulative_mean = np.cumsum(random_numbers) / np.arange(1, length_of_trajectory + 1)

        trajectories.append(cumulative_mean.tolist())

    return trajectories


#16
def gen_logistic_mean_trajectories(distribution, number_of_trajectories, length_of_trajectory):
    # Állítsuk be a seed-et 42-re
    random.seed(42)
    np.random.seed(42)

    trajectories = []

    for _ in range(number_of_trajectories):
        # Generáljunk length_of_trajectory darab véletlen számot a megadott eloszlás alapján
        random_numbers = distribution.random(length_of_trajectory)

        # Számoljuk ki a kumulatív átlagot
        cumulative_mean = np.cumsum(random_numbers) / np.arange(1, length_of_trajectory + 1)

        trajectories.append(cumulative_mean.tolist())

    return trajectories

#17
def gen_laplace_mean_trajectories(distribution, number_of_trajectories, length_of_trajectory):
    random.seed(42)
    np.random.seed(42)

    trajectories = []

    for _ in range(number_of_trajectories):

        random_numbers = distribution.random(length_of_trajectory)


        cumulative_mean = np.cumsum(random_numbers) / np.arange(1, length_of_trajectory + 1)

        trajectories.append(cumulative_mean.tolist())

    return trajectories











