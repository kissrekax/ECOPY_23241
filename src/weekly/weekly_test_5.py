import pandas as pd
import numpy as np
import random

# 1

file_to_load = '/Users/kissr/Downloads/Programming/Python/ECOPY_23241/data/chipotle.tsv'
food = pd.read_csv(file_to_load, sep='\t')


# 2

def change_price_to_float(input_df):
    cleaned_df = input_df.copy()
    cleaned_df['item_price'] = cleaned_df['item_price'].str.replace('$', '')
    cleaned_df['item_price'] = cleaned_df['item_price'].astype(float)

    return cleaned_df


food = change_price_to_float(food)


# 3

def number_of_observations(input_df):
    n_obs = len(input_df)
    return n_obs


# 4

def items_and_prices(input_df):
    items_prices = input_df[['item_name', 'item_price']]
    return items_prices


# 5

def sorted_by_price(input_df):
    sorted = items_and_prices(input_df).sort_values(by='item_price', ascending=False)
    return sorted


# 6

def avg_price(input_df):
    return input_df['item_price'].mean()


# 7

def unique_items_over_ten_dollars(input_df):
    above_10 = input_df[input_df['item_price'] > 10]
    un_above_10 = above_10.drop_duplicates(subset=('item_price', 'item_name', 'choice_description'))
    un_above_10 = un_above_10[['item_name', 'choice_description', 'item_price']]
    return un_above_10

def items_starting_with_s(input_df):
    s_df = input_df['item_name'].str.startswith('S')
    withs_df = input_df[s_df]
    withs_df = withs_df['item_name'].drop_duplicates()
    withs_df = pd.Series(withs_df)
    return withs_df

def first_three_columns(input_df):
    return input_df.iloc[:, :3]

def every_column_except_last_two(input_df):
    return input_df.iloc[:, :-2]

def sliced_view(input_df, columns_to_keep, column_to_filter, rows_to_keep):
    filtered_rows_df = input_df[input_df[column_to_filter].isin(rows_to_keep)]
    filtered_columns_df = filtered_rows_df[columns_to_keep]
    return filtered_columns_df

def generate_quartile(input_df):
    input_df['Quartile'] = pd.cut(
        input_df['item_price'],
        bins=[-1, 9.99, 19.99, 29.99, float('inf')],
        labels=['low-cost', 'medium-cost', 'high-cost', 'premium']
    )
    return input_df

def average_price_in_quartiles(input_df):
    quartiles_df = generate_quartile(input_df).groupby('Quartile').mean()
    return pd.Series(quartiles_df['item_price'])

def minmaxmean_price_in_quartile(input_df):
    mean_df = pd.DataFrame(generate_quartile(input_df).groupby('Quartile').mean())
    mean_df = mean_df[['item_price']]
    min_df = pd.DataFrame(generate_quartile(input_df).groupby('Quartile').min())
    min_df = min_df[['item_price']]
    max_df = pd.DataFrame(generate_quartile(input_df).groupby('Quartile').max())
    max_df = max_df[['item_price']]
    result_df = pd.merge(min_df, max_df, on='Quartile', how='left')
    result_df = pd.merge(result_df, mean_df, on='Quartile', how='left')
    result_df = result_df.rename(columns={'item_price_x': 'min', 'item_price_y': 'max', 'item_price': 'mean'})
    return result_df
