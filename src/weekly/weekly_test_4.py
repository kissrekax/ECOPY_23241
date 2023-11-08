#1
import pandas as pd

euro12 = pd.read_csv('/Users/molnartamas07/Documents/ECOPY_23241/data/Euro_2012_stats_TEAM.csv')

#2
def number_of_participants(input_df):
    num_participants = len(input_df['Team'].unique())
    return num_participants

#3
def goals(input_df):
    goals_df = input_df[['Team', 'Goals']]
    return goals_df

#4
def sorted_by_goal(input_df):
    sorted_df = input_df.sort_values(by='Goals', ascending=False)
    return sorted_df

#5
def avg_goal(input_df):
    avg_goals = input_df['Goals'].mean()
    return avg_goals

#6
def countries_over_five(input_df):
    selected_countries = input_df[input_df['Goals'] >= 6]
    return selected_countries

#7
def countries_starting_with_g(input_df):
    selected_countries = input_df[input_df['Team'].str.startswith('G')]
    return selected_countries

#8
def first_seven_columns(input_df):
    selected_columns = input_df.iloc[:, :7]
    return selected_columns

#9
def every_column_except_last_three(input_df):
    selected_columns = input_df.iloc[:, :-3]
    return selected_columns

#10
def sliced_view(input_df, columns_to_keep, column_to_filter, rows_to_keep):
    selected_columns_df = input_df[columns_to_keep]

    filtered_rows_df = selected_columns_df[input_df[column_to_filter].isin(rows_to_keep)]

    return filtered_rows_df

#11
def generate_quartile(input_df):
    input_df['Quartile'] = pd.cut(input_df['Goals'], [-1, 2, 4, 5, 12], labels=[4, 3, 2, 1], include_lowest=True)
    return input_df

#12
def average_yellow_in_quartiles(input_df):
    quartiles = pd.qcut(input_df['Goals'], q=4, labels=['Q4', 'Q3', 'Q2', 'Q1'])

    avg_yellow_in_quartiles = input_df.groupby(quartiles)['Yellow Cards'].mean().reset_index()
    avg_yellow_in_quartiles.columns = ['Quartile', 'Avg Yellow Cards']

    return avg_yellow_in_quartiles

#13
def minmax_block_in_quartile(input_df):
    quartiles = pd.qcut(input_df['Goals'], q=4, labels=['Q4', 'Q3', 'Q2', 'Q1'])

    minmax_block_in_quartile = input_df.groupby(quartiles)['Blocks'].agg([min, max]).reset_index()
    minmax_block_in_quartile.columns = ['Quartile', 'Min Blocks', 'Max Blocks']

    return minmax_block_in_quartile

#14
import matplotlib.pyplot as plt
def scatter_goals_shots(input_df):
    plt.figure(figsize=(8, 6))
    plt.scatter(input_df['Goals'], input_df['Shots on target'])

    plt.title('Goals and Shot on target')
    plt.xlabel('Goals')
    plt.ylabel('Shots on target')

    plt.show()

    return plt.figure

#15
def scatter_goals_shots_by_quartile(input_df):

    quartiles = pd.qcut(input_df['Goals'], q=4, labels=['Q4', 'Q3', 'Q2', 'Q1'])

    colors = {'Q1': 'red', 'Q2': 'blue', 'Q3': 'green', 'Q4': 'purple'}

    plt.figure(figsize=(8, 6))
    for quartile, color in colors.items():
        subset = input_df[quartiles == quartile]
        plt.scatter(subset['Goals'], subset['Shots on target'], label=quartile, c=color)

    plt.title('Goals and Shot on target')
    plt.xlabel('Goals')
    plt.ylabel('Shots on target')
    plt.legend(title='Quartiles')

    plt.show()

    return plt.figure

#16
import random
class ParetoDistribution:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, size):
        return [random.paretovariate(self.alpha) * self.beta for _ in range(size)]
def gen_pareto_mean_trajectories(pareto_distribution, number_of_trajectories, length_of_trajectory):
    random.seed(42)
    trajectories = []

    for _ in range(number_of_trajectories):
        trajectory = []
        cumulative_sum = 0

        for _ in range(length_of_trajectory):
            generated_value = pareto_distribution(1)[0]
            cumulative_sum += generated_value
            trajectory.append(cumulative_sum)

        trajectories.append(trajectory)

    return trajectories

if __name__ == "__main__":
    pareto = ParetoDistribution(1, 1)
    number_of_trajectories = 5
    length_of_trajectory = 10

    trajectories = gen_pareto_mean_trajectories(pareto, number_of_trajectories, length_of_trajectory)

    for i, trajectory in enumerate(trajectories):
        print(f'Trajectory {i + 1}: {trajectory}')