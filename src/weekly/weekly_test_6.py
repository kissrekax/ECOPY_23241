
import pandas
import pandas as pd
from typing import List, Dict
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

sp500 = pd.read_parquet('/Users/kissr/Downloads/Programming/Python/ECOPY_23241/data/sp500.parquet', engine='fastparquet')
ff_factors=pd.read_parquet('/Users/kissr/Downloads/Programming/Python/ECOPY_23241/data/ff_factors.parquet', engine='fastparquet')

#3
merged_df = pd.merge(sp500, ff_factors, on='Date', how='left')

#4
merged_df['Excess Return'] = merged_df['Monthly Return'] - merged_df['RF']

#5
merged_df = merged_df.sort_values(by=['Symbol', 'Date'])
merged_df['ex_ret_1'] = merged_df.groupby('Symbol')['Excess Return'].shift(-1)

#6
merged_df = merged_df.dropna(subset=['ex_ret_1'])
merged_df = merged_df.dropna(subset=['HML'])

#7
amazon_df = merged_df[merged_df['Symbol'] == 'AMZN']
amazon_df = amazon_df.drop(columns=['Symbol'])

import pandas as pd
import statsmodels as sm
class LinearRegressionSM:

    def __init__(self, left_hand_side, right_hand_side):

        self.left_hand_side = left_hand_side
        self.right_hand_side = right_hand_side
        self._model = None

    def fit(self):

        left_df = self.left_hand_side
        right_df = self.right_hand_side

        right_df = sm.add_constant(right_df)

        model = sm.OLS(left_df['Excess Return'], right_df).fit()

        self._model = model

        return model

    def get_params(self):

        if self._model is not None:
            beta_coefficients = self._model.params
            return pd.Series(beta_coefficients, name='Beta coefficients')
        else:
            raise ValueError("The model has not been fitted yet.")

    def get_pvalues(self):

        if self._model is not None:
            p_values = self._model.pvalues
            return pd.Series(p_values, name='P-values for the corresponding coefficients')
        else:
            raise ValueError("The model has not been fitted yet.")

    def get_wald_test_result(self, restriction_matrix):

        if self._model is not None:
            wald_test = self._model.wald_test(restriction_matrix)
            f_value = round(wald_test.statistic, 3)
            p_value = round(wald_test.pvalue, 3)
            result = f"F-value: {f_value}, p-value: {p_value}"
            return result
        else:
            raise ValueError("The model has not been fitted yet.")

    def get_model_goodness_values(self):

        if self._model is not None:
            adjusted_r_squared = round(self._model.rsquared_adj, 3)
            aic = round(self._model.aic, 3)
            bic = round(self._model.bic, 3)
            result = f"Adjusted R-squared: {adjusted_r_squared}, Akaike IC: {aic}, Bayes IC: {bic}"
            return result
        else:
            raise ValueError("The model has not been fitted yet.")

