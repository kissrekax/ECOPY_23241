import pandas as pd
import statsmodels.api as sm
import numpy as np
import typing


class LinearRegressionSM:
    def __init__(self, left_hand_side, right_hand_side):

        self.left_hand_side = left_hand_side
        self.right_hand_side = right_hand_side
        self._model = None


    def fit(self):

        left_df = self.left_hand_side
        right_df = self.right_hand_side

        right_df = sm.add_constant(right_df)

        model = sm.OLS(left_df, right_df).fit()

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
            f_value = wald_test.statistic
            p_value = wald_test.pvalue
            result = f"F-value: {f_value:.3f}, p-value: {p_value:.3f}"
            return result
        else:
            raise ValueError("The model has not been fitted yet.")

    def get_model_goodness_values(self):
        if self._model is not None:
            adjusted_r_squared = self._model.rsquared_adj
            aic = self._model.aic
            bic = self._model.bic
            result = f"Adjusted R-squared: {adjusted_r_squared:.3f}, Akaike IC: {aic:.3f}, Bayes IC: {bic:.3f}"
            return result
        else:
            raise ValueError("The model has not been fitted yet.")

import pandas as pd
import typing
import numpy as np
from pathlib import Path


class LinearRegressionNP:
    def __init__(self, left_hand_side, right_hand_side):

        self.left_hand_side = left_hand_side
        self.right_hand_side = right_hand_side

    def fit(self):

        Y = self.left_hand_side
        X = self.right_hand_side

        n = X.shape[0]  # Number of rows in right_hand
        ones_column = np.ones((n, 1))
        X = np.hstack((ones_column, X))

        return

    def get_params(self):

        Y = self.left_hand_side
        X = self.right_hand_side

        n = X.shape[0]  # Number of rows in right_hand
        ones_column = np.ones((n, 1))
        X = np.hstack((ones_column, X))

        X_transpose = X.T
        X_transpose_X = X_transpose.dot(X)
        X_transpose_y = X_transpose.dot(Y)

        beta_coefficients = X_transpose_X/X_transpose_y
        return pd.Series(beta_coefficients, name='Beta coefficients')

    def get_pvalues(self):

        Y = self.left_hand_side
        X = self.right_hand_side

        n = X.shape[0]
        ones_column = np.ones((n, 1))
        X = np.hstack((ones_column, X))
        X_transpose = X.T
        X_transpose_X = X_transpose.dot(X)
        X_transpose_y = X_transpose.dot(Y)

        beta_coefficients = X_transpose_X/X_transpose_y

        # Compute the residuals
        residuals = Y - X.dot(beta_coefficients)

        # Degrees of freedom
        n = X.shape[0]  # Number of observations
        p = X.shape[1]  # Number of features including the intercept

        # Residual sum of squares
        rss = np.sum(residuals ** 2)

        # Compute the standard error of the residuals
        residual_std_error = np.sqrt(rss / (n - p))

        # Calculate the standard errors for each coefficient
        X_transpose_X_inv = np.linalg.inv(X.T.dot(X))  # You can use your method for matrix inversion
        std_errors = np.sqrt(np.diagonal(X_transpose_X_inv) * rss / (n - p))

        # Compute the t-statistic for each coefficient
        t_stats = beta / std_errors

        # Calculate the two-tailed p-values
        from scipy.stats import t
        p_values = (1 - t.cdf(np.abs(t_stats), df=n - p)) * 2
