import pandas as pd
import statsmodels.api as sm
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

