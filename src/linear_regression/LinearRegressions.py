import pandas as pd
import statsmodels.api as sm
import numpy as np
import scipy


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
        beta_coefficients = self._model.params
        return pd.Series(beta_coefficients, name='Beta coefficients')

    def get_pvalues(self):
        p_values = self._model.pvalues
        return pd.Series(p_values, name='P-values for the corresponding coefficients')

    def get_wald_test_result(self, restriction_matrix):

        if self._model is not None:
            wald_test = self._model.wald_test(restriction_matrix)
            f_value = float(wald_test.statistic)
            p_value = float(wald_test.pvalue)
            result = f"F-value: {f_value:.3f}, p-value: {p_value:.3f}"
            return result
        else:
            raise ValueError("The model has not been fitted yet.")

    def get_model_goodness_values(self):

        if self._model is not None:
            adjusted_r_squared = float(self._model.rsquared_adj)
            aic = float(self._model.aic)
            bic = float(self._model.bic)
            result = f"Adjusted R-squared: {adjusted_r_squared:.3f}, Akaike IC: {aic:.3f}, Bayes IC: {bic:.3f}"
            return result
        else:
            raise ValueError("The model has not been fitted yet.")


class LinearRegressionNP:
    def __init__(self, left_hand_side, right_hand_side):
        self.left_hand_side = left_hand_side
        self.right_hand_side = right_hand_side
        self.right_hand_side = sm.add_constant(self.right_hand_side)
        self._model = None

    def fit(self):
        x = np.column_stack((np.ones(len(self.right_hand_side)), self.right_hand_side))
        y = self.left_hand_side.values
        beta = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
        alpha = beta[0]
        beta = beta[1:]
        self.alpha = alpha
        self.beta = beta

        return

    def get_params(self):
        Y = self.left_hand_side
        X = self.right_hand_side
        self.XtX = np.dot(np.transpose(self.right_hand_side), self.right_hand_side)
        self.XtX_inv = np.linalg.inv(self.XtX)
        self.Xty = np.dot(np.transpose(self.right_hand_side), self.left_hand_side)
        self.betas = np.dot(self.XtX_inv, self.Xty)
        return pd.Series(self.betas, name='Beta coefficients')

    def get_pvalues(self):
        self.residuals = self.left_hand_side - np.dot(self.right_hand_side, self.betas)
        self.n = len(self.left_hand_side)
        self.K = len(self.right_hand_side.columns)
        self.df = self.n - self.K
        self.variance = np.dot(np.transpose(self.residuals), self.residuals) / self.df
        self.stderror = np.sqrt(self.variance * np.diag(self.XtX_inv))  # * vs .dot
        self.t_stat = np.divide(self.betas, self.stderror)
        self.abs_t_stats = abs(self.t_stat)
        term = np.minimum(scipy.stats.t.cdf(self.t_stat, self.df), 1 - scipy.stats.t.cdf(self.t_stat, self.df))
        p_values = (term) * 2
        return pd.Series(p_values, name='P-values for the corresponding coefficients')

    def get_wald_test_result(self, restr_matrix):
        r = np.transpose(np.zeros((len(restr_matrix))))
        term_1 = np.dot(restr_matrix, self.betas) - r
        term_2 = np.dot(np.dot(restr_matrix, self.XtX_inv), np.transpose(restr_matrix))
        f_stat = (np.dot(np.transpose(term_1), np.dot(np.linalg.inv(term_2), term_1)) / len(
            restr_matrix)) / self.variance
        p_value = (1 - scipy.stats.f.cdf(f_stat, len(restr_matrix), self.df))
        f_stat.astype(float)
        p_value.astype(float)
        return f'Wald: {round(f_stat, 3)}, p-value: {round(p_value, 3)}'

    def get_model_goodness_values(self):
        y_demean = self.left_hand_side - sum(self.left_hand_side) / len(self.left_hand_side)
        SST = np.dot(np.transpose(y_demean), y_demean)
        SSE = np.dot(np.transpose(self.residuals), self.residuals)
        r2 = round(1 - SSE / SST, 3)
        adj_r2 = round(1 - (self.n - 1) / (self.n - self.p) * (1 - r2), 3)
        return f'Centered R-squared: {r2:.3f}, Adjusted R-squared: {adj_r2:.3f}'
