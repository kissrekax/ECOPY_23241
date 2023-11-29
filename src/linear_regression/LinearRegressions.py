
import pandas as pd
import statsmodels.api as sm
import numpy as np
import scipy
from src.utils.distributions import NormalDistribution


class LinearRegressionSM:
    def __init__(self, left_hand_side, right_hand_side):
        self.left_hand_side = left_hand_side
        self.right_hand_side = right_hand_side
        self.right_hand_side = sm.add_constant(self.right_hand_side)
        self._model = None

    def fit(self):
        model = sm.OLS(self.left_hand_side, self.right_hand_side).fit()
        self._model = model

        return model

    def get_params(self):
        beta_coefficients = self._model.params
        return pd.Series(beta_coefficients, name='Beta coefficients')

    def get_pvalues(self):
        p_values = self._model.pvalues
        return pd.Series(p_values, name='P-values for the corresponding coefficients')

    def get_wald_test_result(self, restriction_matrix):
        wald_test = self._model.wald_test(restriction_matrix)
        f_value = float(wald_test.statistic)
        p_value = float(wald_test.pvalue)
        result = f"F-value: {f_value:.3}, p-value: {p_value:.3}"
        return result


    def get_model_goodness_values(self):
        adjusted_r_squared = float(self._model.rsquared_adj)
        aic = float(self._model.aic)
        bic = float(self._model.bic)
        result = f"Adjusted R-squared: {adjusted_r_squared:.3}, Akaike IC: {aic:.3}, Bayes IC: {bic:.3}"
        return result


class LinearRegressionNP:
    def __init__(self, left_hand_side, right_hand_side):
        self.left_hand_side = left_hand_side
        self.right_hand_side = right_hand_side
        self.right_hand_side.insert(0, 'alfa', 1)
        self._model = None

    def fit(self):
        x = self.right_hand_side
        y = self.left_hand_side.values
        self.XtX = x.T@x
        self.XtX_inv = np.linalg.inv(self.XtX)
        self.Xty = x.T@y
        self.betas = self.XtX_inv@self.Xty

        return

    def get_params(self):
        return pd.Series(self.betas, name='Beta coefficients')

    def get_paired_se_and_percentile_ci(self, number_of_bootstrap_samples, alpha, random_seed):
        beta_samples = []
        np.random.seed(random_seed)

        for _ in range(number_of_bootstrap_samples):
            # Generate a bootstrap sample
            idx = np.random.choice(len(self.left_hand_side), size=len(self.left_hand_side), replace=True)
            x_bootstrap = self.right_hand_side.iloc[idx]
            y_bootstrap = self.left_hand_side.iloc[idx].values

            # Fit the model on the bootstrap sample
            XtX_bootstrap = x_bootstrap.T @ x_bootstrap
            XtX_inv_bootstrap = np.linalg.inv(XtX_bootstrap)
            Xty_bootstrap = x_bootstrap.T @ y_bootstrap
            betas_bootstrap = XtX_inv_bootstrap @ Xty_bootstrap
            beta_first = betas_bootstrap[1]
            beta_samples.append(beta_first)

        beta_samples = np.array(beta_samples)

        variance_bootstrap = (1/number_of_bootstrap_samples)*np.sum((beta_samples - beta_samples.mean())**2)
        bse = np.sqrt(variance_bootstrap)

        # Compute confidence intervals (adjust percentiles as needed)
        #z = scipy.stats.norm.ppf(1 - alpha / 2)
        #lb = self.betas[1] - z*bse
        #ub = self.betas[1] + z*bse
        lb = np.quantile(beta_samples, alpha/2, axis=0)
        ub = np.quantile(beta_samples, 1-alpha/2, axis=0)

        return f'Paired Bootstraped SE: {bse:.3f}, CI: [{lb:.3f}, {ub:.3f}]'

    def get_wild_se_and_normal_ci(self, number_of_bootstrap_samples, alpha, random_seed):
        beta_samples = []
        np.random.seed(random_seed)

        for _ in range(number_of_bootstrap_samples):
            # Generate a bootstrap sample
            idx = np.random.choice(len(self.left_hand_side), size=len(self.left_hand_side), replace=True)
            x_bootstrap = self.right_hand_side.iloc[idx]
            V = np.random.normal(0, 1, len(self.left_hand_side))
            y_bootstrap = x_bootstrap@self.betas + V@self.residuals.iloc[idx]

            # Fit the model on the bootstrap sample
            XtX_bootstrap = x_bootstrap.T @ x_bootstrap
            XtX_inv_bootstrap = np.linalg.inv(XtX_bootstrap)
            Xty_bootstrap = x_bootstrap.T @ y_bootstrap
            betas_bootstrap = XtX_inv_bootstrap @ Xty_bootstrap
            beta_first = betas_bootstrap[1]
            beta_samples.append(beta_first)

        beta_samples = np.array(beta_samples)

        variance_bootstrap = (1 / number_of_bootstrap_samples) * np.sum((beta_samples - beta_samples.mean()) ** 2)
        bse = np.sqrt(variance_bootstrap)

        # Compute confidence intervals (adjust percentiles as needed)
        z = scipy.stats.norm.ppf(1 - alpha / 2)
        lb = self.betas[1] - z*bse
        ub = self.betas[1] + z*bse
        #lb = np.quantile(beta_samples, alpha / 2, axis=0)
        #ub = np.quantile(beta_samples, 1 - alpha / 2, axis=0)

        return f'Wild Bootstraped SE: {bse:.3f}, CI: [{lb:.3f}, {ub:.3f}]'

    def get_pvalues(self):
        self.residuals = self.left_hand_side - np.dot(self.right_hand_side, self.betas)
        self.n = len(self.left_hand_side)
        self.K = len(self.right_hand_side.columns)
        self.df = self.n - self.K
        self.variance = self.residuals.T@self.residuals / self.df
        self.stderror = np.sqrt(self.variance * np.diag(self.XtX_inv))
        self.t_stat = np.divide(self.betas, self.stderror)
        term = np.minimum(scipy.stats.t.cdf(self.t_stat, self.df), 1 - scipy.stats.t.cdf(self.t_stat, self.df))
        p_values = term * 2
        return pd.Series(p_values, name='P-values for the corresponding coefficients')

    def get_wald_test_result(self, restr_matrix):
        term_1 = restr_matrix@self.betas
        term_2 = np.linalg.inv(restr_matrix@self.XtX_inv@np.array(restr_matrix).T)
        m = len(restr_matrix)
        f_stat = (term_1.T@term_2@term_1/m)/self.variance
        p_value = 1 - scipy.stats.f.cdf(f_stat, m, self.df)
        return f'Wald: {f_stat:.3f}, p-value: {p_value:.3f}'

    def get_model_goodness_values(self):
        y_demean = self.left_hand_side - self.left_hand_side.mean()
        SST = y_demean.T@y_demean
        SSE = self.residuals.T@self.residuals
        r2 = 1 - SSE / SST
        adj_r2 = 1 - (self.n - 1) / self.df * (1 - r2)
        return f'Centered R-squared: {r2:.3f}, Adjusted R-squared: {adj_r2:.3f}'


class LinearRegressionGLS:

    def __init__(self, left_hand_side, right_hand_side):
        self.left_hand_side = left_hand_side
        self.right_hand_side = right_hand_side
        self.right_hand_side.insert(0, 'alfa', 1)
        self.left_hand_side = self.left_hand_side.values
        self.right_hand_side = self.right_hand_side.values
        self._model = None

    def fit(self):
        Y_OLS = self.left_hand_side
        X_OLS = self.right_hand_side
        self.XtX_OLS = X_OLS.T@X_OLS
        self.XtX_inv_OLS = np.linalg.inv(self.XtX_OLS)
        self.Xty_OLS = X_OLS.T@Y_OLS
        self.betas_OLS = self.XtX_inv_OLS@self.Xty_OLS
        self.residuals_OLS = Y_OLS - X_OLS@self.betas_OLS
        Y_new = np.log(self.residuals_OLS**2)
        X_new = self.right_hand_side
        self.XtX_feas = X_new.T@X_new
        self.XtX_inv_feas = np.linalg.inv(self.XtX_feas)
        self.Xty_feas = X_new.T@Y_new
        self.betas_feas = self.XtX_inv_feas@self.Xty_feas
        pred_Y = X_OLS@self.betas_feas
        pred_Y = np.sqrt(np.exp(pred_Y))
        pred_Y = pred_Y**-1
        self.V_inv = np.diag(pred_Y)

        return

    def get_params(self):
        self.XtX_gls = self.right_hand_side.T@self.V_inv@self.right_hand_side
        self.XtX_inv_gls = np.linalg.inv(self.XtX_gls)
        self.Xty_gls = self.right_hand_side.T@self.V_inv@self.left_hand_side
        self.betas_gls = self.XtX_inv_gls@self.Xty_gls
        return pd.Series(self.betas_gls, name='Beta coefficients')

    def get_pvalues(self):
        self.residuals_gls = self.left_hand_side - self.right_hand_side@self.betas_gls
        self.n = self.right_hand_side.shape[0]
        self.K = self.right_hand_side.shape[1]
        self.df = self.n - self.K
        self.variance_gls = self.residuals_gls.T@self.residuals_gls / self.df
        self.stderror_gls = np.sqrt(self.variance_gls * np.diag(self.XtX_inv_gls))
        self.t_stat_gls = np.divide(self.betas_gls, self.stderror_gls)
        term = np.minimum(scipy.stats.t.cdf(self.t_stat_gls, self.df), 1 - scipy.stats.t.cdf(self.t_stat_gls, self.df))
        p_values = (term) * 2
        return pd.Series(p_values, name='P-values for the corresponding coefficients')

    def get_wald_test_result(self, restr_matrix):
        term_1 = restr_matrix@self.betas_gls
        term_2 = np.linalg.inv(restr_matrix@self.XtX_inv_gls@np.array(restr_matrix).T)
        m = len(restr_matrix)
        f_stat = (term_1.T@term_2@term_1/m)/self.variance_gls
        p_value = 1 - scipy.stats.f.cdf(f_stat, m, self.df)
        return f'Wald: {f_stat:.3f}, p-value: {p_value:.3f}'

    def get_model_goodness_values(self):
        Y = self.left_hand_side
        X = self.right_hand_side
        SST = Y.T@self.V_inv@Y
        SSE = Y.T @ self.V_inv @ X @ np.linalg.inv(X.T @ self.V_inv @ X) @ X.T @ self.V_inv @ Y
        r2 = 1 - SSE / SST
        adj_r2 = 1 - ((self.n - 1) / self.df * (1 - r2))
        return f'Centered R-squared: {r2:.3f}, Adjusted R-squared: {adj_r2:.3f}'


class LinearRegressionML:
    def __init__(self, left_hand_side, right_hand_side):
        self.left_hand_side = left_hand_side
        self.right_hand_side = right_hand_side
        self.right_hand_side.insert(0, 'alfa', 1)
        self.left_hand_side = self.left_hand_side.values
        self.right_hand_side = self.right_hand_side.values
        self.params = None

    def log_likelihood(params, X, y):
        predicted = X @ params
        log_likelihood = -0.5 * (np.log(2 * np.pi * np.var(y)) + np.sum((y - predicted) ** 2) / np.var(y))
        return -log_likelihood

    def fit(self):
        def log_likelihood(params, X, y):
            sigma = params[-1]
            predicted = X @ params[0:-1]
            log_likelihood = -0.5 * (np.log(2 * np.pi * sigma) + np.sum((y - predicted) ** 2) / sigma)
            return -log_likelihood

        initial_params = np.zeros(self.right_hand_side.shape[1]+1) + 0.1

        self.result = scipy.optimize.minimize(log_likelihood, initial_params,args=(self.right_hand_side, self.left_hand_side), method='L-BFGS-B')
        self.params = self.result.x
        self.betas = self.params[0:-1]
        self.hess_inv = self.result.hess_inv

    def get_params(self):
        return pd.Series(self.betas, name='Beta coefficients')

    def get_pvalues(self):
        self.residuals_ml = self.left_hand_side - self.right_hand_side @ self.betas
        self.n = self.right_hand_side.shape[0]
        self.K = self.right_hand_side.shape[1]
        self.df = self.n - self.K
        self.sigma_sq = (self.residuals_ml.T @ self.residuals_ml) / self.df
        self.variance_ml = np.linalg.inv(self.right_hand_side.T@self.right_hand_side)*self.sigma_sq
        #self.std_errors_ml = np.sqrt(np.abs(np.diag(self.hess_inv)))
        self.std_errors_ml = np.sqrt(np.diag(self.variance_ml))
        self.t_stat_ml = self.betas / self.std_errors_ml
        term = np.minimum(scipy.stats.t.cdf(self.t_stat_ml, self.df), 1 - scipy.stats.t.cdf(self.t_stat_ml, self.df))
        p_values = term * 2
        return pd.Series(p_values, name='P-values for the corresponding coefficients')

    def get_model_goodness_values(self):
        y_demean = self.left_hand_side - self.left_hand_side.mean()
        SST = y_demean.T @ y_demean
        SSE = self.residuals_ml.T @ self.residuals_ml
        r2 = 1 - SSE / SST
        adj_r2 = 1 - (self.n - 1) / self.df * (1 - r2)
        return f'Centered R-squared: {r2:.3f}, Adjusted R-squared: {adj_r2:.3f}'
