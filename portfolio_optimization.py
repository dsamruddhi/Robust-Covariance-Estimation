import os
import numpy as np
import pandas as pd

from estimators.linear_shrinkage import LinearShrinkage
from estimators.eigenvalue_clipping import EigenvalueClipping
from estimators.nonlinear_shrinkage import NonlinearShrinkage


class Portfolio:

    window_size = 201
    update_after = 10

    @staticmethod
    def get_data(exchange):
        if exchange == "ASX":
            filename = "ASX data.csv"
        elif exchange == "BSE":
            filename = "BSE data.csv"
        else:
            raise ValueError("Data for the specified exchange not found!")
        parent_dir = os.path.dirname(__file__)
        data_file = os.path.join(parent_dir, "data", filename)
        data = pd.read_csv(data_file, header=None)
        data = np.asarray(data)
        return data

    @staticmethod
    def get_scm(data):
        m, n = data.shape
        sample_mean = np.mean(data, axis=1)
        sample_mean = np.reshape(sample_mean, (sample_mean.shape[0], 1))
        centered_data = data - sample_mean
        scm = (1/n) * np.matmul(centered_data, np.transpose(centered_data))
        return scm

    @staticmethod
    def get_portfolio(estimator, num_features):
        inv_estimator = np.linalg.inv(estimator)
        portfolio = (inv_estimator @ np.ones((num_features, 1)))/(np.ones((1, num_features)) @ inv_estimator @ np.ones((num_features, 1)))
        return portfolio

    @staticmethod
    def get_estimator(technique, scm, num_features, num_samples):
        if technique == "scm":
            estimator = scm
            return estimator
        if technique == "identity_shrinkage":
            estimator = LinearShrinkage.get_target(scm, "identity")
            return estimator
        elif technique == "svzc_shrinkage":
            estimator = LinearShrinkage.svzc_target(scm, "svzc")
            return estimator
        elif technique == "svmc_shrinkage":
            estimator = LinearShrinkage.svzc_target(scm, "svmc")
            return estimator
        elif technique == "eigenvalue_clipping":
            concentration_ratio = num_features/num_samples
            estimator = EigenvalueClipping.clip(scm, concentration_ratio)
            return estimator
        elif technique == "nonlinear_shrinkage":
            pass
        else:
            raise ValueError(f"{technique} not implemented")

    @staticmethod
    def fit_portfolio(exchange, technique):
        data = Portfolio.get_data(exchange)
        num_features, num_samples = data.shape

        returns = []
        updates = num_samples - Portfolio.window_size

        start_day = 0
        for i in range(0, updates):
            end_day = start_day + Portfolio.window_size
            if end_day < num_samples:
                data_window = data[:, start_day:end_day]
                scm = Portfolio.get_scm(data_window)
                estimator = Portfolio.get_estimator(technique, scm, num_features, num_samples)
                portfolio = Portfolio.get_portfolio(estimator, num_features)

                for day in range(0, Portfolio.update_after):
                    new_day = end_day + day
                    if new_day < num_samples:
                        returns.append(portfolio.T @ data[:, new_day])

                start_day = start_day + Portfolio.update_after



        total_returns = np.sum(returns)
        avg_returns = np.mean(returns)
        variance_returns = np.var(returns)

        return total_returns, avg_returns, variance_returns


if __name__ == '__main__':

    exchange = "ASX"
    technique = "scm"
    total_returns, avg_returns, variance_returns = Portfolio.fit_portfolio(exchange, technique)
    print(total_returns)
    print(avg_returns)
    print(variance_returns)
