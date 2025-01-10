import numpy as np
from scipy.optimize import minimize

class MPT:
    def __init__(self, returns, risk_free_rate=0.07, annual_factor=252):
        self.returns = returns
        self.mean_returns = returns.mean()
        self.cov_matrix = returns.cov()
        self.assets = returns.columns.tolist()
        self.num_assets = len(self.assets)
        self.risk_free_rate = risk_free_rate
        self.annual_factor = annual_factor

    def portfolio_performance(self, weights):
        port_return = np.sum(self.mean_returns * weights) * self.annual_factor
        port_std = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix * self.annual_factor, weights)))
        return port_return, port_std

    def sharpe_ratio(self, port_return, port_std):
        return (port_return - self.risk_free_rate) / port_std

    def neg_sharpe_ratio(self, weights):
        p_ret, p_std = self.portfolio_performance(weights)
        return -(p_ret - self.risk_free_rate) / p_std

    def portfolio_volatility(self, weights):
        return self.portfolio_performance(weights)[1]

    def constraint_sum_of_weights(self, weights):
        return np.sum(weights) - 1.0

    def maximize_sharpe(self):
        constraints = ({'type': 'eq', 'fun': self.constraint_sum_of_weights})
        bounds = tuple((0.0, 1.0) for _ in range(self.num_assets))
        result = minimize(
            self.neg_sharpe_ratio,
            self.num_assets * [1.0 / self.num_assets],
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        return result

    def minimize_volatility(self):
        constraints = ({'type': 'eq', 'fun': self.constraint_sum_of_weights})
        bounds = tuple((0.0, 1.0) for _ in range(self.num_assets))
        result = minimize(
            self.portfolio_volatility,
            self.num_assets * [1.0 / self.num_assets],
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        return result

    def calculate_efficient_frontier(self, points=50):
        min_vol_result = self.minimize_volatility()
        max_sharpe_result = self.maximize_sharpe()

        min_vol_ret, _ = self.portfolio_performance(min_vol_result.x)
        max_sharpe_ret, _ = self.portfolio_performance(max_sharpe_result.x)

        target_returns = np.linspace(min_vol_ret, max_sharpe_ret, points)
        frontier_volatilities = []
        bounds = tuple((0.0, 1.0) for _ in range(self.num_assets))

        for tr in target_returns:
            constraints = (
                {'type': 'eq', 'fun': self.constraint_sum_of_weights},
                {'type': 'eq', 'fun': lambda w, tr=tr: self.portfolio_performance(w)[0] - tr}
            )
            result = minimize(
                self.portfolio_volatility,
                self.num_assets * [1.0 / self.num_assets],
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            if result.success:
                frontier_volatilities.append(result.fun)
            else:
                frontier_volatilities.append(np.nan)
        return frontier_volatilities, target_returns