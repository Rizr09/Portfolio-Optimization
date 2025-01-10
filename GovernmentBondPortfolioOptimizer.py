import numpy as np
import cvxpy as cp
import pandas as pd
import plotly.graph_objects as go
from typing import List, Tuple, Optional, Dict
from datetime import datetime

class GovernmentBondPortfolioOptimizer:
    def __init__(
        self,
        bonds_df: pd.DataFrame
    ):
        """
        Initialize the Government Bond Portfolio Optimizer with DataFrame inputs

        :param bonds_df: DataFrame with bond details from Bonds.xlsx
        """
        # Prepare bond details
        self.bonds_df = bonds_df

        # Extract key bond characteristics
        self.issuer_names = bonds_df['Issuer Name'].tolist()
        self.tickers = bonds_df['Ticker'].tolist()
        self.series = bonds_df['Series'].tolist()
        self.yields = bonds_df['Yld to Mty (Mid)'].astype(float) / 100.0  # Konversi ke desimal
        self.coupon_rates = bonds_df['Cpn'].astype(float) / 100  # Convert to decimal
        self.maturities = pd.to_datetime(bonds_df['Maturity'], dayfirst=True).tolist()
        self.currency = bonds_df['Currency'].tolist()

        # Number of bonds
        self.n_bonds = len(self.issuer_names)

        # Compute duration as years to maturity from today
        self.durations = self._compute_durations()

    def _compute_durations(self) -> np.ndarray:
        """Compute the duration of each bond as years to maturity."""
        today = pd.to_datetime(datetime.now().date())
        durations = []
        for mty in self.maturities:
            delta = mty - today
            duration = max(delta.days / 365.25, 0)  # Ensure non-negative
            durations.append(duration)
        return np.array(durations)

    def optimize_portfolio(
        self,
        total_investment: float,  # Total investment in Rupiah
        min_return: float,
        max_position_size: float = 0.4
    ) -> Tuple[np.ndarray, dict]:
        """
        Optimize the government bond portfolio.

        :param total_investment: Total investment amount in Rupiah
        :param min_return: Minimum acceptable return for the portfolio
        :param max_position_size: Maximum allowable position size as a fraction of total portfolio (default is 0.4)
        :return: Tuple containing the optimized weights and a dictionary of results
        """
        w = cp.Variable(self.n_bonds)

        expected_returns = self.yields.values
        portfolio_return = w @ expected_returns
        objective = cp.Maximize(portfolio_return)

        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            w <= max_position_size,
            portfolio_return >= min_return
        ]

        problem = cp.Problem(objective, constraints)
        problem.solve()

        # Get weights and ensure they are non-negative
        weights = w.value
        if weights is None:
            weights = np.zeros(self.n_bonds)
        else:
            weights = np.clip(weights, 0, None)
            if weights.sum() == 0:
                weights = np.zeros(self.n_bonds)
            else:
                weights = weights / np.sum(weights)  # Normalize to sum to 1

        # Store total investment for report generation
        self.total_investment = total_investment

        results = {
            'status': problem.status,
            'optimal_value': problem.value,
            'expected_return': float(portfolio_return.value) if portfolio_return.value is not None else 0.0
        }

        return weights, results

    def generate_report(self, weights: np.ndarray) -> Tuple[pd.DataFrame, Dict, Dict[str, go.Figure]]:
        """Generate a detailed portfolio report."""
        # Combine Ticker, Maturity, and Series to create unique identifiers
        combined_names = [f"{ticker} {mty.strftime('%d/%m/%Y')} {series}" for ticker, mty, series in zip(self.tickers, self.maturities, self.series)]
        
        portfolio_data = pd.DataFrame({
            'Issuer': self.issuer_names,
            'Ticker': self.tickers,
            'Series': self.series,
            'Weight (%)': weights * 100,
            'Yield (%)': self.yields * 100,
            'Duration (years)': self.durations,
            'Maturity': [mty.strftime('%d/%m/%Y') for mty in self.maturities],
            'Investment (Rp)': weights * self.total_investment,
            'Expected Return (%)': weights * self.yields * 100
        })

        # Create summary statistics
        total_investment = self.total_investment
        portfolio_yield = np.sum(weights * self.yields) * 100
        portfolio_duration = np.sum(weights * self.durations)

        summary_data = {
            'Total Investment (Rp)': f"{total_investment:,.2f}",
            'Portfolio Yield (%)': f"{portfolio_yield:.2f}",
            'Portfolio Duration (years)': f"{portfolio_duration:.2f}",
            'Number of Positions': f"{(weights > 0.001).sum()}",
            'Analysis Date': datetime.now().strftime('%Y-%m-%d')
        }

        # Generate interactive plots
        figures = self._create_interactive_plots(portfolio_data, weights)

        return portfolio_data, summary_data, figures

    def _create_interactive_plots(self, portfolio_data: pd.DataFrame, weights: np.ndarray) -> Dict[str, go.Figure]:
        """Create interactive Plotly visualizations."""
        figures = {}
        combined_names = [f"{ticker} {mty.strftime('%d/%m/%Y')} {series}" for ticker, mty, series in zip(self.tickers, self.maturities, self.series)]

        # 1. Portfolio Composition Sunburst
        figures['composition'] = go.Figure(go.Sunburst(
            labels=combined_names + ['Portfolio'],
            parents=['Portfolio'] * len(self.issuer_names) + [''],
            values=np.append(weights * 100, weights.sum() * 100),
            branchvalues='total',
            textinfo='label+percent entry',
            maxdepth=2
        ))
        figures['composition'].update_layout(
            title='Portfolio Composition',
            width=800,
            height=800
        )

        # 2. Duration Distribution Bar Chart
        figures['duration'] = go.Figure(go.Bar(
            x=combined_names,
            y=portfolio_data['Duration (years)'],
            marker_color='indigo'
        ))
        figures['duration'].update_layout(
            title='Duration Distribution',
            xaxis_title='Bond',
            yaxis_title='Duration (years)',
            width=800,
            height=600
        )

        return figures
