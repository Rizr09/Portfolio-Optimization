# CorporateBondPortfolioOptimizer.py

import numpy as np
import cvxpy as cp
import pandas as pd
import plotly.graph_objects as go
from typing import List, Tuple, Optional, Dict
from datetime import datetime

class CorporateBondPortfolioOptimizer:
    def __init__(
        self,
        bonds_df: pd.DataFrame
    ):
        """
        Initialize the Bond Portfolio Optimizer with DataFrame inputs

        :param bonds_df: DataFrame with bond details from Bonds.xlsx
        """
        # Prepare bond details
        self.bonds_df = bonds_df

        # Extract key bond characteristics
        self.names = bonds_df['Issuer Name'].tolist()
        self.tickers = bonds_df['Ticker'].tolist()
        self.series = bonds_df['Series'].tolist()  # New: Extract Series
        self.ratings = bonds_df['PEFINDO Rating'].tolist()

        # Convert coupon rates to decimal
        self.yields = bonds_df['Yld to Mty (Mid)'].astype(float) / 100

        # Parse maturity dates
        self.maturities = bonds_df['Maturity'].tolist()

        # Use Macaulay Duration
        self.durations = bonds_df['Mac Dur (Mid)'].astype(float)

        # Assume face value of 100 for each bond
        self.face_values = np.array([100] * len(self.names))

        # Number of bonds
        self.n_bonds = len(self.names)

        # Enhanced rating quality scores
        self.rating_scores = self._create_rating_scores()

    def _create_rating_scores(self) -> np.ndarray:
        """Create a more comprehensive rating scoring system."""
        rating_scores = {
            'idAAA': 1.0, 'idAA+': 0.95, 'idAA': 0.9, 'idAA-': 0.85,
            'idA+': 0.8, 'idA': 0.75, 'idA-': 0.7,
            'idBBB+': 0.65, 'idBBB': 0.6, 'idBBB-': 0.55,
            'idBB+': 0.5, 'idBB': 0.45, 'idBB-': 0.4,
            'idB+': 0.35, 'idB': 0.3, 'idB-': 0.25
        }
        return np.array([rating_scores.get(r, 0.0) for r in self.ratings])

    def optimize_portfolio(
        self,
        total_investment: float,  # total_investment is in Rupiah
        min_return: float,
        max_position_size: float = 0.4,
        aggregate_rating_score: float = None,  # Made optional
        min_rating_score: float = None  # Made optional
    ) -> Tuple[np.ndarray, dict]:
        """
        Optimize the bond portfolio with either aggregate or minimum rating constraint.
        
        :param total_investment: Total investment amount in Rupiah
        :param min_return: Minimum acceptable return for the portfolio
        :param max_position_size: Maximum allowable position size as fraction of total portfolio
        :param aggregate_rating_score: Target average rating score for the portfolio (optional)
        :param min_rating_score: Minimum allowable rating score for individual bonds (optional)
        :return: Tuple containing the optimized weights and a dictionary of results
        """
        w = cp.Variable(self.n_bonds)

        expected_returns = self.yields
        portfolio_return = w @ expected_returns
        objective = cp.Maximize(portfolio_return)

        # Base constraints
        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            w <= max_position_size,
            portfolio_return >= min_return
        ]

        # Add rating constraint based on which parameter is provided
        if aggregate_rating_score is not None:
            constraints.append(w @ self.rating_scores >= aggregate_rating_score)
        elif min_rating_score is not None:
            invalid_bonds = self.rating_scores < min_rating_score
            constraints.extend([w[i] == 0 for i in np.where(invalid_bonds)[0]])

        problem = cp.Problem(objective, constraints)
        problem.solve()

        # Get weights and ensure they are non-negative
        weights = w.value
        weights = np.clip(weights, 0, None)
        
        if weights is None or weights.sum() == 0:
            weights = np.zeros(self.n_bonds)
            results = {
                'status': 'infeasible',
                'optimal_value': 0,
                'expected_return': 0
            }
        else:
            # Normalize weights to sum to 1
            weights = weights / weights.sum()
            results = {
                'status': problem.status,
                'optimal_value': problem.value,
                'expected_return': float(portfolio_return.value)
            }

        # Store total investment for report generation
        self.total_investment = total_investment

        return weights, results

    def generate_report(self, weights: np.ndarray) -> Tuple[pd.DataFrame, Dict, Dict[str, go.Figure]]:
        """Generate an enhanced report with detailed portfolio analysis."""
        # Combine Ticker, Maturity, and Series to create unique identifiers
        combined_names = [f"{ticker} {maturity} {series}" for ticker, maturity, series in zip(self.tickers, self.maturities, self.series)]
        portfolio_data = pd.DataFrame({
            'Issuer': self.names,
            'Ticker': self.tickers,
            'Series': self.series,  # New: Include Series
            'Weight (%)': weights * 100,
            'Yield (%)': self.yields * 100,
            'Duration (years)': self.durations,
            'Rating': self.ratings,
            'Maturity': self.maturities,
            'Investment (Rp)': weights * self.total_investment,
            'Expected Return (%)': weights * self.yields * 100
        })

        # Create summary statistics
        total_investment = self.total_investment
        portfolio_yield = np.sum(weights * self.yields) * 100
        portfolio_duration = np.sum(weights * self.durations)
        avg_rating_score = np.sum(weights * self.rating_scores)

        summary_data = {
            'Total Investment (Rp)': f"{total_investment:,.2f}",
            'Portfolio Yield (%)': f"{portfolio_yield:.2f}",
            'Portfolio Duration (years)': f"{portfolio_duration:.2f}",
            'Average Rating Score': f"{avg_rating_score:.2f}",
            'Number of Positions': f"{(weights > 0.001).sum()}",
            'Analysis Date': datetime.now().strftime('%Y-%m-%d')
        }

        # Generate interactive plots
        figures = self._create_interactive_plots(portfolio_data, weights)

        return portfolio_data, summary_data, figures

    def _create_interactive_plots(self, portfolio_data: pd.DataFrame, weights: np.ndarray) -> Dict[str, go.Figure]:
        """Create enhanced interactive plotly visualizations."""
        figures = {}
        combined_names = [f"{ticker} {maturity} {series}" for ticker, maturity, series in zip(self.tickers, self.maturities, self.series)]

        # 1. Portfolio Composition Sunburst
        figures['composition'] = go.Figure(go.Sunburst(
            labels=combined_names + ['Portfolio'],
            parents=['Portfolio'] * len(self.names) + [''],
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

        # 2. Rating Distribution Treemap
        rating_data = pd.DataFrame({
            'Rating': self.ratings,
            'Weight': weights * 100
        }).groupby('Rating').sum().reset_index()

        figures['ratings'] = go.Figure(go.Treemap(
            labels=rating_data['Rating'],
            parents=[''] * len(rating_data),
            values=rating_data['Weight'],
            textinfo='label+percent parent',
            textfont=dict(size=20)
        ))
        figures['ratings'].update_layout(
            title='Portfolio Rating Distribution',
            width=800,
            height=600
        )

        return figures
