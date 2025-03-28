import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

class Stock:
    def __init__(self, ticker, start_date='2022-01-01', end_date=None):
        """
        Initialize Stock class
        
        Args:
            ticker (str): Stock ticker symbol
            start_date (str): Start date for data download
            end_date (str): End date for data download (default = today)
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date or pd.Timestamp.today().strftime('%Y-%m-%d')
        
        # Download historical data
        self.download_historical_data()
        
    def download_historical_data(self):
        """
        Download historical stock data
        """
        self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        
    def plot_historical_data(self):
        """
        Plot historical stock price
        """
        plt.figure(figsize=(12, 6))
        plt.plot(self.data['Close'], label=f'{self.ticker} Price')
        plt.title(f'Historical Price for {self.ticker}')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()
        
    def calculate_log_returns(self):
        """
        Calculate and plot log returns
        
        Returns:
            pd.Series: Log returns
        """
        self.log_returns = np.log(self.data['Close'] / self.data['Close'].shift(1))
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.log_returns)
        plt.title(f'Log Returns for {self.ticker}')
        plt.xlabel('Date')
        plt.ylabel('Log Returns')
        plt.show()
        
        return self.log_returns
    
    def estimate_mu_sigma(self, period='all'):
        """
        Estimate mu and sigma from log returns
        
        Args:
            period (str): 'all' or 'short'
        
        Returns:
            tuple: mu and sigma
        """
        if not hasattr(self, 'log_returns'):
            self.calculate_log_returns()
        
        if period == 'short':
            # Use only first half of the data
            returns = self.log_returns[:len(self.log_returns)//2]
        else:
            returns = self.log_returns
        
        # Remove NaN values
        returns = returns.dropna()
        
        mu = returns.mean() * 252  # Annualized
        sigma = returns.std() * np.sqrt(252)  # Annualized volatility
        
        return mu, sigma
    
    def monte_carlo_simulation(self, 
                                mu=None, 
                                sigma=None, 
                                num_simulations=10, 
                                days=252):
        """
        Monte Carlo simulation with Geometric Brownian Motion
        
        Args:
            mu (float): Expected return (if None, calculated from data)
            sigma (float): Volatility (if None, calculated from data)
            num_simulations (int): Number of trajectories
            days (int): Number of simulation days
        
        Returns:
            np.array: Simulated trajectories
        """
        # Estimate mu and sigma if not provided
        if mu is None or sigma is None:
            mu, sigma = self.estimate_mu_sigma()
        
        # Initial price
        S0 = self.data['Close'].iloc[-1]
        
        # Simulation
        np.random.seed(42)  # For reproducibility
        daily_returns = np.random.normal(
            (mu - 0.5 * sigma**2) / 252, 
            sigma / np.sqrt(252), 
            (num_simulations, days)
        )
        
        # Generate trajectories
        stock_paths = np.zeros((num_simulations, days))
        stock_paths[:, 0] = S0
        
        for t in range(1, days):
            stock_paths[:, t] = stock_paths[:, t-1] * np.exp(daily_returns[:, t-1])
        
        return stock_paths
    
    def plot_monte_carlo(self, stock_paths=None, num_simulations=10):
        """
        Plot Monte Carlo trajectories with historical data
        
        Args:
            stock_paths (np.array): Trajectories to plot
            num_simulations (int): Number of trajectories if not provided
        """
        if stock_paths is None:
            stock_paths = self.monte_carlo_simulation(num_simulations=num_simulations)
        
        plt.figure(figsize=(15, 8))
        
        # Historical data
        plt.plot(self.data.index, self.data['Close'], label='Historical Price', color='black', linewidth=2)
        
        # Monte Carlo trajectories
        for i in range(stock_paths.shape[0]):
            dates = pd.date_range(start=self.data.index[-1], periods=len(stock_paths[i]), freq='B')
            plt.plot(dates, stock_paths[i], alpha=0.5)
        
        plt.title(f'Monte Carlo Simulation for {self.ticker}')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

def main():
    # Usage example
    stock = Stock('AAPL')  # You can change the ticker
    
    # Display historical data
    stock.plot_historical_data()
    
    # Calculate log returns
    log_returns = stock.calculate_log_returns()
    
    # Estimate mu and sigma
    print("Estimating mu and sigma:")
    mu_all, sigma_all = stock.estimate_mu_sigma('all')
    mu_short, sigma_short = stock.estimate_mu_sigma('short')
    
    print(f"Mu (all period): {mu_all}")
    print(f"Sigma (all period): {sigma_all}")
    print(f"Mu (short period): {mu_short}")
    print(f"Sigma (short period): {sigma_short}")
    
    # Monte Carlo simulation
    stock_paths = stock.monte_carlo_simulation()
    stock.plot_monte_carlo(stock_paths)

if __name__ == "__main__":
    main()