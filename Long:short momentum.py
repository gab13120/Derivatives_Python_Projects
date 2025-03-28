import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class MomentumValueStrategy:
    def __init__(self, filepath):
        """
        Initialize the strategy
        
        Args:
            filepath (str): Path to the Excel file containing data
        """
        # Load data
        self.data = pd.read_excel(filepath)
        
        # Convert date column to datetime
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        
        # Initialize lists to store results
        self.monthly_returns = []
        self.portfolio_weights = []
    
    def calculate_momentum_score(self, returns):
        """
        Calculate momentum score
        
        Args:
            returns (pd.Series): Monthly returns
        
        Returns:
            float: Standardized momentum score
        """
        # Exclude last month to account for short-term reversions
        returns_12m = returns[:-1]
        
        # Calculate centered standard deviation
        if len(returns_12m) > 0:
            return (returns_12m.mean() - returns_12m.mean().mean()) / returns_12m.std()
        return 0
    
    def calculate_value_score(self, bv_ratio):
        """
        Calculate value score
        
        Args:
            bv_ratio (float): Book-to-market ratio
        
        Returns:
            float: Standardized value score
        """
        # Invert the book-to-market ratio
        inv_bv_ratio = 1 / bv_ratio
        
        # Calculate centered standard deviation
        return (inv_bv_ratio - inv_bv_ratio.mean()) / inv_bv_ratio.std()
    
    def build_portfolio(self, data_slice):
        """
        Build long and short portfolios
        
        Args:
            data_slice (pd.DataFrame): Data slice for a month
        
        Returns:
            tuple: Long and short portfolios with their weights
        """
        # Calculate momentum and value scores
        momentum_scores = data_slice.apply(
            lambda row: self.calculate_momentum_score(
                self.data[self.data['Ticker'] == row['Ticker']]['Return']
            ), 
            axis=1
        )
        
        value_scores = data_slice['BV_Ratio'].apply(self.calculate_value_score)
        
        # Calculate global score
        data_slice['Global_Score'] = (momentum_scores + value_scores) / 2
        
        # Sort and select portfolios
        long_portfolio = data_slice.nlargest(15, 'Global_Score')
        short_portfolio = data_slice.nsmallest(15, 'Global_Score')
        
        # Weight proportionally to absolute scores
        long_weights = long_portfolio['Global_Score'].abs() / long_portfolio['Global_Score'].abs().sum()
        short_weights = short_portfolio['Global_Score'].abs() / short_portfolio['Global_Score'].abs().sum()
        
        return (long_portfolio, long_weights), (short_portfolio, short_weights)
    
    def run_strategy(self):
        """
        Execute strategy over entire period
        """
        # Group data by month
        monthly_groups = self.data.groupby(pd.Grouper(key='Date', freq='M'))
        
        for date, data_slice in monthly_groups:
            # Build portfolios
            (long_portfolio, long_weights), (short_portfolio, short_weights) = self.build_portfolio(data_slice)
            
            # Calculate portfolio return
            long_return = (long_portfolio['Return'] * long_weights).sum()
            short_return = -(short_portfolio['Return'] * short_weights).sum()  # Short positions
            
            portfolio_return = long_return + short_return
            
            # Store results
            self.monthly_returns.append(portfolio_return)
            
            # Store weights
            self.portfolio_weights.append({
                'Date': date,
                'Long_Portfolio': dict(zip(long_portfolio['Ticker'], long_weights)),
                'Short_Portfolio': dict(zip(short_portfolio['Ticker'], short_weights))
            })
    
    def analyze_performance(self):
        """
        Analyze strategy performance
        
        Returns:
            dict: Performance metrics
        """
        returns_series = pd.Series(self.monthly_returns)
        
        performance = {
            'Cumulative_Return': (1 + returns_series).prod() - 1,
            'Annualized_Return': (1 + returns_series).prod() ** (12 / len(returns_series)) - 1,
            'Annualized_Volatility': returns_series.std() * np.sqrt(12),
            'Sharpe_Ratio': returns_series.mean() / returns_series.std() * np.sqrt(12),
            'Maximum_Drawdown': (1 - (1 + returns_series).cumprod().cummax()).max()
        }
        
        return performance
    
    def generate_results_excel(self):
        """
        Generate Excel file with performances and weights
        """
        # Monthly performances
        performance_df = pd.DataFrame({
            'Date': pd.date_range(start=self.data['Date'].min(), end=self.data['Date'].max(), freq='M'),
            'Monthly_Return': self.monthly_returns
        })
        
        # Weights
        weights_df = pd.DataFrame(self.portfolio_weights)
        
        # Save to Excel file
        with pd.ExcelWriter('Momentum_Value_Strategy_Results.xlsx') as writer:
            performance_df.to_excel(writer, sheet_name='Monthly_Returns', index=False)
            weights_df.to_excel(writer, sheet_name='Portfolio_Weights', index=False)
    
    def compare_with_random_strategy(self, num_simulations=1000):
        """
        Compare strategy with random strategies
        
        Args:
            num_simulations (int): Number of random simulations
        
        Returns:
            dict: Comparison results
        """
        strategy_returns = pd.Series(self.monthly_returns)
        random_returns = []
        
        for _ in range(num_simulations):
            # Generate random returns with same length
            random_strategy = np.random.normal(
                strategy_returns.mean(), 
                strategy_returns.std(), 
                len(strategy_returns)
            )
            random_returns.append(random_strategy)
        
        # Calculate random strategy performances
        random_performance = {
            'Cumulative_Return': [(1 + pd.Series(sim)).prod() - 1 for sim in random_returns],
            'Sharpe_Ratio': [(pd.Series(sim).mean() / pd.Series(sim).std()) * np.sqrt(12) for sim in random_returns]
        }
        
        # Compare with our strategy
        comparison = {
            'Strategy_Cumulative_Return': strategy_returns.cumsum()[-1],
            'Strategy_Sharpe_Ratio': strategy_returns.mean() / strategy_returns.std() * np.sqrt(12),
            'Random_Cumulative_Return_Mean': np.mean(random_performance['Cumulative_Return']),
            'Random_Cumulative_Return_Percentile': stats.percentileofscore(
                random_performance['Cumulative_Return'], 
                strategy_returns.cumsum()[-1]
            ),
            'Random_Sharpe_Ratio_Percentile': stats.percentileofscore(
                random_performance['Sharpe_Ratio'], 
                strategy_returns.mean() / strategy_returns.std() * np.sqrt(12)
            )
        }
        
        return comparison

# Usage example
def main():
    filepath = 'DATA.xlsx'  # Replace with your file path
    
    # Initialize and run strategy
    strategy = MomentumValueStrategy(filepath)
    strategy.run_strategy()
    
    # Analyze performance
    performance = strategy.analyze_performance()
    print("Strategy Performance:")
    for metric, value in performance.items():
        print(f"{metric}: {value:.4f}")
    
    # Generate Excel file
    strategy.generate_results_excel()
    
    # Compare with random strategies
    random_comparison = strategy.compare_with_random_strategy()
    print("\nComparison with Random Strategies:")
    for metric, value in random_comparison.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()