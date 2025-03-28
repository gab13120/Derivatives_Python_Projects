import pandas as pd
import numpy as np
from scipy import stats

def load_and_clean_data(filepath):
    """
    Load and clean CRSP data
    
    Args:
        filepath (str): Path to the CSV file
    
    Returns:
        pandas.DataFrame: Cleaned data
    """
    # Load the data
    df = pd.read_csv(filepath)
    
    # Convert date to datetime format
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    
    # Clean the data
    df = df.dropna()  # Remove missing data
    
    # Filter only NYSE and AMEX exchanges
    df = df[df['PRIMEXCH'].isin(['N', 'A'])]
    
    # Convert RET to float
    df['RET'] = df['RET'].astype(float)
    
    return df

def calculate_cumulative_returns(df, months=12):
    """
    Calculate cumulative returns per stock over a given period
    
    Args:
        df (pandas.DataFrame): Cleaned data
        months (int): Return calculation period
    
    Returns:
        pandas.DataFrame: Cumulative returns per stock
    """
    # Group by PERMNO and calculate cumulative return
    cum_returns = df.groupby('PERMNO')['RET'].rolling(window=months).apply(lambda x: (1 + x).prod() - 1).reset_index()
    cum_returns.columns = ['index', 'PERMNO', 'CumulativeReturn']
    
    # Add return deciles
    cum_returns['Portfolio'] = pd.qcut(cum_returns['CumulativeReturn'], q=10, labels=False)
    
    return cum_returns

def momentum_strategy(filepath):
    """
    Implement the Jegadeesh and Titman momentum strategy
    
    Args:
        filepath (str): Path to the CSV file
    
    Returns:
        pandas.DataFrame: Strategy results
    """
    # Load and clean data
    df = load_and_clean_data(filepath)
    
    # List to store strategy returns
    retMom = []
    
    # Iterate over periods
    unique_semesters = sorted(df['Semester'].unique())
    
    for i in range(len(unique_semesters) - 2):
        # Portfolio formation period (12 months)
        formation_period_start = unique_semesters[i]
        formation_period_end = unique_semesters[i+1]
        
        # Investment holding period (6 months)
        placement_period_start = unique_semesters[i+1]
        placement_period_end = unique_semesters[i+2]
        
        # Formation period data
        formation_df = df[df['Semester'].between(formation_period_start, formation_period_end)]
        
        # Calculate cumulative returns and portfolios
        cum_returns = calculate_cumulative_returns(formation_df)
        
        # Select only Losers (0) and Winners (9) portfolios
        winners_losers = cum_returns[cum_returns['Portfolio'].isin([0, 9])]
        
        # Placement period data
        placement_df = df[df['Semester'].between(placement_period_start, placement_period_end)]
        
        # Calculate average monthly portfolio returns
        portfolio_returns = placement_df.merge(winners_losers[['PERMNO', 'Portfolio']], on='PERMNO')
        portfolio_avg_returns = portfolio_returns.groupby('Portfolio')['RET'].mean()
        
        # Calculate return differential between Winners and Losers
        if len(portfolio_avg_returns) == 2:
            retMom.append(portfolio_avg_returns[9] - portfolio_avg_returns[0])
    
    # Convert to DataFrame
    retMom = pd.DataFrame(retMom, columns=['RETStrat'])
    
    # Calculate average profitability and Student's t-test
    mean_return = retMom['RETStrat'].mean()
    t_statistic, p_value = stats.ttest_1samp(retMom['RETStrat'], 0)
    
    print(f"Strategy average return: {mean_return:.4f}")
    print(f"Student's t-test - t-statistic: {t_statistic:.4f}")
    print(f"p-value: {p_value:.4f}")
    
    return retMom

# Usage example (replace with path to your file)
# results = momentum_strategy('path/to/your/file.csv'