import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq
import pandas as pd

def black_scholes_price(S, K, T, r, sigma, q=0, option_type='call'):
    """
    Calculate option price using Black-Scholes model
    
    Parameters:
    S (float): Current stock price
    K (float): Strike price
    T (float): Time to expiration (in years)
    r (float): Risk-free interest rate
    sigma (float): Volatility
    q (float): Dividend yield (default 0)
    option_type (str): 'call' or 'put'
    
    Returns:
    float: Option price
    
    Raises:
    ValueError: If option_type is not 'call' or 'put'
    """
    # Input validation
    if option_type.lower() not in ['call', 'put']:
        raise ValueError("Option type must be 'call' or 'put'")
    
    d1 = (np.log(S/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type.lower() == 'call':
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # put
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    
    return price

def calculate_implied_volatility(S, K, T, r, market_price, option_type='call', method='brentq'):
    """
    Calculate implied volatility using different root-finding methods
    
    Parameters:
    S (float): Current stock price
    K (float): Strike price
    T (float): Time to expiration (in years)
    r (float): Risk-free interest rate
    market_price (float): Market price of the option
    option_type (str): 'call' or 'put'
    method (str): Root-finding method ('brentq', 'newton')
    
    Returns:
    float: Implied volatility
    """
    def option_price_diff(sigma):
        """
        Calculate the difference between market price and theoretical price
        """
        theoretical_price = black_scholes_price(S, K, T, r, sigma, option_type=option_type)
        return theoretical_price - market_price
    
    try:
        if method.lower() == 'brentq':
            # Brent's method - more robust for finding roots
            implied_vol = brentq(option_price_diff, 0.0001, 10)
        elif method.lower() == 'newton':
            # Simple Newton-Raphson method (less robust but included for comparison)
            def newton_method(x0, max_iter=100, tol=1e-5):
                x = x0
                for _ in range(max_iter):
                    fx = option_price_diff(x)
                    if abs(fx) < tol:
                        return x
                    dfx = (option_price_diff(x + 1e-8) - fx) / 1e-8  # Numerical derivative
                    x = x - fx / dfx
                raise ValueError("Newton-Raphson method did not converge")
            
            implied_vol = newton_method(0.5)
        else:
            raise ValueError("Method must be 'brentq' or 'newton'")
        
        return implied_vol
    except ValueError:
        return np.nan

def fetch_option_data(ticker_symbol, expiration=None):
    """
    Fetch option chain data for a given stock
    
    Parameters:
    ticker_symbol (str): Stock ticker symbol
    expiration (str, optional): Specific expiration date to filter (YYYY-MM-DD)
    
    Returns:
    dict: Calls and puts dataframes with additional metadata
    """
    ticker = yf.Ticker(ticker_symbol)
    
    # Get current spot price
    spot_price = ticker.history(period='1d')['Close'][0]
    
    # Get option chains
    options = ticker.options
    
    # If specific expiration is provided, filter
    if expiration:
        options = [expiration]
    
    # Collect data for all (or specified) expiration(s)
    option_data = {
        'spot_price': spot_price,
        'calls': [],
        'puts': []
    }
    
    for exp in options:
        chain = ticker.option_chain(exp)
        calls_df = chain.calls
        puts_df = chain.puts
        
        # Add expiration date to dataframes
        calls_df['expiration'] = exp
        puts_df['expiration'] = exp
        
        option_data['calls'].append(calls_df)
        option_data['puts'].append(puts_df)
    
    # Concatenate dataframes
    option_data['calls'] = pd.concat(option_data['calls']) if option_data['calls'] else pd.DataFrame()
    option_data['puts'] = pd.concat(option_data['puts']) if option_data['puts'] else pd.DataFrame()
    
    return option_data

def clean_option_data(options_data, min_volume=100, max_distance_from_spot=0.2, min_open_interest=50):
    """
    Clean option data based on multiple criteria
    
    Parameters:
    options_data (dict): Dictionary containing options data
    min_volume (int): Minimum trading volume to consider
    max_distance_from_spot (float): Maximum percentage distance from spot price
    min_open_interest (int): Minimum open interest to consider
    
    Returns:
    dict: Cleaned options data
    """
    spot_price = options_data['spot_price']
    
    for option_type in ['calls', 'puts']:
        df = options_data[option_type].copy()
        
        # Filter by volume and open interest
        df = df[(df['volume'] >= min_volume) & (df['openInterest'] >= min_open_interest)]
        
        # Filter by proximity to spot price
        df = df[np.abs(df['strike'] - spot_price) / spot_price <= max_distance_from_spot]
        
        # Sort by volume for better analysis
        df = df.sort_values('volume', ascending=False)
        
        options_data[option_type] = df
    
    return options_data

def visualize_implied_volatility(options_data, option_type='calls', max_expirations=3):
    """
    Visualize implied volatility by strike price and maturity
    
    Parameters:
    options_data (dict): Dictionary containing options data
    option_type (str): 'calls' or 'puts'
    max_expirations (int): Maximum number of expirations to plot
    """
    df = options_data[option_type]
    
    if df.empty:
        print(f"No {option_type} data available for visualization.")
        return
    
    # Group by expiration and plot
    unique_expirations = df['expiration'].unique()[:max_expirations]
    
    plt.figure(figsize=(15, 6))
    plt.title(f'Implied Volatility Surface for {option_type.capitalize()}')
    plt.xlabel('Strike Price')
    plt.ylabel('Implied Volatility')
    
    for expiration in unique_expirations:
        subset = df[df['expiration'] == expiration]
        plt.plot(subset['strike'], subset['impliedVolatility'], 
                 label=f'Expiration: {expiration}', marker='o')
    
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def options_analysis_summary(options_data):
    """
    Generate a summary of options data
    
    Parameters:
    options_data (dict): Dictionary containing options data
    
    Returns:
    dict: Summary statistics
    """
    summary = {
        'spot_price': options_data['spot_price'],
        'calls': {
            'total_count': len(options_data['calls']),
            'avg_volume': options_data['calls']['volume'].mean() if not options_data['calls'].empty else 0,
            'avg_implied_volatility': options_data['calls']['impliedVolatility'].mean() if not options_data['calls'].empty else 0,
        },
        'puts': {
            'total_count': len(options_data['puts']),
            'avg_volume': options_data['puts']['volume'].mean() if not options_data['puts'].empty else 0,
            'avg_implied_volatility': options_data['puts']['impliedVolatility'].mean() if not options_data['puts'].empty else 0,
        }
    }
    return summary

# Example usage
if __name__ == "__main__":
    # Demonstrate usage of the functions
    try:
        # Fetch option data for a stock
        ticker_symbol = "AAPL"  # Example: Apple Inc.
        options_data = fetch_option_data(ticker_symbol)
        
        # Clean the data
        cleaned_options = clean_option_data(options_data)
        
        # Visualize implied volatility
        visualize_implied_volatility(cleaned_options, 'calls')
        
        # Generate summary
        summary = options_analysis_summary(cleaned_options)
        print("Options Analysis Summary:")
        print(summary)
    
    except Exception as e:
        print(f"An error occurred: {e}")