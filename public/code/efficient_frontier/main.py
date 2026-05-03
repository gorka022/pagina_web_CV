import pandas as pd
from datetime import datetime, timedelta

import functions as pf


def main():
    # --- 1. Define Parameters ---
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * 5)
    risk_free_rate = 0.02
    num_assets = len(tickers)
    num_portfolios = 5000

    print(f"Fetching data for: {', '.join(tickers)}...")
    print(f"Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n")

    # --- 2. Fetch Data and Calculate Base Metrics ---
    returns = pf.fetch_data(tickers, start_date, end_date)
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    # --- 3. Optimize for Maximum Sharpe Ratio ---
    opt_sharpe = pf.optimize_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate, num_assets)
    max_sharpe_ret, max_sharpe_std = pf.calculate_portfolio_performance(opt_sharpe.x, mean_returns, cov_matrix)
    max_sharpe_ratio = (max_sharpe_ret - risk_free_rate) / max_sharpe_std

    # --- 4. Optimize for Minimum Volatility ---
    opt_vol = pf.minimize_portfolio_volatility(mean_returns, cov_matrix, num_assets)
    min_vol_ret, min_vol_std = pf.calculate_portfolio_performance(opt_vol.x, mean_returns, cov_matrix)
    min_vol_sharpe_ratio = (min_vol_ret - risk_free_rate) / min_vol_std

    # --- 5. Output Results to Console ---
    print("-" * 45)
    print(" MAXIMUM SHARPE RATIO PORTFOLIO ")
    print("-" * 45)
    print(f"Expected Annual Return: {max_sharpe_ret:.2%}")
    print(f"Annual Volatility:      {max_sharpe_std:.2%}")
    print(f"Sharpe Ratio:           {max_sharpe_ratio:.2f}")
    print("Optimal Weights:")
    for ticker, weight in zip(tickers, opt_sharpe.x):
        print(f"  {ticker}: {weight:.2%}")

    print("\n" + "-" * 45)
    print(" MINIMUM VOLATILITY PORTFOLIO ")
    print("-" * 45)
    print(f"Expected Annual Return: {min_vol_ret:.2%}")
    print(f"Annual Volatility:      {min_vol_std:.2%}")
    print(f"Sharpe Ratio:           {min_vol_sharpe_ratio:.2f}")
    print("Optimal Weights:")
    for ticker, weight in zip(tickers, opt_vol.x):
        print(f"  {ticker}: {weight:.2%}")
    print("-" * 45 + "\n")

    # --- 6. Generate Data for Graphs ---
    print("Simulating portfolios for the Efficient Frontier plot...")
    results = pf.generate_random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate, num_assets)

    print("Calculating Efficient Frontier curve...")
    max_ret = mean_returns.max()
    min_ret = mean_returns.min()
    frontier_y, frontier_x = pf.calculate_efficient_frontier(mean_returns, cov_matrix, num_assets, min_ret, max_ret)

    # --- 7. Plot the Results ---
    print("Opening both graphs simultaneously...")
    pf.plot_portfolio_weights(opt_sharpe.x, opt_vol.x, tickers)
    pf.plot_efficient_frontier(results, max_sharpe_ret, max_sharpe_std, min_vol_ret, min_vol_std, risk_free_rate, frontier_x, frontier_y)


if __name__ == "__main__":
    main()