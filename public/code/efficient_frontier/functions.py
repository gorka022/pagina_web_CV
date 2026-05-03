import yfinance as yf
import numpy as np
import scipy.optimize as sco
import pandas as pd
import matplotlib.pyplot as plt


def fetch_data(tickers, start_date, end_date):
    """Downloads historical adjusted close prices and calculates daily returns."""
    data = yf.download(tickers, start=start_date, end=end_date)
    try:
        prices = data['Adj Close']
    except KeyError:
        prices = data['Close']
    returns = prices.pct_change().dropna()
    return returns


def calculate_portfolio_performance(weights, mean_returns, cov_matrix):
    """Calculates expected annualized return and volatility of a portfolio."""
    returns = np.sum(mean_returns * weights)
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return returns, std_dev


def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    """Objective function to minimize (equivalent to maximizing Sharpe Ratio)."""
    p_ret, p_std = calculate_portfolio_performance(weights, mean_returns, cov_matrix)
    return - (p_ret - risk_free_rate) / p_std


def portfolio_volatility(weights, mean_returns, cov_matrix):
    """Objective function to minimize volatility."""
    return calculate_portfolio_performance(weights, mean_returns, cov_matrix)[1]


def optimize_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate, num_assets):
    """Finds the portfolio weights that maximize the Sharpe Ratio."""
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_guess = num_assets * [1. / num_assets]

    result = sco.minimize(negative_sharpe_ratio, initial_guess, args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)
    return result


def minimize_portfolio_volatility(mean_returns, cov_matrix, num_assets):
    """Finds the portfolio weights that minimize variance (volatility)."""
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_guess = num_assets * [1. / num_assets]

    result = sco.minimize(portfolio_volatility, initial_guess, args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)
    return result


def generate_random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate, num_assets):
    """Generates random portfolios to create the 'cloud' of the efficient frontier."""
    results = np.zeros((3, num_portfolios))

    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        p_ret, p_std = calculate_portfolio_performance(weights, mean_returns, cov_matrix)
        results[0, i] = p_std
        results[1, i] = p_ret
        results[2, i] = (p_ret - risk_free_rate) / p_std

    return results


def efficient_return(mean_returns, cov_matrix, target_return, num_assets):
    """Finds the portfolio weights that minimize volatility for a given target return."""
    args = (mean_returns, cov_matrix)

    def portfolio_return(weights):
        return calculate_portfolio_performance(weights, mean_returns, cov_matrix)[0]

    constraints = (
        {'type': 'eq', 'fun': lambda x: portfolio_return(x) - target_return},
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    )
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_guess = num_assets * [1. / num_assets]

    result = sco.minimize(portfolio_volatility, initial_guess, args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)
    return result


def calculate_efficient_frontier(mean_returns, cov_matrix, num_assets, min_ret, max_ret):
    """Calculates the full efficient/inefficient frontier curve."""
    target_returns = np.linspace(min_ret, max_ret, 50)
    frontier_volatility = []
    
    for target in target_returns:
        opt_result = efficient_return(mean_returns, cov_matrix, target, num_assets)
        frontier_volatility.append(opt_result.fun)
        
    return target_returns, frontier_volatility


def plot_portfolio_weights(weights_sharpe, weights_vol, tickers):
    """Plots a grouped bar chart comparing the components of both optimal portfolios."""
    x = np.arange(2)
    num_tickers = len(tickers)

    total_width = 0.8
    width = total_width / num_tickers

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, ticker in enumerate(tickers):
        values = [weights_sharpe[i] * 100, weights_vol[i] * 100]
        pos = x - (total_width / 2) + (i * width) + (width / 2)

        rects = ax.bar(pos, values, width, label=ticker, alpha=0.8)
        ax.bar_label(rects, fmt='%.1f%%', padding=3, fontsize=9)

    ax.set_ylabel('Allocation Weight (%)')
    ax.set_title('Portfolio Allocations by Strategy')
    ax.set_xticks(x)
    ax.set_xticklabels(['Maximum Sharpe Ratio\n(All Components)', 'Minimum Volatility\n(All Components)'])
    ax.legend(title="Assets", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show(block=False)


def plot_efficient_frontier(results, max_sharpe_ret, max_sharpe_std, min_vol_ret, min_vol_std, risk_free_rate, frontier_x=None, frontier_y=None):
    """Plots the simulated portfolios, optimal points, the Capital Market Line, and the Efficient Frontier."""
    plt.figure(figsize=(12, 7))

    # 1. Plot the random portfolio cloud
    plt.scatter(results[0, :], results[1, :], c=results[2, :], cmap='viridis', marker='o', s=10, alpha=0.5)
    plt.colorbar(label='Sharpe Ratio')
    
    # 1.5 Plot the continuous efficient frontier line
    if frontier_x is not None and frontier_y is not None:
        frontier_x = np.array(frontier_x)
        frontier_y = np.array(frontier_y)
        min_idx = np.argmin(frontier_x)
        
        # Efficient frontier (from min volatility upwards)
        plt.plot(frontier_x[min_idx:], frontier_y[min_idx:], 'k-', linewidth=2, label='Efficient Frontier', zorder=4)
        # Inefficient frontier (from bottom to min volatility)
        plt.plot(frontier_x[:min_idx+1], frontier_y[:min_idx+1], 'k:', linewidth=2, label='Inefficient Frontier', zorder=4)

    # 2. Highlight optimal portfolios
    plt.scatter(max_sharpe_std, max_sharpe_ret, marker='*', color='red', s=400, label='Max Sharpe Ratio (Tangency)',
                zorder=5)
    plt.scatter(min_vol_std, min_vol_ret, marker='*', color='blue', s=400, label='Minimum Volatility', zorder=5)

    # 3. Calculate and Plot the Capital Market Line (CML)
    # The slope of the CML is the Maximum Sharpe Ratio
    max_sharpe_ratio = (max_sharpe_ret - risk_free_rate) / max_sharpe_std

    # Create X values (Risk) from 0 extending past the maximum risk in our simulation
    cml_x = np.linspace(0, max(results[0, :]) * 1.1, 100)

    # Create Y values (Return) using the line equation: y = mx + b
    cml_y = risk_free_rate + (max_sharpe_ratio * cml_x)

    plt.plot(cml_x, cml_y, color='red', linestyle='--', linewidth=2, label='Capital Market Line (CML)')

    # 4. Highlight the Risk-Free Rate starting point
    plt.scatter(0, risk_free_rate, marker='o', color='black', s=100, label=f'Risk-Free Rate ({risk_free_rate:.1%})',
                zorder=5)

    # Formatting
    plt.xlim(xmin=0)  # Force X-axis to start at 0 so we see the CML intercept
    plt.title('Efficient Frontier & Capital Market Line')
    plt.xlabel('Annualized Volatility (Risk)')
    plt.ylabel('Annualized Return')
    plt.legend(labelspacing=0.8, loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.show()