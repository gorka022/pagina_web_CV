# -*- coding: utf-8 -*-
"""

Option pricing and sensitivities.


"""

import warnings 
import numpy as np

from numpy.random import default_rng

from typing import Callable, Tuple, Union, Dict

from scipy.integrate import quad
from scipy.stats import norm

from tools_qfb import numerical_derivative, numerical_second_derivative
from my_stochastic_processes import simulate_geometric_brownian_motion

#%% 

def S(
    S_0: float,
    mu: float,
    sigma: float,
    T: float,
    x: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """ Future (spot) price of the underlying in the Black-Scholes model.
    
    Args:
        S_0: Value of the underlying at t_0.
        mu: Expected return.
        sigma: Volatility.
        T: The price is computed at time (t0 + T).
        x: value of the N(0,1) random variable.

    Returns:
        Price of the underlying in the Black-Scholes model at time (t0 + t).

    Examples:
        
        >>> S_0, mu, sigma, T = 100, 0.05, 0.30, 2.5
        >>> X = np.array([-1.0, 0.0, 1.0])
        >>> np.round(S(S_0, mu, sigma, T, X), decimals=4)
        array([ 63.0122, 101.2578, 162.7169])
    """
    return S_0 * np.exp((mu - 0.5*sigma**2) * T + sigma * np.sqrt(T) * x)

#%% 


def price_european_vanilla_option(
    S_0: float,
    K: float,
    r: float,
    T: float,
    sigma: float,
):
    """ MC price of a European option over two underlyings

    Args:
        S_0: Value of the underlying at :math:`t_0`.
        K: Strike.
        r: Risk-free interest rate.
        T: Time to maturity :math:`(t_0+T)`.
        sigma: Volatility of the underlying.

    Returns:
        Prices of call and put European options.

    Examples:

        Prices of European call and put options

        >>> S_0, K, r, T, sigma = (
        ...     100.0, 105.0, 0.05, 2.5, np.array([0.10, 0.2]))
        >>> price_call, price_put = price_european_vanilla_option(
        ...    S_0, K, r, T, sigma)
        >>> print(np.round((price_call, price_put), 4))
        [[10.4289 16.1179]
         [ 3.0911  8.7801]]

        Call put parity

        >>> S_0, K, r, T, sigma = 100.0, 105.0, 0.05, 2.5, 0.30
        >>> price_call, price_put = price_european_vanilla_option(
        ...    S_0, K, r, T, sigma)
        >>> print(np.round((price_call, price_put), 4))
        [21.9568 14.619 ]

        >>> print(np.round(price_call - price_put, 4),
        ...    np.round(S_0 - K * np.exp(- r * T), 4))
        7.3378 7.3378

    
    """
    discount_factor = np.exp(- r * T)
    discounted_K = K * discount_factor

    total_volatility = sigma * np.sqrt(T)

    d_plus = (np.log(S_0 / discounted_K) / total_volatility
              + 0.5 * total_volatility)
    d_minus = d_plus - total_volatility

    normcdf_d_plus = norm.cdf(d_plus)
    normcdf_d_minus = norm.cdf(d_minus)

    price_call = S_0 * normcdf_d_plus - discounted_K * normcdf_d_minus
   
    price_put = (discounted_K * (1.0 - norm.cdf(d_minus))
                 - S_0 * (1.0 - norm.cdf(d_plus)))

    return price_call, price_put

#%% 

def sensitivities_european_call_option(
    S_0: float,
    K: float,
    r: float,
    T: float,
    sigma: float,
) -> Dict[str, float]:
    """ sensitivities of a European call options

    Args:
        S_0: Value of the underlying at :math:`t_0`.
        K: Strike.
        r: Risk-free interest rate.
        T: Time to maturity :math:`(t_0+T)`.
        sigma: Volatility of the underlying.

    Returns:
        sensitivities :math:`\\delta, \\gamma, \\rho, \\theta, \\nu` (vega) 
        of a European Call option.

    Examples:

        >>> S_0, K, r, T, sigma = 100.0, 105.0, 0.05, 2.5, 0.30
        >>> sensitivities = sensitivities_european_call_option(
        ...    S_0, K, r, T, sigma)
        >>> _ = [print('{} = {}'.format(name, sensitivity))
        ...         for name, sensitivity in sensitivities.items()] 
        delta = 0.6546241898707459
        gamma = 0.007770523595986478
        rho = 108.76410815474543
        theta = -5.672017781288824
        vega = 58.278926969898585

    """
    discount_factor = np.exp(- r * T)
    discounted_K = K * discount_factor

    total_volatility = sigma * np.sqrt(T)

    d_plus = (np.log(S_0 / discounted_K) / total_volatility
              + 0.5 * total_volatility)
    d_minus = d_plus - total_volatility

    normcdf_d_plus = norm.cdf(d_plus)
    normcdf_d_minus = norm.cdf(d_minus)
    normpdf_d_plus = norm.pdf(d_plus)

    sensitivities = dict.fromkeys(('delta', 'gamma', 'rho', 'theta', 'vega'))
    sensitivities = {}  # alternative
    
    sensitivities['delta'] = normcdf_d_plus
    sensitivities['gamma'] = normpdf_d_plus / (S_0 * total_volatility)
    
    sensitivities['rho'] = T * discounted_K * normcdf_d_minus
    
    sensitivities['theta'] = (
        - S_0 * sigma / (2.0 * np.sqrt(T) ) * normpdf_d_plus 
        - r * discounted_K * normcdf_d_minus
    )
    
    sensitivities['vega'] = S_0 * np.sqrt(T) * normpdf_d_plus
    
    return sensitivities


#%% 

def price_european_option(
    S_0: float,
    r: float,
    T: float,
    sigma: float,
    payoff: Callable = lambda S_T: S_T
) ->  float:
    """ Compute the price of a European option over a single underlying.

    Args:
        S_0: Value of the underlying at :math:`t_0`.
        r: Risk-free interest rate.
        T: Time to maturity :math:`(t_0+T)`.
        sigma: Volatility of the underlying.
        payoff: Function that computes the payoff at maturity.

    Returns:
        Price of the European option.

    Examples:

        Price of the underlying.

        >>> S_0, K, r, T, sigma = 100.0, 90.0, 0.05, 2.5, 0.30
        >>> payoff = lambda S_T: S_T
        >>> np.round(price_european_option(S_0, r, T, sigma, payoff), 4)
        100.0

        Price of a European call option.

        >>> S_0, K, r, T, sigma = 100.0, 105.0, 0.05, 2.5, 0.30
        >>> payoff_call = lambda S_T: np.maximum(S_T - K, 0.0)
        >>> price_call = price_european_option(S_0, r, T, sigma, payoff_call)
        >>> np.round(price_call, 4)
        21.9568

        Price of a European put option.

        >>> S_0, K, r, T, sigma = 100.0, 105.0, 0.05, 2.5, 0.30
        >>> payoff_put = lambda S_T: np.maximum(K - S_T, 0.0)
        >>> price_put = price_european_option(S_0, r, T, sigma, payoff_put)
        >>> np.round(price_put, 4)
        14.619

    """

    def integrand_european_option(x):
        return  norm.pdf(x) * payoff(S(S_0, r, sigma, T, x))

    # Interval with appriximately 1.0 minuns machine epsilon of probability.
    x_minus_infinity,  x_plus_infinity = (-10.0, 10.0)
         
    return (
        np.exp(-r*T)
        *quad(integrand_european_option, x_minus_infinity, x_plus_infinity) [0]
    )

#%% 

def sensitivities_european_option(
    S_0: float,
    r: float,
    T: float,
    sigma: float,
    payoff: Callable[[float], float],
) -> Dict[str, float]:
    """ Compute the sensitivities of a European option over a single underlying.

    Args:
        S_0: Value of the underlying at :math:`t_0`.
        r: Risk-free interest rate.
        T: Time to maturity :math:`(t_0+T)`.
        sigma: Volatility of the underlying.
        payoff: Function that computes the payoff of the option.

    Returns:
        Sensitivities of the European option.

    Examples:
        
        >>> S_0, K, r, T, sigma = 100.0, 105.0, 0.05, 2.5, 0.30
        >>> payoff_call = lambda S_T: np.maximum(S_T - K, 0.0) 
        >>> sensitivities = sensitivities_european_option(
        ...    S_0, r, T, sigma, payoff_call)
        >>> _ = [print('{} = {}'.format(name, sensitivity))
        ...         for name, sensitivity in sensitivities.items()] 
        delta = 0.6546106798666075
        gamma = 0.007770167768228474
        rho = 108.7762348817023
        theta = -5.672076878227018
        vega = 58.27584766156709
    """
    sensitivities = dict.fromkeys(('delta', 'gamma', 'rho', 'theta', 'vega'))
   
    price = lambda S_0: price_european_option(S_0, r, T, sigma, payoff)
    sensitivities['delta'] = numerical_derivative(price, S_0)
    sensitivities['gamma'] = numerical_second_derivative(price, S_0)
    
    interest = lambda r: price_european_option(S_0, r, T, sigma, payoff)
    sensitivities['rho'] = numerical_derivative(interest, r)
    
    time = lambda T: price_european_option(S_0, r, T, sigma, payoff)
    sensitivities['theta'] = - numerical_derivative(time, T)
    
    volatility = lambda sigma: price_european_option(S_0, r, T, sigma, payoff)
    sensitivities['vega'] = numerical_derivative(volatility, sigma)
    
    return sensitivities

#%% 

def implied_volatility(
    price: Union[float, np.ndarray],
    pricing_function: Callable[[float], float], 
    vega: Callable[[float], float], #está ya programada
    sigma_seed: float = 0.3,
) -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]:
    """ Compute the impliedy volatility from the price of an option.

    Employs Newton-Rapson method to find the zero of the function 
    f(sigma) = pricing_function(sigma) - price.        
    
    The function is vectorized on price.

    Args:
        price: Price(s) of the option(s). 
        pricing_function: Function that computes the price given sigma 
        vega: Callable[[float], float], 
        sigma_seed: Seed for the Newton-Raphson algorithm,

    Returns:
        Implied volatility (or volatilities).   

    Examples:

        Implied volatility in the price of a European call option.

        >>> S_0, K, r, T, sigma = 100.0, 105.0, 0.05, 2.5, 0.30
        >>> price = 14.0
        >>> pricing_function_EU_call = lambda sigma: (
        ...    price_european_vanilla_option(S_0, K, r, T, sigma)[0])
        >>> vega_EU_call = lambda sigma: (
        ...    sensitivities_european_call_option(S_0, K, r, T, sigma)['vega'])
        >>> implied_sigma, error = implied_volatility(
        ...    price, pricing_function_EU_call, vega_EU_call)
        >>> _ = print('implied volatility = {} ({})'.format(sigma, error))               
        implied volatility = 0.16348244635274214 (2.917402792329157e-11)
    
        
        >>> print(
        ...    price,
        ...    price_european_vanilla_option(S_0, K, r, T, implied_sigma)[0],
        ... )
        [14.0, 21.9568] [14.     21.9568]
    """
    
    

    TOL_ABS = 1.0e-6
    MAX_ITERS = 50
    
    delta_sigma = 2.0 * TOL_ABS   

    sigma = sigma_seed #* np.ones_like(price)
    n_iters = 0
    while (np.any(np.abs(delta_sigma) > TOL_ABS)) and (n_iters < MAX_ITERS):
        n_iters += 1
        delta_sigma = (pricing_function(sigma) - price)/vega(sigma)
        sigma = sigma - delta_sigma

     
    return sigma, delta_sigma

#%% 

def price_european_option_MC(
    S_0: float,
    r: float,
    T: float,
    sigma: float,
    payoff: Callable[[float], float],
    n_simulations: Union[int, None] = int(1e6),
    X: np.ndarray = None,
    seed: Union[int, None] = None,
    antithetic: bool = False,
) -> Tuple[float, float]:
    """ Monte Carlo pricing of a European option.

    Args:
        S_0: Value of the underlying at :math:`t_0`.
        r: Risk-free interest rate.
        T: Time to maturity :math:`(t_0+T)`.
        sigma: Volatility of the underlying.
        payoff: Function that computes the payoff at maturity.        
        n_simulations: Number of trajectories. If None, X must be given.
        X: one-dimensional array of N(0,1) iidrv's.
        seed: Seed of the random number generator (for reproducibility).
        antithetic: If True, antithetic variables variance reduction is applied.
        
    Returns:
        Monte Carlo estimate of the price of a European option
        over a single underlying and standard deviation of this estimate.

    Examples:

        MC pricing of the underlying.

        >>> S_0, r, T, sigma = 100.0, 0.05, 2.5, 0.30
        >>> payoff = lambda S_T: S_T
        >>> price_MC, stdev_MC = price_european_option_MC(
        ...     S_0, r, T, sigma, payoff, n_simulations=int(1e5), seed=0)
        >>> print('price (MC) = {:.3f} ({:.3f})'.format(price_MC, stdev_MC))
        price (MC) = 99.957 (0.159)

        MC pricing of a European call option.
        
        >>> S_0, K, r, T, sigma = 100.0, 105.0, 0.05, 2.5, 0.30
        >>> payoff = lambda S_T: np.maximum(S_T - K, 0.0)
        >>> price_MC, stdev_MC = price_european_option_MC(
        ...     S_0, r, T, sigma, payoff, n_simulations=int(1e5), seed=0)
        >>> print('price (MC) = {:.3f} ({:.3f})'.format(price_MC, stdev_MC)) 
        price (MC) = 21.929 (0.124)
        
        MC pricing of a European call option (antithetic variance reduction).
        
        >>> S_0, K, r, T, sigma = 100.0, 105.0, 0.05, 2.5, 0.30
        >>> payoff = lambda S_T: np.maximum(S_T - K, 0.0)
        >>> price_MC, stdev_MC = price_european_option_MC(
        ...    S_0, r, T, sigma, payoff, n_simulations=int(1e5), seed=0, 
        ...    antithetic=True)
        >>> print('price (MC) = {:.3f} ({:.3f})'.format(price_MC, stdev_MC)) 
        price (MC) = 21.967 (0.073)
        
        MC pricing of a European put option.

        >>> S_0, K, r, T, sigma = 100.0, 105.0, 0.05, 2.5, 0.30
        >>> payoff = lambda S_T: np.maximum(K - S_T, 0.0)
        >>> price_MC, stdev_MC = price_european_option_MC(
        ...    S_0, r, T, sigma, payoff, n_simulations=int(1e5), seed=0)
        >>> print('price (MC) = {:.3f} ({:.3f})'.format(price_MC, stdev_MC))
        price (MC) = 14.635 (0.058)
        
        MC price of a digital call option.

        >>> S_0, K, r, T, sigma = 100.0, 95.0, 0.05, 2.5, 0.30
        >>> A = 10.0
        >>> payoff = lambda S_T: A * (S_T > K)
        >>> price_MC, stdev_MC = price_european_option_MC(
        ...    S_0, r, T, sigma, payoff, n_simulations=int(1e5), seed=0)
        >>> print('price (MC) = {:.3f} ({:.3f})'.format(price_MC, stdev_MC)) 
        price (MC) = 4.880 (0.014)
        
    """         
    if X is None:
        rng = default_rng(seed)
        X = rng.standard_normal(n_simulations)

    payoff_MC = payoff(S(S_0, r, sigma, T, X))
    discount_factor = np.exp(-r * T)

    if antithetic:
        payoff_MC_minus = payoff(S(S_0, r, sigma, T, -X))
        payoff_MC = 0.5 * (payoff_MC + payoff_MC_minus)
        
    
    price_MC = discount_factor * np.mean(payoff_MC)

    stdev_MC = (discount_factor 
                * np.std(payoff_MC) 
                / np.sqrt(n_simulations)
    )

    return (price_MC, stdev_MC)

#%% 

def sensitivities_european_option_MC(
    S_0: float,
    r: float,
    T: float,
    sigma: float,
    payoff: Callable[[float], float],      
    n_simulations: Union[int, None] = int(1e6),
    seed: Union[int, None] = None,
) -> Dict[str, Tuple[float, float]]:
    """ Sensitivities of a European option by Monte Carlo (MC).

    Args:
        S_0: Value of the underlying at :math:`t_0`.
        r: Risk-free interest rate.
        T: Time to maturity :math:`(t_0+T)`.
        sigma: Volatility of the underlying.
        payoff: Function that computes the payoff at maturity.        
        n_simulations: Number of trajectories. If None, X must be given.
        seed: Seed of the random number generator (for reproducibility).

    Returns:
        Monte Carlo estimates of the sensitivities of a European option over 
        a single underlying and of the standard deviation of these estimates.
        Common random numbers are used to reduce sampling errors.

    Examples:

        >>> S_0, K, r, T, sigma = 100.0, 105.0, 0.05, 2.5, 0.30
        >>> payoff = lambda S_T: np.maximum(S_T - K, 0.0) 
        >>> sensitivities_MC = sensitivities_european_option_MC(
        ...        S_0, r, T, sigma, payoff, n_simulations=int(1e6), seed=0) 
        >>> _ = [print('{} = {:.4f} ({:.4f})'.format(
        ...    name, sensitivity[0], sensitivity[1]))
        ...    for name, sensitivity in sensitivities_MC.items()] 
        delta = 0.6555 (0.0008)
        gamma = 0.0076 (0.0002)
        rho = 108.8526 (0.1156)
        theta = -5.6854 (0.0112)
        vega = 58.4731 (0.1684)

    """

    # Use common random numbers (CRN)
    
    rng = default_rng(seed)
    X = rng.standard_normal(n_simulations)
        
    reduction_factor = np.sqrt(n_simulations)
    
    def discounted_payoff(
            S_0: float, r: float, sigma: float, T:float
    ) ->  float: 
        return   np.exp(-r *T) * payoff(S(S_0, r, sigma, T, X))
                 #Esto tiene que ir dentro. Porque sino las derivadas que haces si definieses discount factor, 
                 # serían sobre una constante en vez de sobre r y T   
    def derivative_MC(
        discounted_payoff: Callable[[float], float], 
        z_0: float
    ) -> Tuple[float, float]:
        """ MC estimate of the derivative of a price and its stdev."""
        delta_z = z_0 *1.0e-6 #balance rounding an truncation errors
        derivative_trajectory = (
            (discounted_payoff(z_0 + delta_z) - discounted_payoff(z_0 - delta_z))) / (2.0 * delta_z)
        
        return(
            np.mean(derivative_trajectory),
            np.std(derivative_trajectory) / reduction_factor
        )

        
    def second_derivative_MC(
        discounted_payoff: Callable[[float], float], 
        z_0: float
    ) -> Tuple[float, float]:
        """ MC estimate of the 2nd derivative of a price and its stdev."""
        
        delta_z = z_0 *1.0e-3 #balance rounding an truncation errors
        second_derivative_trajectory = (
            (discounted_payoff(z_0 + delta_z) - 2 * discounted_payoff(z_0) + discounted_payoff(z_0 - delta_z))
            / delta_z**2
        )
        return(
            np.mean(second_derivative_trajectory),
            np.std(second_derivative_trajectory) / reduction_factor
        )




    sensitivities = dict.fromkeys(('delta', 'gamma', 'rho', 'theta', 'vega'))
        
    discounted_payoff_S_0 = lambda S_0: discounted_payoff(S_0, r, sigma, T)
    
    
    sensitivities['delta'] = derivative_MC(discounted_payoff_S_0, S_0)
    sensitivities['gamma'] = second_derivative_MC(discounted_payoff_S_0, S_0) 
    
    sensitivities['rho'] = derivative_MC(
        lambda r: discounted_payoff(S_0, r, sigma, T), r)
    
    theta = derivative_MC(lambda T: discounted_payoff(S_0, r, sigma, T), T)
    sensitivities['theta'] = (- theta[0], theta[1])

    sensitivities['vega'] = derivative_MC(
        lambda sigma: discounted_payoff(S_0, r, sigma, T), 
        sigma
    )

    return sensitivities

#%% 

def price_european_2_underlyings(
    S_0_1: float,
    sigma_1: float,
    S_0_2: float,
    sigma_2: float,
    rho: float,
    r: float,
    T: float,
    payoff: Callable[[float, float], float],
) ->  float:
    """ Price of a European option over two underlyings 

    Args:
        S_0_1: Value of the underlying 1 at :math:`t_0`.
        sigma_1: Volatility of underlying 1.
        S_0_2: Value of the underlying 2 at :math:`t_0`.
        sigma_2: Volatility of underlying 2.
        rho: Correlations between (log-returns of) underlyings 1 and 2. 
        r: Risk-free interest rate.
        T: Time to maturity :math:`(t_0+T)`.
        payoff: Function that computes the payoff at maturity.

    Returns:
        Price of a European option over two underlyings.

    Examples:
  
       Price of a European call option.

       >>> S_0_1, sigma_1 = 70.0, 0.30
       >>> S_0_2, sigma_2 = 30.0, 0.30
       >>> rho = 1.0
       >>> r, T, K = 0.05, 2.5, 105.0 
       >>> payoff = (lambda S_T_1, S_T_2: 
       ...                   np.maximum(S_T_1 + S_T_2 - K, 0.0))
       >>> price = price_european_2_underlyings(
       ...    S_0_1, sigma_1, S_0_2, sigma_2, rho, r, T, payoff)
       >>> np.round(price, 4)
       21.9568

       Price of a European call basket option.

       >>> S_0_1, sigma_1 = 100.0, 0.30
       >>> S_0_2, sigma_2 = 120.0, 0.20
       >>> rho = 0.7
       >>> c_1, c_2 = 0.7, 1.2
       >>> r, T, K = 0.05, 0.5, 230.0 
       >>> payoff = (lambda S_T_1, S_T_2: 
       ...                   np.maximum(c_1 * S_T_1 + c_2 * S_T_2 - K, 0.0))
       >>> price = price_european_2_underlyings(
       ...    S_0_1, sigma_1, S_0_2, sigma_2, rho, r, T, payoff)
       >>> np.round(price, 4)
       8.7597

    """
    
    # Interval with appriximately 1.0 minuns machine epsilon of probability.
    x_minus_infinity,  x_plus_infinity = -10.0, 10.0

    def inner_integrand(x_1, x_2):
        return 
    

    def outer_integrand(x_1):  
        return 
    
    return 

 
#%% 

def price_european_2_underlyings_MC(
    S_0_1: float,
    sigma_1: float,
    S_0_2: float,
    sigma_2: float,
    rho: float,
    r: float,
    T: float,
    payoff: Callable[[float, float], float],
    n_simulations: int = int(1e6),
    seed: Union[int, None] = None,
) -> Tuple[float, float]:
    """ Compute the price of an European option over a single underlying

    Args:
        S_0_1: Value of the underlying 1 at :math:`t_0`.
        sigma_1: Volatility of underlying 1.
        S_0_2: Value of the underlying 2 at :math:`t_0`.
        sigma_2: Volatility of underlying 2.
        rho: Correlations between (log-returns of) underlyings 1 and 2. 
        r: Risk-free interest rate.
        T: Time to maturity :math:`(t_0+T)`.
        payoff: Function that computes the payoff at maturity.
        n_simulations: Number of trajectories.
        seed: Seed of the random number generator (for reproducibility).

    Returns:
        Monte Carlo estimate of the price of a European option
        over two underlyings and standard deviation of this estimate.
     
    Examples:

        Price of a European call option.

        >>> S_0_1, sigma_1 = 70.0, 0.30
        >>> S_0_2, sigma_2 = 30.0, 0.30
        >>> rho = 1.0
        >>> r, T, K = 0.05, 2.5, 105.0 
        >>> payoff = (lambda S_T_1, S_T_2:
        ...              np.maximum(S_T_1 + S_T_2 - K, 0.0))
        >>> price_MC, stdev_MC  = price_european_2_underlyings_MC(
        ...    S_0_1, sigma_1, S_0_2, sigma_2, rho, r, T, payoff, 
        ...    n_simulations=int(1e5), seed=0)
        >>> print('price (MC) = {:.3f} ({:.3f})'.format(price_MC, stdev_MC)) 
        price (MC) = 21.929 (0.124)

        
       Price of a European call basket option.

       >>> S_0_1, sigma_1 = 100.0, 0.30
       >>> S_0_2, sigma_2 = 120.0, 0.20
       >>> rho = 0.7
       >>> c_1, c_2 = 0.7, 1.2
       >>> r, T, K = 0.05, 0.5, 230.0 
       >>> payoff = (lambda S_T_1, S_T_2: 
       ...                   np.maximum(c_1 * S_T_1 + c_2 * S_T_2 - K, 0.0))
       >>> price_MC, stdev_MC  = price_european_2_underlyings_MC(
       ...    S_0_1, sigma_1, S_0_2, sigma_2, rho, r, T, payoff, 
       ...    n_simulations=int(1.0e5), seed=0) 
       >>> print('price (MC) = {:.3f} ({:.3f})'.format(price_MC, stdev_MC)) 
       price (MC) = 8.755 (0.055)
    
"""
    rng = default_rng(seed)
    
    X_1 = rng.standard_normal(n_simulations)
    X_2 = rng.standard_normal(n_simulations)

    Z_1 = X_1
    Z_2 = rho * X_1 + np.sqrt(1 - rho**2)*X_2


    payoff_MC = payoff(S(S_0_1, r, sigma_1, T, Z_1), 
                       S(S_0_2, r, sigma_2, T, Z_2)
    )
    
    discount_factor = np.exp(-r * T)
 
    price_MC = discount_factor * np.mean(payoff_MC)

    stdev_MC = (discount_factor 
                * np.std(payoff_MC) 
                / np.sqrt(n_simulations)
    )
          
    
    return (price_MC, stdev_MC)

#%%
def price_european_3_underlyings_MC(
    S_0_1: float,
    sigma_1: float,
    S_0_2: float,
    sigma_2: float,
    S_0_3: float,
    sigma_3: float,
    rho: np.ndarray,
    r: float,
    T: float,
    payoff: Callable[[float, float, float], float],
    n_simulations: int = int(1e6),
    seed: Union[int, None] = None,
) -> Tuple[float, float]:

    rng = default_rng(seed)
    
    X = rng.standard_normal((3, n_simulations))


    Z = np.dot(np.linalg.cholesky(rho), X)


    payoff_MC = payoff(S(S_0_1, r, sigma_1, T, Z[0]), 
                       S(S_0_2, r, sigma_2, T, Z[1]),
                       S(S_0_3, r, sigma_3, T, Z[2])
    )
    
    discount_factor = np.exp(-r * T)
 
    price_MC = discount_factor * np.mean(payoff_MC)

    stdev_MC = (discount_factor 
                * np.std(payoff_MC) 
                / np.sqrt(n_simulations)
    )
          
    
    return (price_MC, stdev_MC)

#%%

def price_geometric_mean_call_option(
    S_0: float,
    K: float,
    r: float,
    T: float,
    sigma: float,
    N: int,
):
    """ Price of a call option over the geometric mean.
   
    Args:
        S_0: Value of the underlying at :math:`t_0`.
        K: Strike.
        r: Risk-free interest rate.
        T: Time to maturity :math:`(t_0+T)`.
        sigma: Volatility of the underlying.
        N: Number of equally-spaced observations for the geometric mean.

    Returns:
        Price of a call option over the geometric mean.
        
    Example:

        >>> S_0, K, r, T, sigma = (100.0, 105.0, 0.05, 2.5, 0.3)
        >>> N = int(12 * T)
        >>> price = price_geometric_mean_call_option(S_0, K, r, T, sigma, N)
        >>> print(price)
        10.192149700300735
        
    """
    discount_factor = np.exp(- r * T)
    
    sigma_eff = sigma * np.sqrt((2.0 * N**2 + 3.0 * N + 1.0) / (6.0 * N**2))
    r_eff = 0.5 * (r * (N + 1.0) / N - sigma**2 / 6.0 * (1.0 - 1.0 / N**2))
    
    capitalization_factor_eff = np.exp(r_eff * T)

    total_volatility = sigma_eff * np.sqrt(T)
    discounted_K = K / capitalization_factor_eff

    d_plus = (np.log(S_0 / discounted_K) / total_volatility
              + 0.5 * total_volatility)
    d_minus = d_plus - total_volatility

    normcdf_d_plus = norm.cdf(d_plus)
    normcdf_d_minus = norm.cdf(d_minus)

    price_call = discount_factor * ( 
        S_0 * capitalization_factor_eff * normcdf_d_plus - K * normcdf_d_minus
    )
   
    return price_call

 
#%%

def price_exotic_option_MC(
    S_0: float,
    r: float,
    times: np.ndarray,
    sigma: float,
    payoff: Callable[[float], float],
    n_simulations: Union[int, None] = int(1e6),
    seed: Union[int, None] = None,
) -> Tuple[float, float]:
    """ Monte Carlo pricing of an exotic option (single payoff at maturity).

    Args:
        S_0: Value of the underlying at :math:`t_0`.
        r: Risk-free interest rate.
        times: Monitoring times (times[0]: initial time; times[-1]: maturity).
        sigma: Volatility of the underlying.
        payoff: Function that computes the payoff at maturity.        
        n_simulations: Number of trajectories. If None, X must be given.
        seed: Seed of the random number generator (for reproducibility).
        
    Returns:
        Monte Carlo estimate of the price and standard deviation of an 
        exotic option over a single underlying with a single payoff at maturity.

    Examples:

        Price of the underlying.

        >>> S_0, r, T, sigma = 100.0, 0.05, 2.5, 0.30
        >>> t_0 = 1.5
        >>> n_timesteps = int(12*T) # monthly monitoring
        >>> times = np.linspace(t_0, t_0 + T, num=n_timesteps + 1) 
        >>> payoff = lambda S_T: S_T[:, -1]
        >>> price_MC, stdev_MC = price_exotic_option_MC(
        ...    S_0, r, times, sigma, payoff, n_simulations=int(1e5), seed=0) 
        >>> print('price (MC) = {:.3f} ({:.3f})'.format(price_MC, stdev_MC)) 
        price (MC) = 100.077 (0.159)
   
        Price of an Asian call option on the arithmetic mean.

        >>> S_0, K, r, T, sigma = 100.0, 105.0, 0.05, 2.5, 0.30
        >>> t_0 = 1.5
        >>> n_timesteps = int(12*T) # monthly monitoring
        >>> times = np.linspace(t_0, t_0 + T, num=n_timesteps + 1) 
        >>> def payoff(S_T):
        ...     return np.maximum(np.mean(S_T[:, 1:], axis=1) - K, 0.0)
        >>> price_MC, stdev_MC = price_exotic_option_MC(
        ...    S_0, r, times, sigma, payoff, n_simulations=int(1e5), seed=0)
        >>> print('price (MC) = {:.3f} ({:.3f})'.format(price_MC, stdev_MC))
        price (MC) = 11.362 (0.062)
        
        Price of an Asian call option on the geometric mean.

        >>> S_0, K, r, T, sigma = 100.0, 105.0, 0.05, 2.5, 0.30
        >>> t_0 = 1.5
        >>> n_timesteps = int(12*T) # monthly monitoring
        >>> times = np.linspace(t_0, t_0 + T, num=n_timesteps + 1) 
        >>> def payoff(S_T):
        ...        return np.maximum(
        ...            np.exp(np.mean(np.log(S_T[:, 1:]), axis=1)) - K, 0.0)
        >>> price_MC, stdev_MC = price_exotic_option_MC(
        ...    S_0, r, times, sigma, payoff, n_simulations=int(1e5), seed=0) 
        >>> print('price (MC) = {:.3f} ({:.3f})'.format(price_MC, stdev_MC))
        price (MC) = 10.222 (0.056)

        Price of an up-and-out call option.

        >>> S_0, K, B, r, T, sigma = 100.0, 105.0, 180.0, 0.05, 2.5, 0.30
        >>> t_0 = 1.5
        >>> n_timesteps = int(12*T) # monthly monitoring
        >>> times = np.linspace(t_0, t_0 + T, num=n_timesteps + 1) 
        >>> def payoff(S_T):
        ...        return (np.all(S_T < B, axis=1) 
        ...                * np.maximum(S_T[:, -1] - K, 0.0))
        >>> price_MC, stdev_MC = price_exotic_option_MC(
        ...    S_0, r, times, sigma, payoff, n_simulations=int(1e5), seed=0) 
        >>> print('price (MC) = {:.3f} ({:.3f})'.format(price_MC, stdev_MC))
        price (MC) = 6.543 (0.043)

    """
    
    S = simulate_geometric_brownian_motion(
        times, S_0, r, sigma, n_simulations, seed
    )
    
    
    return (price_MC, stdev_MC)


#%%
def pricing_control_variate_MC(
    S_0: float,
    r: float,
    times: np.ndarray,
    sigma: float,
    payoff: Callable[[float], float],
    payoff_control: Callable[[float], float],
    price_control: Callable[[float, float, float, float], float],
    n_simulations: Union[int, None] = int(1e6),
    seed: Union[int, None] = None,
) -> Tuple[float, float]:
    """
    Args:
        S_0: Value of the underlying at :math:`t_0`.
        r: Risk-free interest rate.
        times: Monitoring times (times[0]: initial time; times[-1]: maturity).
        sigma: Volatility of the underlying.
        payoff: Function that computes the payoff at maturity.        
        n_simulations: Number of trajectories. If None, X must be given.
        seed: Seed of the random number generator (for reproducibility).
        
    Returns:
        Monte Carlo estimate of the price and standard deviation of an 
        exotic option over a single underlying with a single payoff at maturity.

    Examples:

        Price of an Asian call option on the arithmetic mean using a
        call option on the geometric mean as control variate.

        >>> S_0, K, r, T, sigma = 100.0, 105.0, 0.05, 2.5, 0.30
        >>> t_0 = 1.5
        >>> n_timesteps = int(12*T) # monthly monitoring
        >>> times = np.linspace(t_0, t_0 + T, num=n_timesteps + 1) 
        >>> def payoff(S_T):
        ...     return np.maximum(np.mean(S_T[:, 1:], axis=1) - K, 0.0)
        >>> def payoff_control(S_T):
        ...     from scipy.stats import gmean
        ...     return np.maximum(gmean(S_T[:, 1:], axis=1) - K, 0.0)
        >>> def price_control(S_0, r, T, sigma): 
        ...     return price_geometric_mean_call_option(
        ...         S_0, K, r, T, sigma, n_timesteps)
        >>> price_MC, stdev_MC = pricing_control_variate_MC(
        ...     S_0, r, times, sigma, payoff, payoff_control, price_control, 
        ...     n_simulations=int(1e5), seed=0)
        >>> print('price (MC) = {:.4f} ({:.4f})'.format(price_MC, stdev_MC))
        price (MC) = 11.3283 (0.0044)
        
    """
    S = simulate_geometric_brownian_motion(
        times, S_0, r, sigma, n_simulations, seed
    )
    
    T = times[-1] - times[0]
    discount_factor = np.exp(- r * T)


    return (price_MC_corrected, stdev_MC * np.sqrt(1.0 - rho_square))

#%%

# Run examples and test results

if __name__ == "__main__":
    import doctest
    doctest.testmod()
