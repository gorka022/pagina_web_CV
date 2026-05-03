# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from numpy.random import default_rng
from typing import Callable, Union

#%%

def simulate_arithmetic_brownian_motion(
    times: float, 
    B_0: float, 
    mu: float, 
    sigma: float, 
    n_trajectories: int,
    seed: Union[int, None] = None,
) -> np.ndarray:
    """ Simulation of arithmetic Brownian motion.

        SDE:    dB(t) = mu*dt + sigma*dW(t)
    
    Args:
        times: Integration (monitoring) grid (measurement times).
        B_0: Initial level of the process.
        mu: Location parameter of the process.
        sigma: Scale parameter of the process.
        n_trajectories: Number of simulated trajectories.
        seed: Seed of the random number generator (for reproducibility).
        
    Returns:
        Simulation consisting of n_trajectories trajectories.
        Each trajectory is a row vector composed of the process values at t.      

    Example:
        >>> times = np.linspace(2.0, 10.0, num=100)
        >>> B_0, mu, sigma = 10.0, 0.5, 0.4
        >>> B = simulate_arithmetic_brownian_motion(
        ...    times, B_0, mu, sigma, n_trajectories=50)
        >>> fig, ax = plt.subplots() 
        >>> _ = ax.plot(times, B.T)
        >>> _ = ax.set_xlabel('t')
        >>> _ = ax.set_ylabel('B(t)') 
        >>> _ = ax.set_title('Arithmetic Brownian motion in 1D')

    """
    delta_T = np.diff(times)  # integration intervals
    n_times = len(times)
  
    rng = default_rng(seed)
    X = rng.standard_normal((n_trajectories, n_times - 1))

    B = np.empty((n_trajectories, n_times))
    
    B[:, 0] = B_0
    
    B[:, 1:] = mu*delta_T + sigma * np.sqrt(delta_T) * X
    B = np.cumsum(B, axis=1)

    return B

#%%

def simulate_geometric_brownian_motion(
    times: float, 
    S_0: float, 
    mu: float, 
    sigma: float, 
    n_trajectories: int,
    seed: Union[int, None] = None,
) -> np.ndarray:
    """ Simulation of geometric Brownian motion.

        SDE:    dS(t) = mu*S(t)*dt + sigma*S(t)*dW(t)
    
    Args:
        times: Integration (monitoring) grid (measurement times).
        S_0: Initial level of the process.
        mu: Location parameter of the process.
        sigma: Scale parameter of the process.
        n_trajectories: Number of simulated trajectories.
        seed: Seed of the random number generator (for reproducibility).
            
    Returns:
        Simulation consisting of n_trajectories trajectories.
        Each trajectory is a row vector composed of the process values at t.      

    Example:
        
        >>> times = np.linspace(2.0, 10.0, num=100)
        >>> S_0, mu, sigma = 100.0, 0.1, 0.2
        >>> S = simulate_geometric_brownian_motion(
        ...    times, S_0, mu, sigma, n_trajectories=50)
        >>> fig, ax = plt.subplots() 
        >>> _ = ax.plot(times, S.T)
        >>> _ = ax.set_xlabel('t')
        >>> _ = ax.set_ylabel('S(t)') 
        >>> _ = ax.set_title('Geometric Brownian motion in 1D')
    
    """
    delta_T = np.diff(times)  # integration intervals

    n_times = len(times)
   
    rng = default_rng(seed)
    X = rng.standard_normal((n_trajectories, n_times - 1))
    
    S = np.empty((n_trajectories, n_times))
    S[:, 0] = S_0    
    S[:, 1:] = np.exp((mu - 0.5 * sigma**2) * delta_T 
                      + sigma * np.sqrt(delta_T) * X)
    S = np.cumprod(S, axis=1)
    
    return S

#%%

def simulate_Brownian_bridge(
    times: float, 
    BB_0: float, 
    BB_T: float, 
    sigma: float, 
    n_trajectories: int,
    seed: Union[int, None] = None,    
) -> np.ndarray:
    """ Simulation of a Brownian bridge process
    
    Args:
        t: Integration (monitoring) grid (measurement points).
        BB_0: Initial level of the process.
        BB_T: Final level of the process.
        sigma: Scale parameter of the process.
        n_trajectories: Number of simulated trajectories.
        seed: Seed of the random number generator (for reproducibility).
    
    Returns:
        Simulation consisting of M trajectories.
        Each trajectory is a row vector composed of the process values at t. 

    Example:
        
        >>> times = np.linspace(2.0, 10.0, num=100)
        >>> BB_0, BB_T, sigma = 10.0, 12.0, 0.4
        >>> BB = simulate_Brownian_bridge(
        ...    times, BB_0, BB_T, sigma, n_trajectories=50)
        >>> fig, ax = plt.subplots() 
        >>> _ = ax.plot(times, BB.T)
        >>> _ = ax.set_xlabel('t')
        >>> _ = ax.set_ylabel('BB(t)') 
        >>> _ = ax.set_title('Brownian bridge')

    """       
    
    B = simulate_arithmetic_brownian_motion(
        times, 
        BB_0, mu=0.0, sigma=sigma, 
        n_trajectories=n_trajectories, seed=seed
    )
    
    BB = B + (
        np.multiply.outer(BB_T - B[:,-1], 
                          (times - times[0]) / (times[-1] - times[0]))
    )
    
    return BB

#%%

def simulate_Ornstein_Uhlenbeck_process(
    times: np.ndarray, 
    OU_process_0: float, 
    k: float,  
    D: float, 
    n_trajectories: int,
    seed: Union[int, None] = None,
) -> np.ndarray:
    """ Simulation of Ornstein-Uhlenbeck process trajectories
    
        SDE:    dX(t) = - k * X(t) * dt + sqrt(D)*dW(t)
        
    Args:
       times: Integration (monitoring) grid (measurement points).
       OU_process_0: Initial level of the process
       k: Rate or inverse correlation time of the process
       D: Diffusion constant.
       n_trajectories: Number of simulated trajectories.
       seed: Seed of the random number generator (for reproducibility).
        
    Returns:
        Simulation consisting of n_trajectories trajectories.
        Each trajectory is a row vector composed of the process values at t.      
    
    Example:
        
        >>> times = np.linspace(2.0, 10.0, num=100)
        >>> OU_process_0, k, D = (10.0, 0.5, 0.4)
        >>> OU_process = simulate_Ornstein_Uhlenbeck_process(
        ...    times, OU_process_0, k, D, n_trajectories=50)
        >>> fig, ax = plt.subplots() 
        >>> _ = ax.plot(times, OU_process.T)
        >>> _ = ax.set_xlabel('t')
        >>> _ = ax.set_ylabel('OU(t)') 
        >>> _ = ax.set_title('Ornstein-Uhlenbeck process')

    """


    # TO DO: return OU_process

#%% 

def simulate_SDE_euler_maruyana(
    times: np.ndarray,  
    S_0: float,
    drift_term: Callable[[float, float], float],
    diffusion_term: Callable[[float, float], float], 
    n_trajectories: int,
    seed: Union[int, None] = None,
) -> np.ndarray:
    r""" Integration of an Îto SDE with the Euler-Maruyana method (order 1/2).

        SDE:  dS(t) = drif_term(times, S(t))*dt 
                      + diffusion_term(times, S(t))*dW(t)
    
    Args:
        times: Integration (monitoring) grid (measurement times).
        S_0: Initial level of the process.
        drift_term: Drift (deterministic, advection) term: a(times, S(t)).
        diffusion_term: Diffusion (stochastic) term: b(times, S(t)).
        n_trajectories: Number of simulated trajectories.
        seed: Seed of the random number generator (for reproducibility).
        
    Returns:
        Simulation consisting of n_trajectories trajectories.
        Each trajectory is a row vector composed of the process values at t.      

    Example:

        Black-Scholes model (geometric Brownian motion)
        
        >>> times = np.linspace(2.0, 10.0, num=100)
        >>> S_0, mu, sigma = 100.0, 0.15, 0.4
        >>> drift_term = lambda t, S_t: mu * S_t
        >>> difussion_term = lambda t, S_t: sigma * S_t
        >>> S = simulate_SDE_euler_maruyana(
        ...     times, S_0, drift_term, difussion_term, n_trajectories=50)
        >>> fig, ax = plt.subplots() 
        >>> _ = ax.plot(times, S.T)
        >>> _ = ax.set_xlabel('t')
        >>> _ = ax.set_ylabel('S(t)') 
        >>> _ = ax.set_title('Black-Scholes model (Euler-Maruyana sumulation') 

        Stochastic volatility model with reversion to the mean

        >>> times = np.linspace(2.0, 12.0, num=100)
        >>> sigma_0, sigma_infty, alpha, xi = (0.5, 0.2, 0.5, 0.1)
        >>> def drift_term(t, sigma_t): 
        ...    return - alpha * (sigma_t - sigma_infty)
        >>> difussion_term = lambda t, sigma_t: xi * sigma_t
        >>> sigma = simulate_SDE_euler_maruyana(
        ...     times, sigma_0, drift_term, difussion_term, n_trajectories=50)
        >>> fig, ax = plt.subplots() 
        >>> _ = ax.plot(times, sigma.T)
        >>> _ = ax.set_xlabel('t')
        >>> _ = ax.set_ylabel(r'$\sigma(t)$') 
        >>> _ = ax.set_title('Stochastic volatility model\n'  
        ...                      + 'with reversion to the mean') 
        
    """
      
    delta_T = np.diff(times)  # integration intervals
    n_times = len(times)

    S = np.empty((n_trajectories, n_times))
    
    rng = default_rng(seed)
    X = rng.standard_normal((n_trajectories, n_times - 1))

    S[:, 0] = S_0
    
    
    # TO DO: Integrate the SDE using the Euler-Maruyama integration scheme
    
    # return S


#%%

def simulate_SDE_milstein(
    times: np.ndarray,  
    S_0: float,
    drift_term: Callable[[float, float], float],
    diffusion_term: Callable[[float, float], float], 
    diffusion_term_derivative: Callable[[float, float], float], 
    n_trajectories: int,
    seed: Union[int, None] = None,
) -> np.ndarray:
    r""" Integration of an Îto SDE with the Milstein method (order 1).

        SDE:  dS(t) = drif_term(times, S(t))*dt 
                      + diffusion_term(times, S(t))*dW(t)
    
    Args:
        times: Integration (monitoring) grid (measurement times).
        S_0: Initial level of the process.
        drift_term: Drift (deterministic, advection) term: a(times, S(t)).
        diffusion_term: Diffusion (stochastic) term: b(times, S(t)).
        n_trajectories: Number of simulated trajectories.
        seed: Seed of the random number generator (for reproducibility).
        
    Returns:
        Simulation consisting of n_trajectories trajectories.
        Each trajectory is a row vector composed of the process values at t.      

    Example:

        Black-Scholes model (geometric Brownian motion)
        
        >>> times = np.linspace(2.0, 10.0, num=100)
        >>> S_0, mu, sigma = 100.0, 0.15, 0.4
        >>> drift_term = lambda t, S_t: mu * S_t
        >>> difussion_term = lambda t, S_t: sigma * S_t
        >>> difussion_term_derivative = lambda t, S_t: sigma
        >>> S = simulate_SDE_milstein(times, S_0, 
        ...     drift_term, difussion_term, difussion_term_derivative, 
        ...     n_trajectories=50)
        >>> fig, ax = plt.subplots() 
        >>> _ = ax.plot(times, S.T)
        >>> _ = ax.set_xlabel('t')
        >>> _ = ax.set_ylabel('S(t)') 
        >>> _ = ax.set_title('Black-Scholes model (Milstein simulation') 

        Stochastic volatility model with reversion to the mean

        >>> times = np.linspace(2.0, 12.0, num=100)
        >>> sigma_0, sigma_infty, alpha, xi = (0.5, 0.2, 0.5, 0.1)
        >>> def drift_term(t, sigma_t): 
        ...    return - alpha * (sigma_t - sigma_infty)
        >>> difussion_term = lambda t, sigma_t: xi * sigma_t
        >>> difussion_term_derivative = lambda t, sigma_t: xi
        >>> sigma = simulate_SDE_milstein(times, sigma_0, 
        ...     drift_term, difussion_term, difussion_term_derivative, 
        ...     n_trajectories=50)
        >>> fig, ax = plt.subplots() 
        >>> _ = ax.plot(times, sigma.T)
        >>> _ = ax.set_xlabel('t')
        >>> _ = ax.set_ylabel(r'$\sigma(t)$') 
        >>> _ = ax.set_title('Stochastic volatility model\n'  
        ...                      + 'with reversion to the mean') 
        
    """
    
   

#%%
# Run examples and test results


if __name__ == "__main__":
    import doctest
    doctest.testmod()
