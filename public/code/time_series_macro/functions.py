import pandas as pd
import numpy as np

def lagn(x, n):
    """Genera retardos (lags) de una serie o matriz."""
    if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
        x = x.values
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    T, V = x.shape
    out = np.full((T, V), np.nan)
    out[n:, :] = x[:-n, :]
    return out

def get_lag_matrix(data, nlags, ndet, trend_vector=None):
    """Construye la matriz X de retardos para el VAR."""
    T, nvars = data.shape
    X = np.zeros((T - nlags, nvars * nlags))

    for k in range(1, nlags + 1):
        X[:, nvars*(k-1):nvars*k] = data[(nlags-k):(T-k), :]

    ones_col = np.ones((X.shape[0], 1))
    if ndet == 1:
        X_final = np.hstack([ones_col, X])
    elif ndet == 2:
        trend_cut = trend_vector[nlags:].reshape(-1,1)
        X_final = np.hstack([ones_col, trend_cut, X])
    else:
        X_final = X

    return X_final

def compute_impulse_response(beta, impact_matrix, ndet, nlags, nsteps):
    """
    Calcula la respuesta al impulso.
    beta: Coeficientes OLS (T_lags x N_vars)
    impact_matrix: Matriz de impacto estructural (A0 o inversa de C)
    """
    nvars = beta.shape[1]
    A_coeffs = beta[ndet:, :].T  

    irf = np.zeros((nsteps, nvars, nvars))
    irf[0, :, :] = impact_matrix

    for h in range(1, nsteps):
        for v in range(nvars): 
            lags_vec = np.zeros((nvars * nlags, 1))

            for L in range(1, nlags + 1):
                idx_start = (L-1)*nvars
                idx_end = L*nvars
                if h - L >= 0:
                    lags_vec[idx_start:idx_end, 0] = irf[h-L, :, v]
                else:
                    pass 

            irf[h, :, v] = (A_coeffs @ lags_vec).flatten()

    return irf

def variance_decomposition(nvars, nsteps, Psi):
    """
    Calcula la descomposición de varianza.
    """
    mse = np.cumsum(Psi**2, axis=0) 

    total_mse = np.sum(mse, axis=2) 

    vardecm = np.zeros((nsteps * nvars, nvars)) 

    for k in range(nvars): 
        share = mse[:, :, k] / total_mse
        col_vec = []
        for v in range(nvars): 
            col_vec.append(share[:, v])
        vardecm[:, k] = np.concatenate(col_vec)

    return vardecm
