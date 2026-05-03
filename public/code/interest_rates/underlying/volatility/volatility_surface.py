import pandas as pd
import numpy as np
from scipy.interpolate import RegularGridInterpolator


class VolatilitySurface:
    """
    Superficie de volatilidad 2D (vencimiento × strike) con interpolación bilineal.

    Construida a partir de un DataFrame con:
    - Índice: vencimientos en años (e.g., 0.5, 1.0, 2.0, ...)
    - Columnas: strikes en valor absoluto (e.g., 0.01, 0.015, 0.02, ...)
    - Valores: volatilidades (ya divididas por 10,000 si vienen en bps)

    Parameters
    ----------
    df_volatility : pd.DataFrame
        DataFrame con la superficie de volatilidad.
    method : str
        Método de interpolación: 'linear' o 'nearest' (default: 'linear').
    """

    def __init__(self, df_volatility: pd.DataFrame, method: str = 'linear'):

        # Extraemos ejes
        self.df_volatility = df_volatility
        self.maturities = self.df_volatility.index.values.astype(float)
        self.strikes = self.df_volatility.columns.values.astype(float)
        self.vol_values = self.df_volatility.values

        # Creamos el interpolador 2D
        self.interpolator = RegularGridInterpolator(
            (self.maturities, self.strikes),
            self.vol_values,
            method=method,
            bounds_error=False,
            fill_value=None
        )

    def get_volatility(self, expiry_years: float, strike: float) -> float:
        """
        Obtiene la volatilidad interpolada para un vencimiento y strike dados.

        Parameters
        ----------
        expiry_years : float
            Tiempo hasta vencimiento en años.
        strike : float
            Strike del caplet/floorlet.

        Returns
        -------
        float
            Volatilidad interpolada.
        """
        vol = self.interpolator((expiry_years, strike))

        return float(vol)