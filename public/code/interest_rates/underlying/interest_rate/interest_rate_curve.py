import numpy as np
from datetime import datetime
from typing import List
from scipy.interpolate import interp1d

from basics.day_counter import DayCounter


class InterestRateCurve:
    """
    Curva de tipos de interés construida a partir de discount factors observados.

    Soporta interpolación lineal y log-linear (lineal en log de discount factors,
    que preserva la positividad de los tipos forward implícitos).

    Parameters
    ----------
    start_date : datetime
        Fecha de valoración (t=0, DF=1.0).
    end_dates : List[datetime]
        Fechas de los nodos de la curva.
    discount_factors : List[float]
        Discount factors correspondientes a cada nodo.
    interpolation : str
        Método de interpolación: 'linear' o 'log-linear'.
    day_count : str
        Convención de conteo de días para calcular las fracciones de año.
    """

    def __init__(self, start_date: datetime,
                 end_dates: List[datetime],
                 discount_factors: List[float],
                 interpolation: str,
                 day_count: str):

        self.start_date = start_date
        self.end_dates = end_dates
        self.discount_factors = discount_factors
        self.interpolation = interpolation
        self.day_count = day_count

        self.delta_time = [
            DayCounter.year_fraction(self.day_count, self.start_date, date)
            for date in end_dates
        ]

        #  LOG-LINEAR: interpolamos en el espacio log(DF)
        if self.interpolation == 'log-linear':
            y_values = np.log(self.discount_factors)
            kind_to_use = 'linear'

        else:
            y_values = self.discount_factors
            kind_to_use = self.interpolation


        self.interpolador = interp1d(
            x=self.delta_time,
            y=y_values,
            kind = kind_to_use)


    def interpolate(self, date: datetime) -> float:
        """
        Calcula el factor de descuento interpolado para cualquier fecha.

        Parameters
        ----------
        date : datetime
            Fecha para la cual se quiere obtener el discount factor.

        Returns
        -------
        float
            Discount factor interpolado.
        """
        delta = DayCounter.year_fraction(self.day_count, self.start_date, date)
        valor_interp = self.interpolador(delta)

        #  LOG-LINEAR: deshacemos el log
        if self.interpolation == 'log-linear':
            return float(np.exp(valor_interp))

        else:
            return float(valor_interp)


    def forward_rate(self, start_date: datetime, end_date: datetime, day_count_basis: str = 'Act/360') -> float:
        """
        Calcula el tipo de interés forward simple entre dos fechas futuras.

        F(t1, t2) = (DF(t1) / DF(t2) - 1) / τ(t1, t2)

        Parameters
        ----------
        start_date : datetime
            Fecha inicio del periodo forward.
        end_date : datetime
            Fecha fin del periodo forward.
        day_count_basis : str
            Convención de conteo de días para τ.

        Returns
        -------
        float
            Tipo forward simple.
        """
        discount_f_start = self.interpolate(start_date)
        discount_f_end = self.interpolate(end_date)

        tau = DayCounter.year_fraction(day_count_basis, start_date, end_date)

        if tau == 0:
            return 0.0

        return (discount_f_start / discount_f_end - 1) / tau
