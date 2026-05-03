from datetime import datetime

from basics.day_counter import DayCounter
from underlying.interest_rate.interest_rate_curve import InterestRateCurve


class FloatingInterestRateCoupon:
    """
    Representa un cupón de tipo variable de un swap de tipos de interés.

    El tipo forward se proyecta a partir de la curva de estimación (e.g., Euribor 6M)
    y el descuento se realiza con la curva de descuento (e.g., OIS/€STR).

    NPV = N × F(t1, t2) × τ × DF(t_pago)

    Parameters
    ----------
    notional : float
        Nominal del swap.
    payment_date : datetime
        Fecha de pago del cupón.
    start_date : datetime
        Fecha inicio del periodo de devengo (fixing).
    end_date : datetime
        Fecha fin del periodo de devengo.
    day_count : str
        Convención de conteo de días (e.g., 'act360').
    """

    def __init__(self, notional: float, payment_date: datetime,
                 start_date: datetime, end_date: datetime, day_count: str):

        self.notional = notional
        self.payment_date = payment_date
        self.start_date = start_date
        self.end_date = end_date
        self.day_count = day_count
        self.delta_time = DayCounter.year_fraction(day_count, self.start_date, self.end_date)

    def npv(self, estimation_curve: InterestRateCurve, discount_curve: InterestRateCurve) -> float:
        """Calcula el valor presente del cupón variable. Si la fecha de pago ya pasó, devuelve 0."""
        if self.payment_date < discount_curve.start_date:
            return 0.0

        forward_rate = estimation_curve.forward_rate(self.start_date, self.end_date, self.day_count)

        discount_factor = discount_curve.interpolate(self.payment_date)

        return self.notional * forward_rate * self.delta_time * discount_factor