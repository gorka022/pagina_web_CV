from datetime import datetime

from basics.day_counter import DayCounter
from underlying.interest_rate.interest_rate_curve import InterestRateCurve


class FixedInterestRateCoupon:
    """
    Representa un cupón de tipo fijo de un swap de tipos de interés.

    El valor presente se calcula como:
        NPV = N × c × τ × DF(t_pago)

    donde N es el nominal, c el tipo fijo, τ la fracción de año y DF el discount factor.

    Parameters
    ----------
    notional : float
        Nominal del swap.
    coupon : float
        Tipo fijo del cupón (e.g., 0.024215 para 2.4215%).
    payment_date : datetime
        Fecha de pago del cupón.
    start_date : datetime
        Fecha inicio del periodo de devengo.
    end_date : datetime
        Fecha fin del periodo de devengo.
    day_count : str
        Convención de conteo de días (e.g., '30E360', 'act360').
    """

    def __init__(self, notional: float, coupon: float, payment_date: datetime,
                 start_date: datetime, end_date: datetime, day_count: str):
        self.notional = notional
        self.coupon = coupon
        self.payment_date = payment_date
        self.start_date = start_date
        self.end_date = end_date
        self.day_count = day_count
        self.delta_time = DayCounter.year_fraction(day_count, self.start_date, self.end_date)


    def npv(self, discount_curve: InterestRateCurve) -> float:
        """Calcula el valor presente del cupón fijo. Si la fecha de pago ya pasó, devuelve 0."""
        if self.payment_date < discount_curve.start_date:
            return 0
        else:
            discount_factor = discount_curve.interpolate(self.payment_date)
            return self.notional * discount_factor * self.coupon * self.delta_time