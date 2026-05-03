import calendar
from typing import List
from datetime import datetime
from dateutil.relativedelta import relativedelta

from basics.day_counter import DayCounter
from product.interest_rate.fixed_interest_rate_coupon import FixedInterestRateCoupon
from product.interest_rate.floating_interest_rate_coupon import FloatingInterestRateCoupon
from underlying.interest_rate.interest_rate_curve import InterestRateCurve
from basics.bussiness_calendar import apply_modified_following


def _is_end_of_month(date: datetime) -> bool:
    """Comprueba si una fecha cae en el ultimo dia del mes."""
    last_day = calendar.monthrange(date.year, date.month)[1]
    return date.day == last_day


def _adjust_eom(date: datetime) -> datetime:
    """Ajusta una fecha al ultimo dia de su mes."""
    last_day = calendar.monthrange(date.year, date.month)[1]
    return date.replace(day=last_day)


def generate_schedule_backward(start_date: datetime, end_date: datetime,
                                freq_months: int, eom: bool = True) -> List[datetime]:
    """
    Genera un schedule de fechas con rolling backward desde end_date.
    Si la fecha final cae en fin de mes y eom=True, aplica regla End-Of-Month.
    """
    is_eom = eom and _is_end_of_month(end_date)

    dates = [end_date]
    current = end_date

    while True:
        current = current - relativedelta(months=freq_months)
        if is_eom:
            current = _adjust_eom(current)
        if current <= start_date:
            dates.insert(0, start_date)
            break
        dates.insert(0, current)

    return dates


class InterestRateSwap:
    """Valoracion de un Interest Rate Swap Fixed vs. Floating (dual-curve)."""

    def __init__(self, start_date: datetime, maturity_years: int, notional: float,
                 fixed_rate: float, fixed_freq_months: int, fixed_day_count: str,
                 float_freq_months: int, float_day_count: str):

        self.start_date = start_date
        self.end_date = start_date + relativedelta(years=maturity_years)
        self.notional = notional
        self.fixed_rate = fixed_rate

        # Generar schedules con rolling backward (EOM)
        fixed_dates = generate_schedule_backward(start_date, self.end_date, fixed_freq_months)
        float_dates = generate_schedule_backward(start_date, self.end_date, float_freq_months)

        # Pata Fija
        self.fixed_leg = []
        for i in range(len(fixed_dates) - 1):
            adj_start = apply_modified_following(fixed_dates[i])
            adj_end = apply_modified_following(fixed_dates[i + 1])
            coupon = FixedInterestRateCoupon(
                notional=notional, coupon=fixed_rate, payment_date=adj_end,
                start_date=adj_start, end_date=adj_end, day_count=fixed_day_count
            )
            self.fixed_leg.append(coupon)

        # Pata Flotante
        self.floating_leg = []
        for i in range(len(float_dates) - 1):
            adj_start = apply_modified_following(float_dates[i])
            adj_end = apply_modified_following(float_dates[i + 1])
            coupon = FloatingInterestRateCoupon(
                notional=notional, payment_date=adj_end,
                start_date=adj_start, end_date=adj_end, day_count=float_day_count
            )
            self.floating_leg.append(coupon)

    def npv(self, discount_curve: InterestRateCurve, estimation_curve: InterestRateCurve) -> float:
        """NPV = NPV(Fixed Leg) - NPV(Floating Leg)"""
        fixed_npv = sum(coupon.npv(discount_curve) for coupon in self.fixed_leg)
        float_npv = sum(coupon.npv(estimation_curve, discount_curve) for coupon in self.floating_leg)
        return fixed_npv - float_npv