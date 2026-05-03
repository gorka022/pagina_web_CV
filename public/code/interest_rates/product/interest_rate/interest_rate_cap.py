import numpy as np
from scipy.stats import norm
from dateutil.relativedelta import relativedelta
from scipy.optimize import brentq

from basics.day_counter import DayCounter
from basics.bussiness_calendar import apply_modified_following


class InterestRateCap:
    """Valoracion de un Interest Rate Cap (modelo Normal + Shifted Log-normal)."""

    def __init__(self, start_date, maturity_years, notional, strike, freq_months, day_count):
        self.start_date = start_date
        self.end_date = start_date + relativedelta(years=maturity_years)
        self.notional = notional
        self.strike = strike
        self.freq_months = freq_months
        self.day_count = day_count
        self.caplets_info = []

    def caplet_price(self, forward_rate, strike, vol, time_to_maturity, df, tau):
        """Precio unitario de un caplet con el modelo Normal (Bachelier)."""
        if time_to_maturity <= 1e-6:
            return 0.0
        d = (forward_rate - strike) / (vol * np.sqrt(time_to_maturity))
        price = df * tau * ((forward_rate - strike) * norm.cdf(d) + vol * np.sqrt(time_to_maturity) * norm.pdf(d))
        return price

    def shifted_caplet_price(self, forward_rate, strike, vol, time_to_maturity, df, tau, shift=0.03):
        """Precio unitario de un caplet con el modelo Shifted Log-normal."""
        if time_to_maturity <= 1e-6 or vol <= 1e-6:
            return 0.0
        shifted_forward = forward_rate + shift
        shifted_strike = strike + shift
        if shifted_forward <= 0 or shifted_strike <= 0:
            return 0.0
        d1 = (np.log(shifted_forward / shifted_strike) + 0.5 * vol ** 2 * time_to_maturity) / (vol * np.sqrt(time_to_maturity))
        d2 = d1 - vol * np.sqrt(time_to_maturity)
        price = df * tau * (shifted_forward * norm.cdf(d1) - shifted_strike * norm.cdf(d2))
        return price

    def npv(self, valuation_date, discount_curve, estimation_curve, vol_surface):
        """Calcula el NPV total del Cap sumando todos los caplets (modelo Normal)."""
        npv_total = 0.0
        self.caplets_info.clear()

        unadj_date = self.start_date
        adj_start = apply_modified_following(self.start_date)

        while unadj_date < self.end_date:
            unadj_next = unadj_date + relativedelta(months=self.freq_months)
            adj_next = apply_modified_following(unadj_next)

            time_to_maturity = (adj_start - valuation_date).days / 365.0
            tau = DayCounter.year_fraction(self.day_count, adj_start, adj_next)
            forward_rate = estimation_curve.forward_rate(adj_start, adj_next, self.day_count)
            df = discount_curve.interpolate(adj_next)
            vol = vol_surface.get_volatility(time_to_maturity, self.strike)

            unit_price = self.caplet_price(forward_rate, self.strike, vol, time_to_maturity, df, tau)
            npv_total += unit_price * self.notional

            self.caplets_info.append({
                'start': adj_start, 'end': adj_next,
                'forward_rate': forward_rate, 'time_to_maturity': time_to_maturity,
                'df': df, 'tau': tau, 'vol': vol,
                'caplet_npv': unit_price * self.notional
            })

            unadj_date = unadj_next
            adj_start = adj_next

        return npv_total

    def implied_volatility(self, target_npv, shift=0.03):
        """Volatilidad implicita flat del Cap en el modelo Shifted Log-normal (Brent)."""
        def price_difference(vol_prueba):
            total = 0.0
            for info in self.caplets_info:
                unit_price = self.shifted_caplet_price(
                    info['forward_rate'], self.strike, vol_prueba,
                    info['time_to_maturity'], info['df'], info['tau'], shift
                )
                total += unit_price * self.notional
            return total - target_npv

        return brentq(price_difference, 0.0001, 5.0)