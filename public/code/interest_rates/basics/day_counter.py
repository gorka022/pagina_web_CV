from datetime import datetime


class DayCounter:
    """
    Implementa las convenciones estándar de conteo de días para productos de tipos de interés.

    Convenciones soportadas:
    - Act/365: Días reales / 365
    - Act/360: Días reales / 360
    - 30/360:  30/360 US (Bond Basis)
    - 30E/360: 30E/360 European (Eurobond Basis) — Estándar para swaps EUR
    """

    @staticmethod
    def year_fraction(day_count: str,
                      start_date: datetime,
                      end_date: datetime) -> float:
        """Calcula la fracción de año entre dos fechas según la base elegida."""

        day_count = day_count.lower().replace(" ", "").replace("_", "").replace("/", "")

        if day_count == 'act365':
            return (end_date - start_date).days / 365

        elif day_count == 'act360':
            return (end_date - start_date).days / 360

        elif day_count == '30360':
            return DayCounter.thirty_360(start_date, end_date)

        elif day_count == '30e360':
            return DayCounter.thirty_e_360(start_date, end_date)

        else:
            raise ValueError(f"La base '{day_count}' no es una base valida. "
                             f"Opciones: act365, act360, 30360, 30e360")

    @staticmethod
    def thirty_360(start_date: datetime, end_date: datetime) -> float:
        """
        Calcula la fracción de año con la convención 30/360 US (Bond Basis).

        Reglas:
        - Si d1 == 31, se ajusta a 30
        - Si d2 == 31 y d1 >= 30, d2 se ajusta a 30
        """
        d1 = start_date.day
        m1 = start_date.month
        y1 = start_date.year

        d2 = end_date.day
        m2 = end_date.month
        y2 = end_date.year

        if d1 == 31:
            d1 = 30

        if d2 == 31 and d1 == 30:
            d2 = 30

        days = 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)

        return days / 360

    @staticmethod
    def thirty_e_360(start_date: datetime, end_date: datetime) -> float:
        """
        Calcula la fracción de año con la convención 30E/360 (European / Eurobond Basis).

        Estándar para swaps en el mercado EUR.

        Reglas:
        - Si d1 == 31, se ajusta a 30
        - Si d2 == 31, se ajusta a 30 (sin condición sobre d1, a diferencia de 30/360 US)
        """
        d1 = min(start_date.day, 30)
        d2 = min(end_date.day, 30)

        m1, y1 = start_date.month, start_date.year
        m2, y2 = end_date.month, end_date.year

        days = 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)

        return days / 360
