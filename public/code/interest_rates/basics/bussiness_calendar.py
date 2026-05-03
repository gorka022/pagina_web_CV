import pandas as pd
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, GoodFriday, EasterMonday
from pandas.tseries.offsets import CustomBusinessDay


class TargetHolidays(AbstractHolidayCalendar):
    """
    Calendario de festivos TARGET (Trans-European Automated Real-time Gross Settlement).

    Festivos incluidos: Año Nuevo, Viernes Santo, Lunes de Pascua,
    Día del Trabajo, Navidad y San Esteban.
    """
    rules = [
        Holiday('New Year', month=1, day=1),
        GoodFriday,
        EasterMonday,
        Holiday('Labor Day', month=5, day=1),
        Holiday('Christmas', month=12, day=25),
        Holiday('Boxing Day', month=12, day=26)
    ]


business_day_target = CustomBusinessDay(calendar=TargetHolidays())

# Alias para compatibilidad con código existente
bussiness_day_target = business_day_target


def apply_modified_following(date):
    """
    Ajusta una fecha según la convención Modified Following con calendario TARGET.

    Si la fecha es hábil, se devuelve sin cambios. Si no, se avanza al siguiente
    día hábil; pero si ese día cae en un mes diferente, se retrocede al día hábil
    anterior.
    """
    ts = pd.Timestamp(date)

    # Si es habil, se queda igual
    if business_day_target.is_on_offset(ts):
        return date

    # Movemos hacia adelante
    next_day = (ts + business_day_target).to_pydatetime()

    # Si cambiamos de mes al avanzar, retrocedemos
    if next_day.month != date.month:
        return (ts - business_day_target).to_pydatetime()

    return next_day


def apply_spot_lag(valuation_date, spot_lag_days=2):
    """
    Calcula la fecha spot aplicando el spot lag en días hábiles TARGET.

    Parameters
    ----------
    valuation_date : datetime
        Fecha de valoración.
    spot_lag_days : int
        Número de días hábiles de desfase (default: 2 para T+2).

    Returns
    -------
    datetime
        Fecha spot ajustada al calendario TARGET.
    """
    ts = pd.Timestamp(valuation_date)
    return (ts + spot_lag_days * business_day_target).to_pydatetime()