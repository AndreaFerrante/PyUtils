from datetime import datetime, timedelta


def get_yesterday(format: str = "%Y%m%d", days: int = 1) -> str:
    """Return the date `days` ago formatted as a string."""
    return (datetime.now() - timedelta(days=days)).strftime(format)


def get_today(format: str = "%Y%m%d") -> str:
    """Return today's date formatted as a string."""
    return datetime.now().strftime(format)


def get_recent_monday(format: str = "%Y%m%d") -> str:
    """Return the most recent Monday (or today if today is Monday)."""
    current_weekday = datetime.today().weekday()
    return (datetime.now() - timedelta(days=current_weekday)).strftime(format)


def get_last_monday(format: str = "%Y%m%d") -> str:
    """Return the Monday of the previous week."""
    current_weekday = datetime.today().weekday()
    return (datetime.now() - timedelta(days=current_weekday + 7)).strftime(format)


def get_current_month_startdate(format: str = "%Y%m%d") -> str:
    """Return the first day of the current month."""
    return datetime.today().replace(day=1).strftime(format)


def get_current_year_startdate(format: str = "%Y%m%d") -> str:
    """Return the first day of the current year."""
    return datetime.today().replace(month=1, day=1).strftime(format)


def month_reduction(month_delta: int, month_current: int, year_current: int) -> str:
    """Return YYYYMMDD of the first day of the month `month_delta` months before the given month."""
    total_months = year_current * 12 + (month_current - 1) - month_delta
    year = total_months // 12
    month = (total_months % 12) + 1
    return f"{year}{month:02d}01"


def get_time_month(delta_month: int, current_month: int, current_year: int) -> str:
    """Return YYYYMMDD of the first day of the month `delta_month` months before the given month."""
    total_months = current_year * 12 + (current_month - 1) - delta_month
    year = total_months // 12
    month = (total_months % 12) + 1
    return f"{year}{month:02d}01"
