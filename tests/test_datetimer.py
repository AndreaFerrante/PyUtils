import pytest
from pyutils.service_factory.datetimer import (
    get_today, get_yesterday, get_recent_monday, get_last_monday,
    get_current_month_startdate, get_current_year_startdate,
    month_reduction, get_time_month
)


def test_get_today_format():
    result = get_today()
    assert len(result) == 8
    assert result.isdigit()


def test_get_yesterday_format():
    result = get_yesterday()
    assert len(result) == 8
    assert result.isdigit()


def test_get_recent_monday_format():
    result = get_recent_monday()
    assert len(result) == 8
    assert result.isdigit()


def test_get_last_monday_format():
    result = get_last_monday()
    assert len(result) == 8
    assert result.isdigit()


def test_get_current_month_startdate_ends_in_01():
    result = get_current_month_startdate()
    assert result[-2:] == '01'


def test_get_current_year_startdate_ends_in_0101():
    result = get_current_year_startdate()
    assert result[-4:] == '0101'


def test_month_reduction_returns_string():
    # Bug: np not imported → NameError; no return → None
    result = month_reduction(month_delta=2, month_current=5, year_current=2026)
    assert result is not None
    assert isinstance(result, str)
    assert len(result) == 8


def test_month_reduction_correct_value():
    # 2 months before May 2026 = March 2026 → '20260301'
    result = month_reduction(month_delta=2, month_current=5, year_current=2026)
    assert result == '20260301'


def test_month_reduction_wraps_year():
    # 3 months before Feb 2026 = Nov 2025 → '20251101'
    result = month_reduction(month_delta=3, month_current=2, year_current=2026)
    assert result == '20251101'


def test_get_time_month_format():
    result = get_time_month(delta_month=2, current_month=5, current_year=2026)
    assert isinstance(result, str)
    assert len(result) == 8
