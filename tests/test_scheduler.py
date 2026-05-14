# tests/test_scheduler.py
import datetime
from pyutils.scheduler.scheduler import FunctionScheduler


def test_scheduler_runs_function():
    called = []

    def my_func():
        called.append(True)

    scheduler = FunctionScheduler()
    # Schedule for 1 second in the past so it runs immediately
    past_time = datetime.datetime.now() - datetime.timedelta(seconds=1)
    scheduler.schedule(my_func, past_time)
    scheduler.start()

    assert len(called) == 1


def test_scheduler_runs_multiple_functions():
    results = []

    def f1():
        results.append('f1')

    def f2():
        results.append('f2')

    scheduler = FunctionScheduler()
    past = datetime.datetime.now() - datetime.timedelta(seconds=1)
    scheduler.schedule(f1, past)
    scheduler.schedule(f2, past)
    scheduler.start()

    assert 'f1' in results
    assert 'f2' in results
