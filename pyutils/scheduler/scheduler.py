import time
import datetime
from threading import Thread
from typing import Callable


class FunctionScheduler:
    """Schedule Python callables to run at specific datetimes."""

    def __init__(self) -> None:
        self.scheduled_functions: list[tuple[Callable, datetime.datetime]] = []

    def schedule(self, function: Callable, target_time: datetime.datetime) -> None:
        """Register function to be called at target_time."""
        self.scheduled_functions.append((function, target_time))

    def _run_at(self, function: Callable, target_time: datetime.datetime) -> None:
        while datetime.datetime.now() < target_time:
            time.sleep(1)
        function()

    def start(self) -> None:
        """Start all scheduled functions in daemon threads and wait for completion."""
        threads = [
            Thread(target=self._run_at, args=(fn, t), daemon=True)
            for fn, t in self.scheduled_functions
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
