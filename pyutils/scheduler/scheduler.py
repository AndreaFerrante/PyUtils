import time
import datetime
from threading import Thread


class FunctionScheduler:

    def __init__(self):
        self.scheduled_functions = list()
        
    def schedule(self, function, target_time):
        self.scheduled_functions.append((function, target_time))
        
    def _run_function_at_time(self, function, target_time):
        while True:
            current_time = datetime.datetime.now()
            if current_time >= target_time:
                function()
                break
            time.sleep(1)  # Sleep for 1 second

    def start(self):

        threads = list()

        for function, target_time in self.scheduled_functions:
            thread = Thread(target=self._run_function_at_time, args=(function, target_time))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()