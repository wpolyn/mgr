import psutil as pu
import time
import threading

class performance:
    def __init__(self):
        self.sample_freq = 0.1
        self.mem_usage = []
        self.cpu_usage = []
        self._measuring = False
        self._thread = None

    def _monitoring(self):
        while self._measuring:
            self.mem_usage.append(pu.virtual_memory().percent)
            self.cpu_usage.append(pu.cpu_percent(interval=0, percpu=False))
            time.sleep(self.sample_freq)

    def measure_start(self):
        self.mem_usage.clear()
        self.cpu_usage.clear()
        self._measuring = True
        self._thread = threading.Thread(target=self._monitoring)
        self._thread.start()

    def measure_stop(self):
        self._measuring = False
        self._thread.join()
        mem_avg = (sum(self.mem_usage) / len(self.mem_usage)) if self.mem_usage else 0
        cpu_avg = (sum(self.cpu_usage) / len(self.cpu_usage)) if self.cpu_usage else 0
        mem_avg = round(mem_avg, 2)
        cpu_avg = round(cpu_avg, 2)
        print(f"Avg memory usage during the process: {mem_avg}%")
        print(f"Avg CPU usage during the process: {cpu_avg}%")
        return mem_avg, cpu_avg

    def duration_start(self):
        self._start_time = time.time()

    def duration_stop(self):
        duration = time.time() - self._start_time
        duration = round(duration, 3)
        self._start_time = None
        print(f"Processing time: {duration}s.")
        return duration