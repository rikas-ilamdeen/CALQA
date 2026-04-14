import time

class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.end = time.perf_counter()
        self.elapsed_ms = (self.end - self.start) * 1000
