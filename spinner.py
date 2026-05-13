# terminal spinner made by deepseek v4 pro
import threading
import time
import sys

class Spinner:
    def __init__(self, message: str = "Loading"):
        self.message = message
        self.spinner_chars = ["|", "/", "-", "\\"]
        self._stop_event = threading.Event()
        self._thread = None
        self._start_time = None

    def _spin(self):
        i = 0
        while not self._stop_event.is_set():
            sys.stdout.write(f"\r{self.spinner_chars[i % len(self.spinner_chars)]} {self.message}...")
            sys.stdout.flush()
            i += 1
            time.sleep(0.1)

    def start(self):
        self._stop_event.clear()
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        self._stop_event.set()
        if self._thread:
            self._thread.join()
        elapsed = time.time() - self._start_time
        # clear the spinner line
        sys.stdout.write(f"\r{' ' * (len(self.message) + 20)}\r")
        sys.stdout.flush()
        return elapsed