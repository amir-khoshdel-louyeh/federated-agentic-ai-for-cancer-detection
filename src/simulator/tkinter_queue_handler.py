import logging
import queue

class TkinterQueueHandler(logging.Handler):
    """A logging handler that sends log records to a thread-safe queue for Tkinter GUI consumption."""
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        try:
            msg = self.format(record)
            self.log_queue.put(msg)
        except Exception:
            self.handleError(record)
