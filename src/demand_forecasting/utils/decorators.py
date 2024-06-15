from typing import Union
from contextlib import contextmanager
import threading
import _thread


class TimeoutException(Exception):
    def __init__(self, msg=""):
        self.msg = msg


@contextmanager
def time_limit(seconds: Union[int, float], msg: str=""):
    """Decorator for timeout exception.

    Args:
        seconds (float): Timeout in seconds.
        msg (str): Timeout message.
    """
    
    timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
    timer.start()
    try:
        yield
    except KeyboardInterrupt:
        raise TimeoutException("Timed out for operation {}".format(msg))
    finally:
        timer.cancel()