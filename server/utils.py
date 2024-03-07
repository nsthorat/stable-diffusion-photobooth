import logging
import time
from typing import Any


class DebugTimer:
  """A context manager that prints the time elapsed in a block of code.

  ```py
    with DebugTimer('dot product'):
      np.dot(np.random.randn(1000), np.random.randn(1000))
  ```

  $ dot product took 0.001s.
  """

  def __init__(self, name: str) -> None:
    self.name = name

  def __enter__(self) -> "DebugTimer":
    """Start a timer."""
    self.start = time.perf_counter()
    log(f"{self.name} started...")
    return self

  def __exit__(self, *args: list[Any]) -> None:
    """Stop the timer and print the elapsed time."""
    log(f"{self.name} took {(time.perf_counter() - self.start):.3f}s.")


def log(*args: Any) -> None:
  """Print and logs a message so it shows up in the logs on cloud."""
  print(*args)
  logging.info([" ".join(map(str, args))])
