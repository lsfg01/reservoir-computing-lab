import time
from contextlib import contextmanager
from typing import Generator


@contextmanager
def timer() -> Generator[dict, None, None]:
    """
    Context manager que mide el tiempo transcurrido en segundos.

    Uso:
        with timer() as t:
            ...
        print(t["elapsed"])  # segundos
    """
    result: dict = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["elapsed"] = time.perf_counter() - start
