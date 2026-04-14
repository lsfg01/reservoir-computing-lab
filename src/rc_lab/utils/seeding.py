import random

import numpy as np


def set_seed(seed: int) -> None:
    """Fija la semilla global de numpy y random para reproducibilidad."""
    np.random.seed(seed)
    random.seed(seed)
