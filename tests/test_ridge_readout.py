import warnings

import numpy as np
from scipy.linalg import LinAlgWarning

from rc_lab.readouts.ridge import RidgeReadout


def test_ridge_readout_svd_solver_avoids_ill_conditioned_warning():
    F = np.ones((8, 3))
    Y = np.arange(8.0).reshape(-1, 1)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        readout = RidgeReadout(ridge_param=1e-20)
        readout.fit(F, Y)

    warning_messages = [str(warning.message).lower() for warning in caught]
    assert not any(
        issubclass(warning.category, LinAlgWarning) or "ill-conditioned" in message
        for warning, message in zip(caught, warning_messages)
    )
