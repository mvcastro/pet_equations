import numpy as np
from numpy.typing import NDArray


def calculate(
    tmean: NDArray[np.floating],
    delta: NDArray[np.floating],
    es: NDArray[np.floating],
    ea: NDArray[np.floating],
    psi_const: NDArray[np.floating],
    rn: NDArray[np.floating],
    u2: NDArray[np.floating],
    g: NDArray[np.floating] ,
) -> NDArray[np.floating]:

    eto = (
        0.408 * delta * (rn - g) + psi_const * 900.0 * u2 * (es - ea) / (tmean + 273)
    ) / (delta + psi_const * (1 + 0.34 * u2))

    return eto
