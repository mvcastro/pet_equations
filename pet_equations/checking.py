from typing import Any, List

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def _check_param_type(param: Any) -> NDArray[Any]:
    """Checking if parameters are the correct type

    Args:
        param (Any): Parameters from the PET equations

    Raises:
        ValueError: raises error if not correct type

    Returns:
        NDArray[Any]: correspondent array
    """
    if isinstance(param, (int, float)):
        param = np.array(param)
    elif not isinstance(param, (np.ndarray, pd.Series)):
        raise ValueError('Parameters must be of type float or ndarray or Series')
    return param


def _check_latitude(latitude: Any, max_size: int) -> None:
    """Checking if latitude has the correct size

    Args:
        latitude (Any): Expected type of latitude

    Raises:
        ValueError: Raises error if not correct type
    """
    if latitude.size != 1 and latitude.size != max_size:
        raise ValueError(
            "Latitude array must have length = 1 or the same length as others parameters.")


def _check_array_sizes(params: List[Any], max_size: int) -> None:
    """Checking if all arrays of variables ahave the same size

    Args:
        params (List[Any]): Variables of the PET equation
        max_size (int): Size of the biggest array

    Raises:
        ValueError: Raises error if arrays don't have the same size.
    """
    for param in params:
        if param.size != max_size:
            raise ValueError(
                "Arrays of tmin, tmax, tmean and doy must have the same length.")
