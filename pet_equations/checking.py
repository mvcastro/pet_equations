"""Module functions to check the correct type parameters """

from typing import Any, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def _check_param_float_type(param: Any) -> NDArray[np.float64]:
    """Checking if parameters are the correct type

    Args:
        param (Any): Parameters from the PET equations

    Raises:
        ValueError: raises error if not correct type

    Returns:
        NDArray[np.float64]: correspondent array
    """

    if isinstance(param, (int, float)):
        param = np.array(param)
        return param

    if isinstance(param, np.ndarray):
        if param.dtype in (float, int):
            return param

    if isinstance(param, pd.Series):
        if param.dtype in (float, int):
            return param.to_numpy()

    raise ValueError(
        'Parameters must be of type float or ndarray or Series of float values')


def _check_param_int_type(param: Any) -> NDArray[np.int64]:
    """Checking if parameters are the correct type

    Args:
        param (Any): Parameters from the PET equations

    Raises:
        ValueError: raises error if not correct type

    Returns:
        NDArray[np.int64]: correspondent array
    """
    if isinstance(param, (int, float)):
        param = np.array(int(param))
        return param

    if isinstance(param, np.ndarray):
        if param.dtype in (float, int):
            return param.astype(int)

    if isinstance(param, pd.Series):
        if param.dtype in (float, int):
            return param.astype(int).to_numpy()

    raise ValueError(
        'Parameters must be of type int or ndarray or Series of integer values')


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


def _check_array_sizes(params: list[Any], max_size: int) -> None:
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
