import numpy as np
import copy
from pycbc.types import TimeSeries

def derivative_interpolant(x, deriv_i, deriv_f, x_i, x_f):
    """Evaluates a quadratic function Q(x) determined by the value of its
    derivative at two locations. Note that this quadratic function is only
    determined up to a constant. 

    Parameters
    ----------
    x : float
        x-coordinate at which to evaluate the quadratic function Q.
    deriv_i : float
        value of the derivative of the quadratic function Q at x_i.
    deriv_f : float
        value of the derivative of the quadratic function Q at x_f.
    x_i : float
        x-coordinate such that Q'(x_i) = deriv_i
    x_f : float
        x-coordinate such that Q'(x_f) = deriv_f

    Returns
    -------
    q_x : float
        The quadratic function evaluated at x.
    """

    linear_term = (deriv_i * x_f - deriv_f * x_i) * x
    quadratic_term = 0.5 * (deriv_f - deriv_i) * x**2
    q_x = (quadratic_term + linear_term) / (x_f - x_i)
    return q_x

def one_sided_planck_taper(e, n, N):
    """Returns a value between 0 and 1 based on the Planck-taper window:
    https://en.wikipedia.org/wiki/Window_function
    Expects that n is an index into an ordered dataset (e.g. time series).

    Parameters
    ----------
    e : float
        The fraction of the data to apply windowing to.
    n : int
        The index of the data point on which to apply windowing.
    N : int
        The total number of data points. 

    Returns
    -------
    f : float
        The fraction by which to multiply the value of the data
        indexed by `n`.
    """

    if n == 0.0:
        return 0.0
    
    if n >= e * N:
        return 1.0
    
    z_plus = N * e * (1.0 / n + 1.0 / (n - e * N))
    
    if n < e * N:
        return 1.0 / (np.exp(z_plus) + 1.0)
    else:
        return 1.0

def apply_tapering(td_waveform, e):
    """Applies tapering to the beginning of a time-domain waveform using the
    Planck-taper window. The fraction of the waveform to taper is given by `e`.

    Parameters
    ----------
    td_waveform : TimeSeries
        Time series to apply tapering to.
    e : float
        fraction of the waveform to taper, starting from the left.

    Returns
    -------
    tapered_waveform : TimeSeries
         Time series after applying tapering.
    """

    N = len(td_waveform)
    bound = int(e * N)-1
    tapered_waveform = td_waveform.copy()
    for n in range (0, bound):
        tapered_waveform[n] = one_sided_planck_taper(e, n, N) * td_waveform[n]
    return tapered_waveform
