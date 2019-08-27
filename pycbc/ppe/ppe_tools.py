import numpy as np
import lal
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

def ddeltaphidf_early_inspiral(f, total_mass, beta, b):
    """Evaluates the derivative of the phase change in the early inspiral with
    respect to the frequency at some frequency f. The phase change in the early
    inspiral is given by beta * v ** b, where v = (pi * total_mass * f) ** (1/3)
    The derivative is then given by beta * b * v ** (b-1) * dv/df, where dv/df
    is given by (1/3) * (pi * total_mass * f) ** (-2/3) * pi * total_mass,
    or (1/3) * v ** -2 * pi * total_mass.

    Parameters
    ----------
    f : float
        The frequency, in Hz, at which to evaluate the derivative.
    total_mass : float
        The total mass of the binary, in solar masses.
    beta : float
        The ppE parameter beta.
    b : float
        The ppE parameter b.

    Returns
    -------
    deriv : float
         The value of the derivative at the frequency f.
    """

    total_mass_in_seconds = total_mass * lal.MTSUN_SI
    v = pow(np.pi * total_mass_in_seconds * f, 1. / 3.)
    deriv = b * beta * np.pi * total_mass_in_seconds * pow(v, b - 3) / 3.0
    return deriv

def ddeltaphidf_late_inspiral(f, total_mass, epsilon):
    """Evaluates the derivative of the phase change in the late inspiral with
    respect to the frequency at some frequency f. The phase change in the late
    inspiral is given by epsilon * v, where v = (pi * total_mass * f) ** (1/3).
    The derivative is then given by epsilon * dv/df, where dv/df
    is given by (1/3) * (pi * total_mass * f) ** (-2/3) * pi * total_mass,
    or (1/3) * v ** -2 * pi * total_mass.

    Parameters
    ----------
    f : float
        The frequency, in Hz, at which to evaluate the derivative.
    total_mass : float
        The total mass of the binary, in solar masses.
    epsilon : float
        The ppE parameter epsilon.

    Returns
    -------
    deriv : float
         The value of the derivative at the frequency f.
    """

    total_mass_in_seconds = total_mass * lal.MTSUN_SI
    v = pow(np.pi * total_mass_in_seconds * f, 1. / 3.)
    deriv = epsilon * np.pi * total_mass_in_seconds * pow(v, -2) / 3.0
    return deriv

def light_ring_frequency_in_hz(total_mass):
    """The light ring frequency for a nonspinning BH. 

    Parameters
    ----------
    total_mass : float
        The total mass of the binary, in solar masses.

    Returns
    -------
    f_LR : float
         The light ring frequency, in Hz.
    """

    c = 2.998e8 # m/s
    GM_sun = 1.327e20 # m^3/s^2
    f_LR c**3 / (6 * np.pi * GM_sun * total_mass)
    return f_LR


def ringdown_frequency(fd_waveform, total_mass):
    """Numerically finds the ringdown frequency, f_RD, given a frequency
    domain waveform. The ringdown frequency is as defined in:
    https://arxiv.org/pdf/1508.07253.pdf

    Parameters
    ----------
    fd_waveform : FrequencySeries
        The frequency domain waveform of the coalescence.
    total_mass : float
        The total mass of the binary, in solar masses.

    Returns
    -------
    f_RD : float
         The ringdown frequency, in Hz.
    """

    light_ring_index = np.where(fd_waveform.sample_frequencies.data \
      > light_ring_frequency_in_hz(total_mass))[0][0]
    four_times_light_ring_index = np.where(fd_waveform.sample_frequencies.data \
      > 2 * light_ring_frequency_in_hz(total_mass))[0][0]
    mindex = np.argmin(np.gradient(np.unwrap(np.angle( \
      fd_waveform.data[light_ring_index:four_times_light_ring_index]))))
    f_RD = fd_waveform.sample_frequencies[mindex + light_ring_index]
    return f_RD

def apply_ppe_correction(fd_waveform, total_mass, beta, b, f_low, epsilon, delta_f):
    """Applies the parameterized post-Einsteinian corrections to a waveform
    in the frequency domain such that the derivative of the phase is continuous
    across the different frequency regimes, where a different correction is
    applied in each one. These frequency regimes are:
    * Up to f_low:
    No correction is applied. The waveform is only defined above
    the frequency f_low, so the content contained in frequencies below f_low
    are unphysical.
    * Between f_low and Mf = 0.018:
    The beta-ppe correction is applied, to the phase through
    delta phi = beta * v ^ b, where v = (pi * M * f) ^ (1 / 3).
    * Between Mf = 0.018 and f = 0.75f_RD, the ringdown frequency:
    The epsilon-ppe correction is applied, also to the phase, through
    delta phi = epsilon * v.
    * After f = 0.75f_RD, no correction is applied.

    In between these four frequency ranges, we also have three transition
    regions across which the derivatives of the four ranges are matched at
    each end. The width of this transition region is set by delta_f.

    Parameters
    ----------
    fd_waveform : FrequencySeries
        The frequency domain waveform of the coalescence.
    total_mass : float
        The total mass of the binary, in solar masses.
    beta : float
        The ppE parameter beta.
    b : float
        The ppE parameter b.
    epsilon : float
        The ppE parameter epsilon.
    delta_f : float
        The width of the transition region between each frequency regime.

    Returns
    -------
    corrected_fd_waveform : FrequencySeries
         The frequency domain waveform of the coalescence, after applying
         the ppE corrections.
    """

    total_mass_in_seconds = total_mass * lal.MTSUN_SI
    pi_M = np.pi * total_mass_in_seconds
    corrected_fd_waveform = fd_waveform.copy()

    #  Defines up to where to apply the beta-ppe correction as well as up
    #  to where to apply the epsilon-ppe correction.
    freq_IM_begin = 0.018 / total_mass_in_seconds
    freq_IM_end = 0.75 * ringdown_frequency(fd_waveform, total_mass)

    #  Defines the boundaries of the transition regions:
    #  Transition region between No Correction and Beta Correction:
    f_1 = f_low - 0.5 * delta_f
    f_2 = f_low + 0.5 * delta_f
    #  Transition region between Beta Correction and Epsilon Correction:
    f_3 = freq_IM_begin - 0.5 * delta_f
    f_4 = freq_IM_begin + 0.5 * delta_f
    #  Transition region between Epsilon Correction and Constant Correction:
    f_5 = freq_IM_end - 0.5 * delta_f
    f_6 = freq_IM_end + 0.5 * delta_f 

    #  The values of the derivatives of the phase change at each endpoint of
    #  the transition region. The derivative is then linearly interpolated
    #  across the transition region using the `derivative_interpolant`, which
    #  returns the integral of the linear interpolant, which is the phase
    #  change itself.
    ddelphidf_1 = 0.0
    ddelphidf_2 = dphase_df_early_inspiral(f_2, total_mass, beta, b)
    ddelphidf_3 = dphase_df_early_inspiral(f_3, total_mass, beta, b)
    ddelphidf_4 = dphase_df_late_inspiral(f_4, total_mass, epsilon)
    ddelphidf_5 = dphase_df_late_inspiral(f_5, total_mass, epsilon)
    ddelphidf_6 = 0.0

    #  As `derivative_interpolant` returns an integral, it is only defined up
    #  to a constant. The accumulated phase change must separately be kept
    #  track of, such that the phase change is continuous across the different
    #  frequency ranges, as well as the derivative of the phase change.
    delphi_1 = derivative_interpolant(f_1, ddelphidf_1, ddelphidf_2, f_1, f_2)
    delphi_2 = derivative_interpolant(f_2, ddelphidf_1, ddelphidf_2, f_1, f_2)
    delphi_3 = derivative_interpolant(f_3, ddelphidf_3, ddelphidf_4, f_3, f_4)
    delphi_4 = derivative_interpolant(f_4, ddelphidf_3, ddelphidf_4, f_3, f_4)
    delphi_5 = derivative_interpolant(f_5, ddelphidf_5, ddelphidf_6, f_5, f_6)
    delphi_6 = derivative_interpolant(f_6, ddelphidf_5, ddelphidf_6, f_5, f_6)

    #  The accumulated phase change across the frequency ranges:
    phase_change_1 = -delphi_1
    phase_change_2 = delphi_2 + phase_change_1 - beta * pow(pi_M*f_2, b/3.)
    phase_change_3 = -delphi_3 + phase_change_2 + beta * pow(pi_M*f_3, b/3.)
    phase_change_4 = delphi_4 + phase_change_3 - epsilon * pow(pi_M*f_4,  1/3.)
    phase_change_5 = -delphi_5 + phase_change_4 + epsilon * pow(pi_M*f_5, 1/3.)
    phase_change_6 = delphi_6 + phase_change_5 
   
    # Apply the ppE correction over the various ranges: 
    for i in range(0, len(corrected_fd_waveform.data)):
        freq_i = corrected_fd_waveform.sample_frequencies.data[i]
        vel_i = pow(pi_M * freq_i, 1.0/3.0)

        # Transition from No Correction to Beta Correction:
        if(freq_i > f_1 and freq_i < f_2):
            phase_change = \
              derivative_interpolant(freq_i, ddelphidf_1, ddelphidf_2, f_1, f_2)
            phase_correction = np.exp((phase_change + phase_change_1) * 1j)

        #  Apply Beta Correction:
        if(freq_i > f_2 and freq_i < f_3):
            phase_change = beta * pow(vel_i, b)
            phase_correction = np.exp((phase_change + phase_change_2) * 1j)

        # Transition from Beta Correction to Epsilon Correction:
        if(freq_i > f_3 and freq_i < f_4):
            phase_change = \
              derivative_interpolant(freq_i, ddelphidf_3, ddelphidf_4, f_3, f_4)
            phase_correction = np.exp((phase_change + phase_change_3) * 1j)

        #  Apply Epsilon Correction:
        if(freq_i > f_4 and freq_i < f_5):
            phase_change = epsilon * vel_i
            phase_correction = np.exp(phase_change + phase_change_4 * 1j)

        # Transition from Epsilon Correction to Constant Correction:
        if(freq_i > f_5 and freq_i < f_6):
            phase_change = \
              derivative_interpolant(freq_i, ddelphidf_5, ddelphidf_6, f_5, f_6)
            phase_correction = np.exp((phase_change + phase_change_5) * 1j)

        # Apply Constant Correction:  
        if(freq_i > f_6):
            phase_correction = np.exp(phase_change_6 * 1j)

      corrected_fd_waveform.data[i] = \
        corrected_fd_waveform.data[i] * phase_correction
    return corrected_fd_waveform 
