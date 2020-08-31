'''
These are the unittests for the ppe module.
'''
import lal
import pycbc
import pycbc.ppe.ppe_tools as ppe
from pycbc.types import TimeSeries
from pycbc.waveform import get_td_waveform, _lalsim_fd_waveform
import unittest
import numpy as np
from utils import parse_args_all_schemes, simple_exit


class TestPPE(unittest.TestCase):

    def test_name_no_ppe(self):
        s_ppe = "IMRPhenomPv2_ppE"
        s = "IMRPhenomPv2"
        self.assertEqual("IMRPhenomPv2", ppe.name_no_ppe(s_ppe))
        self.assertEqual("IMRPhenomPv2", ppe.name_no_ppe(s))

    def test_derivative_interpolant(self):
        x = 0.51
        deriv_i = 0.2
        deriv_f = -0.7
        x_i = -1.7
        x_f = 2.2
        #A quadratic satisfying these properties is: -(0.45*x**2 + 0.75*x) / 3.9
        self.assertAlmostEqual(
          ppe.derivative_interpolant(x, deriv_i, deriv_f, x_i, x_f),
          -(0.45*x**2 + 0.75*x) / 3.9)
        x = -0.47
        self.assertAlmostEqual(
          ppe.derivative_interpolant(x, deriv_i, deriv_f, x_i, x_f),
          -(0.45*x**2 + 0.75*x) / 3.9)
        x = 9.23
        self.assertAlmostEqual(
          ppe.derivative_interpolant(x, deriv_i, deriv_f, x_i, x_f),
          -(0.45*x**2 + 0.75*x) / 3.9)

    def test_apply_tapering(self):
        mass1 = 20.0
        mass2 = 30.0
        delta_t = 1.0/4096
        f_lower = 30.0
        hp, hc = get_td_waveform(approximant="SEOBNRv4_opt",
                         mass1=mass1,
                         mass2=mass2,
                         delta_t=delta_t,
                         f_lower=f_lower)
        hp.resize(4096)
        hp_tapered = ppe.apply_tapering(hp, 0.0625)

        #  Values computed locally.
        self.assertEqual(-3.70740680186999e-20, hp[100])
        self.assertEqual(-9.576880290197069e-20, hp[150])
        self.assertEqual(1.7453813195748116e-19, hp[200])
        self.assertEqual(-1.0572297855920886e-20, hp_tapered[100])
        self.assertEqual(-6.417023498011562e-20, hp_tapered[150])
        self.assertEqual(1.682780752157663e-19, hp_tapered[200])
        self.assertEqual(hp[256], hp_tapered[256])

    def test_ddeltaphidf_early_inspiral(self):
        f = 60.0
        total_mass = 85.0
        beta = 0.5
        b = -1.0
        #  We approximate the derivative via a finite difference:
        delta_f = 1.e-4
        deltaphi_ahead = beta * \
          pow(np.pi * total_mass * lal.MTSUN_SI * (f + delta_f), b / 3.)
        deltaphi_behind = beta * \
          pow(np.pi * total_mass * lal.MTSUN_SI * (f - delta_f), b / 3.)
        approx_deriv = (deltaphi_ahead - deltaphi_behind) / (2.0 *  delta_f)
        self.assertAlmostEqual(approx_deriv,
          ppe.ddeltaphidf_early_inspiral(f, total_mass, beta, b))

    def test_ddeltaphidf_late_inspiral(self):
        f = 40.0
        total_mass = 65.0
        epsilon = 2.0
        #  We approximate the derivative via a finite difference:
        delta_f = 1.e-4
        deltaphi_ahead = epsilon * \
          pow(np.pi * total_mass * lal.MTSUN_SI * (f + delta_f), 1. / 3.)
        deltaphi_behind = epsilon * \
          pow(np.pi * total_mass * lal.MTSUN_SI * (f - delta_f), 1. / 3.)
        approx_deriv = (deltaphi_ahead - deltaphi_behind) / (2.0 *  delta_f)
        self.assertAlmostEqual(approx_deriv,
          ppe.ddeltaphidf_late_inspiral(f, total_mass, epsilon))

    def test_light_ring_frequency_in_hz(self):
        total_mass = 50.0
        #  Value computed locally.
        self.assertEqual(
          ppe.light_ring_frequency_in_hz(total_mass), 215.45314367540433)

    def test_ringdown_frequency(self):
        mass1 = 20.0
        mass2 = 30.0
        delta_t = 1.0/4096
        f_lower = 30.0
        hp, hc = get_td_waveform(approximant="SEOBNRv4_opt",
                         mass1=mass1,
                         mass2=mass2,
                         delta_t=delta_t,
                         f_lower=f_lower)

        hp_tilde = hp.to_frequencyseries()
        #  Value computed locally.
        self.assertEqual(
          ppe.ringdown_frequency(hp_tilde, mass1 + mass2), 426.9718957588146)

    def test_apply_ppe_correction(self):
        mass1 = 20.0
        mass2 = 30.0
        delta_t = 1.0/4096
        f_lower = 30.0
        hp, hc = get_td_waveform(approximant="SEOBNRv4_opt",
                         mass1=mass1,
                         mass2=mass2,
                         delta_t=delta_t,
                         f_lower=f_lower)
        hp.resize(4096)
        hp_tap = ppe.apply_tapering(hp, 0.0625)
        hp_tap_tilde = hp_tap.to_frequencyseries()
 
        beta = 2.0
        b = -1.0
        epsilon = 1.0
        delta_f = 8.0
        new_hp_tap_tilde = ppe.apply_ppe_correction_dep(hp_tap_tilde, \
          mass1 + mass2, beta, b, f_lower, epsilon, delta_f)
        new_hp = new_hp_tap_tilde.to_timeseries()

        #  For this choice of parameters, the No Correction regime ends at 30Hz,
        #  the Beta Correction regime ends at 73Hz, and the Epsilon Correction
        #  regime ends at 265Hz.  

        #  Values computed locally.
        self.assertAlmostEqual(-1.6921568775561246e-19, hp[0], delta = 1.0e-25)
        self.assertAlmostEqual(-6.140782138315621e-20, hp[1024], delta = 1.0e-25)
        self.assertAlmostEqual(-6.606212295280634e-20, hp[1536], delta = 1.0e-25)
        self.assertAlmostEqual(2.0358901097288783e-22, hp[1664], delta = 1.0e-25)
        self.assertAlmostEqual(0.0, hp[2048])
        self.assertAlmostEqual(2.4923860073761226e-21, new_hp[0], delta = 1.0e-25)
        self.assertAlmostEqual(1.0098579793661773e-19, new_hp[1024], delta = 1.0e-25)
        self.assertAlmostEqual(-4.2015861360493625e-19, new_hp[1536], delta = 1.0e-25)
        self.assertAlmostEqual(3.949054577517775e-22, new_hp[1664], delta = 1.0e-25)
        self.assertAlmostEqual(-5.340958147428273e-23, new_hp[2048], delta = 1.0e-25)

        self.assertAlmostEqual(-7.908682071495725e-24, hp_tap_tilde[0].real, delta = 1.0e-25)
        self.assertAlmostEqual(-6.258156232014422e-21, hp_tap_tilde[60].real, delta = 1.0e-25)
        self.assertAlmostEqual(-3.859513135497637e-21, hp_tap_tilde[80].real, delta = 1.0e-25)
        self.assertAlmostEqual(-1.522447602506527e-21, hp_tap_tilde[200].real, delta = 1.0e-25)
        self.assertAlmostEqual(-6.862493404260128e-22, hp_tap_tilde[300].real, delta = 1.0e-25)
        self.assertAlmostEqual(-7.908682071495725e-24, new_hp_tap_tilde[0].real, delta = 1.0e-25)
        self.assertAlmostEqual(-3.467591052503212e-21, new_hp_tap_tilde[60].real, delta = 1.0e-25)
        self.assertAlmostEqual(-1.639700771035731e-21, new_hp_tap_tilde[80].real, delta = 1.0e-25)
        self.assertAlmostEqual(-9.465021656090846e-22, new_hp_tap_tilde[200].real, delta = 1.0e-25)
        self.assertAlmostEqual(1.2410412323990297e-21, new_hp_tap_tilde[300].real, delta = 1.0e-25)

    def test_apply_ppe_correction2(self):
        #  Tests that a dictionary can be created and passed to
        #  `_lalsim_fd_waveform` and that this function returns.

        p = {
          "phase_order": -1,
          "amplitude_order": -1,
          "spin_order": -1,
          "tidal_order": -1,
          "eccentricity_order": -1,
          "lambda1": None,
          "lambda2": None,
          "lambda_octu1": None,
          "lambda_octu2": None,
          "quadfmode1": None,
          "quadfmode2": None,
          "octufmode1": None,
          "octufmode2": None,
          "dquad_mon1": None,
          "dquad_mon2": None,
          "numrel_data": None,
          "modes_choice": None,
          "frame_axis": None,
          "side_bands": None,
          "mode_array": None,
          "mass1": 20.0,
          "mass2": 30.0,
          "spin1x": 0.0,
          "spin1y": 0.0,
          "spin1z": 0.2,
          "spin2x": 0.0,
          "spin2y": 0.0,
          "spin2z": 0.1,
          "distance": 450,
          "inclination": 0.0,
          "coa_phase": 0.0,
          "long_asc_nodes": 0.0,
          "eccentricity": 0.0,
          "mean_per_ano": 0.0,
          "delta_f": 1.0/4096,
          "f_lower": 20.0,
          "f_final": 0,
          "f_ref": 20.0,
          "ppe_beta": 1.0,
          "ppe_b" : -1.0,
          "ppe_epsilon" : 2.0,
          "ppe_kind" : "C1_eps",
          "approximant": "IMRPhenomPv2"}

        hp, hc = _lalsim_fd_waveform(**p)

suite = unittest.TestSuite()
suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestPPE))

if __name__ == '__main__':
    results = unittest.TextTestRunner(verbosity=2).run(suite)
    simple_exit(results)
