'''
These are the unittests for the ppe module.
'''
import lal
import pycbc
import pycbc.ppe.ppe_tools as ppe
from pycbc.types import TimeSeries
import unittest
import numpy as np
from utils import parse_args_all_schemes, simple_exit


class TestPPE(unittest.TestCase):

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

    def test_ddeltaphidf_early_inspiral(self):
        f = 60.0
        total_mass = 85.0
        beta = 0.5
        b = -1.0
        #We approximate the derivative via a finite difference:
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
        #We approximate the derivative via a finite difference:
        delta_f = 1.e-4
        deltaphi_ahead = epsilon * \
          pow(np.pi * total_mass * lal.MTSUN_SI * (f + delta_f), 1. / 3.)
        deltaphi_behind = epsilon * \
          pow(np.pi * total_mass * lal.MTSUN_SI * (f - delta_f), 1. / 3.)
        approx_deriv = (deltaphi_ahead - deltaphi_behind) / (2.0 *  delta_f)
        self.assertAlmostEqual(approx_deriv,
          ppe.ddeltaphidf_late_inspiral(f, total_mass, epsilon))

suite = unittest.TestSuite()
suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestPPE))

if __name__ == '__main__':
    results = unittest.TextTestRunner(verbosity=2).run(suite)
    simple_exit(results)
