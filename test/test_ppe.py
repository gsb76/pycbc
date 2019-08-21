'''
These are the unittests for the ppe module.
'''
import pycbc
import pycbc.ppe.ppe_tools as ppe
from pycbc.types import TimeSeries
import unittest
import numpy
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


suite = unittest.TestSuite()
suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestPPE))

if __name__ == '__main__':
    results = unittest.TextTestRunner(verbosity=2).run(suite)
    simple_exit(results)
