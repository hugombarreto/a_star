"""
specializer AStarSpecializer
"""

import unittest
from tests.time_comparison import TimeComparison

if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TimeComparison('test_growing_size'))

    unittest.TextTestRunner(verbosity=1).run(suite)
