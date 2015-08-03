import unittest
from tests.test_a_star import TestAStar, get_random_grid
from a_star.specializers import SpecializedGrid, GridAsArray


class TestSpecializedAStar(TestAStar):
    def __init__(self, *args, **kwargs):
        super(TestSpecializedAStar, self).__init__(*args, **kwargs)
        self.grid_type = SpecializedGrid

    # @profile
    def test_compare_specialization(self):
        grid, start, finish = get_random_grid(self.grid_size, self.grid_type)
        c_cost = grid.get_a_star_cost(start, finish)
        py_cost = GridAsArray.get_a_star_cost(grid, start, finish)

        self.assertEqual(py_cost, c_cost)

    def test_many_comparisons(self):
        for _ in xrange(4):
            self.test_compare_specialization()


if __name__ == '__main__':
    # suite = unittest.TestLoader().loadTestsFromTestCase(TestSpecializedAStar)

    suite = unittest.TestSuite()
    suite.addTest(TestSpecializedAStar('test_random_grid'))

    unittest.TextTestRunner(verbosity=0).run(suite)


