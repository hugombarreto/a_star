import unittest
from tests.test_a_star import TestAStar, get_random_grid
from a_star.specializers import SpecializedGrid, decompose_coordinates


class TestSpecializedAStar(TestAStar):
    def __init__(self, *args, **kwargs):
        super(TestSpecializedAStar, self).__init__(*args, **kwargs)
        self.grid_type = SpecializedGrid

    # @profile
    def test_compare_specialization(self):
        grid, start, finish = get_random_grid(self.grid_size, self.grid_type)
        print "Start C"
        c_cost = self.get_a_star_cost(grid, start, finish)
        print "Finish C\nStart Python"
        py_cost = TestAStar.get_a_star_cost(grid, start, finish)
        print "Finish Python"

        self.assertEqual(py_cost, c_cost)

    def test_many_comparisons(self):
        for _ in xrange(4):
            self.test_compare_specialization()

    @staticmethod
    def get_a_star_cost(grid, start, finish):
        path_trace = grid.specialized_a_star(start, finish)

        if start == finish:
            return 0

        if path_trace[finish] < 0:
            return None

        node = finish
        cost = 0
        while node != start:
            cost += grid.grid[node]
            node = decompose_coordinates(path_trace[node], grid.grid_shape)
        return cost


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSpecializedAStar)

    # suite = unittest.TestSuite()
    # suite.addTest(TestSpecializedAStar('test_many_random_grids'))

    unittest.TextTestRunner(verbosity=0).run(suite)


