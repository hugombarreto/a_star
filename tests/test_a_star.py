import os
import unittest
from a_star.astar import GridAsArray
from a_star.grid_tools import get_random_grid, load_grid, save_grid


class TestAStar(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestAStar, self).__init__(*args, **kwargs)
        self.grid_type = GridAsArray
        self.grid_size = (100, 50, 3)

    def test_random_grid(self):
        grid, start, finish = get_random_grid(self.grid_size, self.grid_type)
        self._test_grid(grid, start, finish)

    def test_many_random_grids(self):
        for _ in xrange(4):
            self.test_random_grid()

    def test_grid_from_file(self):
        import glob
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "grids/*.p")
        files = glob.glob(path)
        for file_name in files:
            grid, start, finish = load_grid(file_name, self.grid_type)
            self._test_grid(grid, start, finish,
                            "grid from file:" + file_name, False)

    def _test_grid(self, grid, start, finish, msg=None, save=True):
        start_finish_cost = grid.get_a_star_cost(start, finish)
        finish_start_cost = grid.get_a_star_cost(finish, start)

        if save and (start_finish_cost != finish_start_cost):
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "grids")
            save_grid(grid, "random_grid", path, start, finish)

        self.assertEqual(start_finish_cost, finish_start_cost, msg)


if __name__ == '__main__':
    # print "start"
    # suite = unittest.TestLoader().loadTestsFromTestCase(TestAStar)
    # unittest.TextTestRunner(verbosity=0).run(suite)
    # print "finish"
    suite = unittest.TestSuite()
    suite.addTest(TestAStar('test_many_random_grids'))

    unittest.TextTestRunner(verbosity=1).run(suite)
