import os
import pickle
import unittest
import numpy as np
from os import listdir
from a_star.astar import GridAsArray


def get_random_grid(dimension, grid_type, barrier_probability=0.3):
    if not isinstance(grid_type, type):
        grid_type = type(grid_type)
    generate_barriers = np.vectorize(lambda i: np.inf
        if i < barrier_probability else np.rint(i*1000)+1)
    grid_array = generate_barriers(np.random.rand(*dimension))

    margin_grid = np.pad(grid_array, 1, 'constant', constant_values=-1)
    margin_grid[margin_grid < 0] = np.inf

    start = tuple(np.random.random_integers(1, i) for i in dimension)
    finish = tuple(np.random.random_integers(1, i) for i in dimension)

    margin_grid[start] = 1
    margin_grid[finish] = 1

    return grid_type(margin_grid), start, finish


def load_grid(filename, grid_class=None):
    grid, start, finish = pickle.load(open(filename, "rb"))
    if grid_class is not None:
        grid = grid_class(grid)
    return grid, start, finish


def save_grid(grid, file_prefix, directory, start_node=None, end_node=None):
    file_name = file_prefix + ".p"
    files_in_directory = listdir(directory)
    file_counter = 2
    while file_name in files_in_directory:
        file_name = file_prefix + str(file_counter) + ".p"
        file_counter += 1
    complete = (grid.grid, start_node, end_node)
    pickle.dump(complete, open(os.path.join(directory, file_name), "wb"))
    print 'saved: "' + directory + file_name + '"'


class TestAStar(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestAStar, self).__init__(*args, **kwargs)
        self.grid_type = GridAsArray
        self.grid_size = (4096, 2160)

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
        start_finish_cost = self._get_a_star_cost(grid, start, finish)
        finish_start_cost = self._get_a_star_cost(grid, finish, start)

        if save and (start_finish_cost != finish_start_cost):
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "grids")
            save_grid(grid, "random_grid", path, start, finish)

        self.assertEqual(start_finish_cost, finish_start_cost, msg)

    def _get_a_star_cost(self, grid, start, finish):
        path_trace = grid.a_star(start, finish)
        if tuple(finish) not in path_trace:
            return None

        cost = 0
        node = finish
        while tuple(node) != tuple(start):
            current_parent = path_trace[tuple(node)].parent
            cost += grid.get_neighbor_edges(current_parent)[tuple(node)]
            node = current_parent

        return cost


if __name__ == '__main__':
    print "start"
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAStar)
    unittest.TextTestRunner(verbosity=0).run(suite)
    print "finish"
