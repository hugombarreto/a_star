import pickle
import unittest
import numpy as np
from os import listdir
from AStarSpecializer.a_star import GridAsArray


def get_random_grid(dimension, grid_type, barrier_probability=0.3):
    if not isinstance(grid_type, type):
        grid_type = type(grid_type)
    generate_barriers = np.vectorize(lambda i: np.inf
        if i < barrier_probability else np.rint(i*1000)+1)
    grid_array = generate_barriers(np.random.rand(*dimension))

    start = tuple(np.random.random_integers(0, i - 1) for i in dimension)
    finish = tuple(np.random.random_integers(0, i - 1) for i in dimension)

    grid_array[start] = 1
    grid_array[finish] = 1
    return grid_type(grid_array), start, finish


def load_grid(filename):
    grid, start, finish = pickle.load(open(filename, "rb"))
    return grid, start, finish


def save_grid(grid, file_prefix, directory, start_node=None, end_node=None):
    file_name = file_prefix + ".p"
    files_in_directory = listdir(directory)
    file_counter = 2
    while file_name in files_in_directory:
        file_name = file_prefix + str(file_counter) + ".p"
        file_counter += 1
    pickle.dump((grid, start_node, end_node),
                open(directory + file_name, "wb"))
    print 'saved: "' + directory + file_name + '"'


class TestAStar(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestAStar, self).__init__(*args, **kwargs)
        self.grid_type = GridAsArray
        self.grid_size = (100, 100)

    def test_random_grid(self):
        self._test_grid(*get_random_grid(self.grid_size, self.grid_type))

    def test_many_random_grids(self):
        for i in xrange(100):
            self.test_random_grid()

    def test_grid_from_file(self):
        import glob
        files = glob.glob("grids/*.p")
        for file_name in files:
            grid, start, finish = load_grid(file_name)
            self._test_grid(grid, start, finish, "grid from file:" + file_name)

    def _test_grid(self, grid, start, finish, msg=None):
        start_finish_cost = self._get_a_star_cost(grid, start, finish)
        finish_start_cost = self._get_a_star_cost(grid, finish, start)

        if start_finish_cost != finish_start_cost:
            save_grid(grid, "random_grid", "boards/", start, finish)

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
