import pickle
import unittest
import numpy as np
from AStarSpecializer.a_star import Node, GridAsArray
from AStarSpecializer.specializers import get_specialized_a_star_grid


Grid = get_specialized_a_star_grid(GridAsArray)
#Grid = GridAsArray


class GraphGenerator(object):
    barrier_probability = 0.3

    @staticmethod
    def get_random_grid(x, y):
        generate_barriers = np.vectorize(lambda i: np.inf
            if i < GraphGenerator.barrier_probability else np.rint(i*1000)+1)
        grid_array = generate_barriers(np.random.rand(x, y))
        start = (np.random.random_integers(0, x - 1),
                 np.random.random_integers(0, y - 1))
        finish = (np.random.random_integers(0, x - 1),
                  np.random.random_integers(0, y - 1))
        grid_array[start] = 0
        grid_array[finish] = 0
        return Grid(grid_array), start, finish

    @staticmethod
    def get_grid_from_file(filename):
        grid, start, finish = pickle.load(open(filename, "rb"))
        print "file opened"
        return grid, start, finish


class TestAStar(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestAStar, self).__init__(*args, **kwargs)
        self.grid_size = (100, 100)
        self.graph_size = self.grid_size[0] * self.grid_size[1]

    def test_random_grid(self):
        grid, start, finish = GraphGenerator.get_random_grid(*self.grid_size)
       # grid.save("random_grid", "boards/", start, finish)
        self.assertEqual(get_a_star_cost(grid, start, finish),
                         get_a_star_cost(grid, finish, start))
        print "OK"

    def test_grid_from_file(self):
        import glob
        files = glob.glob("boards/grid*.p")
        for file_name in files:
            grid, start, finish = GraphGenerator.get_grid_from_file(file_name)
            self.assertEqual(get_a_star_cost(grid, start, finish),
                             get_a_star_cost(grid, finish, start),
                             "grid from file: " + file_name)
            print "calculate shortest path"


def get_a_star_cost(graph, start, finish):
    path_trace = graph.a_star(Node(start), Node(finish))

    if finish not in path_trace:
        return None

    cost = 0
    current_node = finish

    while current_node != start:
        current_parent = path_trace[current_node].parent
        cost += graph.get_neighbor_edges(current_parent)[current_node]
        current_node = current_parent

    return cost
