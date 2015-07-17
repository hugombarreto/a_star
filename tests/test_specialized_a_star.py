import unittest
from tests.test_a_star import TestAStar, get_random_grid
from AStarSpecializer.specializers import SpecializedGrid


class TestSpecializedAStar(TestAStar):
    def __init__(self, *args, **kwargs):
        super(TestSpecializedAStar, self).__init__(*args, **kwargs)
        self.grid_type = SpecializedGrid

    def test_compare_specialization(self):
        grid, start, finish = get_random_grid(self.grid_size, self.grid_type)
        py_cost = super(TestSpecializedAStar, self)._get_a_star_cost(
            grid, start, finish)
        c_cost = self._get_a_star_cost(grid, start, finish)

        self.assertEqual(py_cost, c_cost)

    def _get_a_star_cost(self, grid, start, finish):
        # print graph._grid
        path_trace = grid.specialized_a_star(start, finish)
        path_trace = grid.a_star(start, finish)
        # print "start: ", start
        # print "finish: ", finish
        # print path_trace
        # if tuple(finish) not in path_trace:
        #     return None
        #
        # cost = 0
        # current_node = finish
        #
        # while tuple(current_node) != tuple(start):
        #     current_parent = path_trace[tuple(current_node)].parent
        #     cost += graph.get_neighbor_edges(current_parent)[tuple(current_node)]
        #     current_node = current_parent
        #
        # return cost

        return 0


if __name__ == '__main__':
    print "start"
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSpecializedAStar)
    unittest.TextTestRunner(verbosity=0).run(suite)
    print "finish"
