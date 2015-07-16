from collections import defaultdict
import pickle
from os import listdir
import heapq
import numpy as np
from AStarSpecializer.np_functional import np_map, np_reduce, np_elementwise


class PriorityQueue(object):
    """Implementation of a Priority Queue using a heap"""

    def __init__(self):
        self._heap = []

    def push(self, item, priority=None):
        """Insert item in the queue"""
        if priority is not None:
            item.set_priority(priority)
        heapq.heappush(self._heap, item)

    def pop(self):
        return heapq.heappop(self._heap)

    def __len__(self):
        return len(self._heap)


class Node(object):
    """Used to make the nodes ordered based on the priority only"""

    def __init__(self, node_id, priority=None):
        self.id = node_id
        self.priority = priority

    def __lt__(self, other):
        return self.priority < other.priority


class Graph(object):
    """Graph class to be used with the A* algorithm"""

    def __init__(self):
        self._edges = {}

    def get_neighbor_edges(self, node_id):
        return self._edges[node_id]

    def _get_neighbor_weight_list(self, node_id):
        return self.get_neighbor_edges(node_id).iteritems()

    def insert_edge(self, a, b, weight):
        if a.get_id() in self._edges:
            self._edges[a.get_id()][b.get_id()] = weight
        else:
            self._edges[a.get_id()] = {b.get_id(): weight}

    def a_star(self, start_id, target_id):
        nodes_info = defaultdict(self.NodeInfo)
        open_list = PriorityQueue()
        open_list.push(Node(start_id, 0))

        start_info = self.NodeInfo()
        start_info.f = 0
        start_info.g = 0
        nodes_info[start_id] = start_info

        while len(open_list) > 0:
            current_node = open_list.pop()
            current_node_id = current_node.id

            if current_node_id == target_id:
                break

            current_node_info = nodes_info[current_node_id]
            current_node_info.closed = True

            for adj_node_id, adj_node_to_parent_weight in \
                    self._get_neighbor_weight_list(current_node_id):
                adj_node_info = nodes_info[adj_node_id]

                if not adj_node_info.closed:
                    g = current_node_info.g + adj_node_to_parent_weight
                    if g < adj_node_info.g:
                        adj_node_info.parent = current_node_id
                        h = self._calculate_heuristic_cost(adj_node_id,
                                                           target_id)
                        adj_node_info.g = g
                        adj_node_info.f = g + h
                        open_list.push(Node(adj_node_id, adj_node_info.f))
                        nodes_info[adj_node_id] = adj_node_info

            nodes_info[current_node_id] = current_node_info

        return nodes_info

    def save(self, file_prefix, directory, start_node=None, end_node=None):
        file_name = file_prefix + ".p"
        files_in_directory = listdir(directory)
        file_counter = 2
        while file_name in files_in_directory:
            file_name = file_prefix + str(file_counter) + ".p"
            file_counter += 1
        pickle.dump((self, start_node, end_node),
                    open(directory + file_name, "wb"))
        print 'saved: "' + directory + file_name + '"'

    @staticmethod
    def _calculate_heuristic_cost(current_node_id, target_node_id):
        # no heuristics by default, works as Dijkstra's shortest path
        return 0

    class NodeInfo(object):
        def __init__(self):
            self.f = None
            self.g = np.inf
            self.parent = None
            self.closed = False


class BaseGrid(Graph):
    """Base class for grids, implements grids heuristic"""

    def __init__(self, grid):
        """
        Args:
          grid (numpy.array): The any dimensions grid, the barriers are
            represented by `numpy.inf`
        """
        super(BaseGrid, self).__init__()
        self.grid_shape = grid.shape
        identity_matrix = np.eye(len(self.grid_shape), dtype=int)
        self.neighborhood_matrix = np.concatenate((-identity_matrix,
                                                   identity_matrix))

    def _get_neighbors(self, grid, node_position):
        no_filter_neighbors = list(node_position + self.neighborhood_matrix)
        neighbors = filter(
            lambda i: np.logical_and((i >= 0), (i < self.grid_shape)).all(),
            no_filter_neighbors)

        weighted_neighbors = []
        for n in neighbors:
            tuple_n = tuple(n)
            weight = grid[tuple_n]
            if weight != np.inf:
                weighted_neighbors.append((tuple_n, weight))
        return weighted_neighbors

    @staticmethod
    def _calculate_heuristic_cost(current_node_id, target_node_id):
        # Using 1-norm
        current_node_id_array = np.array(current_node_id)
        target_node_id_array = np.array(target_node_id)
        coordinates_difference = np_elementwise(lambda x, y: x - y,
                                                current_node_id_array,
                                                target_node_id_array)

        return np_reduce(lambda x, y: x + y, np_map(
            lambda z: -z if z < 0 else z, coordinates_difference))


class GridAsGraph(BaseGrid):
    """This class converts Grids to Graphs."""

    def __init__(self, grid):
        """
        Args:
          grid (numpy.array): The any dimensions grid to be converted
            and used as a graph, the barriers are represented by `numpy.inf`
        """
        super(GridAsGraph, self).__init__(grid)

        it = np.nditer(grid, flags=['multi_index'])
        while not it.finished:
            if it[0] != 1:
                neighbors = self._get_neighbors(grid, it.multi_index)
                for neigh, weight in neighbors:
                    edge = (Node(it.multi_index), Node(tuple(neigh)), weight)
                    self.insert_edge(*edge)
            it.iternext()
        print "ready!"


class GridAsArray(BaseGrid):
    """ This class uses the grid as a numpy.array instead of a graph"""

    def __init__(self, grid):
        """
        Args:
          grid (numpy.array): The any dimensions grid to be used for the
            A* algorithm, the barriers are represented by `numpy.inf`
        """
        super(GridAsArray, self).__init__(grid)
        self._grid = grid

    def insert_edge(self, a, b, weight):
        raise NotImplementedError

    def get_neighbor_edges(self, node_id):
        return dict(self._get_neighbor_weight_list(node_id))

    def _get_neighbor_weight_list(self, node_id):
        return self._get_neighbors(self._grid, node_id)
