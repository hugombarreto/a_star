==========================
Creating an A* Specializer
==========================

.. warning:: Read the other file, this is not done yet. Some parts of this one
   were moved to the other file.

Introduction
------------
This tutorial creates a specializer for the A* Algorithm. We will guide you
through the steps you need to create your own specializer. This assumes you
read the **Introduction to Specializers** and are familiar with the basic
structure of specializers, visitors and transformers.

The A* Algorithm is used to find the shortest path between two nodes in a
graph. This algorithm achieves better results by using a heuristics function.
Depending on the application, different heuristics are used. The idea here is
that the specializer user will be able to implement their own heuristic
function in python and use it in the specializer. In this specializer, we're
focusing on Grid Graphs as they're the most common scenario fo the A*.

Writing a specializer usually consists of the following steps:

- `Setup a project`_;
- `Writing the Python Code to be Specialized`_;
- `Create tests`_ that will be used to check the python code and later to test
  the specialized code;
- `Write specialization transformers`_ to help convert the code to C;
- Write specialization transformers to use a platform (like OpenMP, OpenCL,
  CUDA, etc.);


Setup a Project
---------------
Make sure you have ctree installed::

    sudo pip install ctree

Create a specializer project with the help of the ``ctree`` command, ``-sp``
stands for *Start Project*::

    ctree -sp ProjectName

A directory with the project structure will be created inside the current
directory, using the *ProjectName* you provided.

For the A* we are calling the project *a_star*::

    ctree -sp a_star

Go into the directory created. The project structure should look like this:

.. image:: images/project_files.png
   :width: 800px


Writing the Python Code to be Specialized
-----------------------------------------

.. note:: This section provides details on the implementation of the A*
   Algorithm in the python side. If you're in a hurry and just want to know
   about the specializer, you may skip this section and return if you need more
   details about a specific python class.

The python implementation should be placed in the ``a_star/a_star`` directory.

First, to implement the A* algorithm we need a priority queue. Priority queues
can be easily implemented on Python using the heapq library. An implementation
of a Python priority queue can be seen below.

.. _PriorityQueue:

.. code:: python

    import heapq

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

To make sure elements are ordered using the priority only, the Node_ class was
created. A ``Node`` object has a priority and an id but uses only the priority
for the __lt__ method.

.. _Node:

.. code:: python

    class Node(object):
        """Used to make the nodes ordered based on the priority only"""

        def __init__(self, node_id, priority=None):
            self.id = node_id
            self.priority = priority

        def __lt__(self, other):
            return self.priority < other.priority

Now that we have the ``PriorityQueue`` set we may implement the actual A*
Algorithm. For this we created the Graph_ class:

.. _Graph:

.. code:: python

    from collections import defaultdict
    import numpy as np

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

        def _calculate_heuristic_cost(self, current_node_id, target_node_id):
            # no heuristics by default, works as Dijkstra's shortest path
            return 0

        class NodeInfo(object):
            def __init__(self):
                self.f = None
                self.g = np.inf
                self.parent = None
                self.closed = False

The ``Graph`` class has a generic implementation of the A* Algorithm for any
kind of graph. The following class (BaseGrid_) subclasses the ``Graph``
class for the specific case of grids.

.. _BaseGrid:

.. code:: python

    import numpy as np

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
            self.identity_matrix = np.eye(len(self.grid_shape), dtype=int)

        def _get_neighbors(self, grid, node_position):
            no_filter_neighbors = list(np.concatenate(
                (node_position - self.identity_matrix,
                 node_position + self.identity_matrix)))

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

        def _calculate_heuristic_cost(self, current_node_id, target_node_id):
            # Using 1-norm
            current_node_id = np.array(current_node_id)
            target_node_id = np.array(target_node_id)
            return self._calculate_1_norm(np_elementwise(
                lambda x, y: x - y, current_node_id, target_node_id))

        @staticmethod
        def _calculate_1_norm(vector):
            return np_reduce(lambda x, y: x + y,
                             np_map(lambda z: -z if z < 0 else z, vector))


.. GridAsArray
.. code:: python

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


Create Tests
------------
This part is optional but is highly recommended as unit tests make the
development process much easier.

Our tests for the A* will compare the cost for both the path from the origin to
the destination and from the destination to the origin. If our A* code is
correct they must be the same. Note the paths may differ but they must have the
same cost which should be minimal. This procedure doesn't prove the code
correctness but helps tracking errors.

First we create a function to generate random grids:

.. code:: python

    def get_random_grid(dimension, barrier_probability=0.3):
        generate_barriers = np.vectorize(lambda i: np.inf
            if i < barrier_probability else np.rint(i*1000)+1)
        grid_array = generate_barriers(np.random.rand(*dimension))

        start = tuple(np.random.random_integers(0, i - 1) for i in dimension)
        finish = tuple(np.random.random_integers(0, i - 1) for i in dimension)

        grid_array[start] = 1
        grid_array[finish] = 1
        return Grid(grid_array), start, finish

Here ``dimension`` is a tuple with the grid dimensions, it can have any number
of dimensions. This grid has some tiles with infinity cost that we called
"barriers". If the tile is not a "barrier" it will have a random integer cost
associated with it.

Now that we have random grids to test our code, we can write the actual
unittest:

.. code:: python

    class TestAStar(unittest.TestCase):
        def __init__(self, *args, **kwargs):
            super(TestAStar, self).__init__(*args, **kwargs)
            self.grid_size = (10, 10, 10)

        def test_random_grid(self):
            grid, start, finish = get_random_grid(self.grid_size)
            start_finish_cost = get_a_star_cost(grid, start, finish)
            finish_start_cost = get_a_star_cost(grid, finish, start)

            self.assertEqual(start_finish_cost, finish_start_cost)

        def test_many_random_grids(self):
            for i in xrange(100):
                self.test_random_grid()

We're using a function ``get_a_star_cost`` to get the cost of the path found by
the A* Algorithm. This function is implemented as follow:

.. code:: python

    def get_a_star_cost(graph, start, finish):
        path_trace = graph.a_star(start, finish)

        if tuple(finish) not in path_trace:
            return None

        cost = 0
        current_node = finish

        while tuple(current_node) != tuple(start):
            current_parent = path_trace[tuple(current_node)].parent
            cost += graph.get_neighbor_edges(current_parent)[tuple(current_node)]
            current_node = current_parent

        return cost

Now we are done with both the python side implementation of the A* and the
unittest for it. It's time to specialize.

Write Specialization Transformers
---------------------------------
This section explains how to create a specializer for your python code. Here
we create the specializer for the A* Algorithm, but the steps will be similar
to what you need to do for your own specializer.

If you open the ``main.py`` file on ``a_star/a_star/`` you will see it looks
like this:

.. code:: python

    """
    specializer a_star
    """

    from ctree.jit import LazySpecializedFunction


    class a_star(LazySpecializedFunction):

        def transform(self):
            pass

    if __name__ == '__main__':
        pass






Next sections should also contain: C templates
