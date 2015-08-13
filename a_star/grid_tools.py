import numpy as np
import pickle
import os
from os import listdir


def get_random_grid(dimension, grid_type, barrier_prob=0.3, weight=False):
    if not isinstance(grid_type, type):
        grid_type = type(grid_type)
    generate_barriers = np.vectorize(lambda i: np.inf
        if i < barrier_prob else int(weight) * np.rint(i*1000) + 1)
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
