import time
import csv
from a_star.astar import GridAsArray
from a_star.grid_tools import get_random_grid
from a_star.specializers import SpecializedGrid


def time_comparison(max_grid_size=10000000, dimensions=range(2, 6),
                      sizes_per_dim=10, num_samples=10):
    grid_type = SpecializedGrid
    py_times = []
    ctree_times = []
    for num_dim in dimensions:
        max_dim_size = int(max_grid_size**(1.0/num_dim))
        size_inc = max_dim_size/sizes_per_dim
        for dim_size in xrange(size_inc, max_dim_size+1, size_inc):
            print num_dim, " : ", dim_size
            for _ in xrange(num_samples):
                grid = get_random_grid((dim_size,)*num_dim, grid_type)

                start = time.time()
                GridAsArray.get_a_star_cost(*grid)
                finish = time.time()
                interval = finish - start
                py_times.append((num_dim, dim_size, interval))
                print "python: ", interval

                start = time.time()
                SpecializedGrid.get_a_star_cost(*grid)
                finish = time.time()
                interval = finish - start
                ctree_times.append((num_dim, dim_size, interval))
                print "ctree: ", interval

    save_times_csv(py_times, 'astar_times_python.csv')
    save_times_csv(ctree_times, 'astar_times_ctree.csv')


def save_times_csv(time_samples, filename):
    with open(filename, 'wb') as csv_file:
        csv_writer = csv.writer(csv_file)
        for sample in time_samples:
            csv_writer.writerow(sample)

if __name__ == '__main__':
    time_comparison()
