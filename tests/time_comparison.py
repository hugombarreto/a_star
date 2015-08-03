import time
import csv
import unittest
from tests.test_a_star import TestAStar, get_random_grid
from tests.test_specialized_a_star import TestSpecializedAStar


class TimeComparison(TestSpecializedAStar):
    def test_growing_size(self):
        max_grid_size = 50000000
        dimensions = range(2, 6)
        sizes_per_dim = 10
        num_samples = 10

        py_times = []
        ctree_times = []
        for num_dim in dimensions:
            max_dim_size = int(max_grid_size**(1.0/num_dim))
            size_inc = max_dim_size/sizes_per_dim
            for dim_size in xrange(size_inc, max_dim_size+1, size_inc):
                print num_dim, " : ", dim_size
                for _ in xrange(num_samples):
                    grid = get_random_grid((dim_size,)*num_dim, self.grid_type)

                    start = time.time()
                    TestAStar.get_a_star_cost(*grid)
                    finish = time.time()
                    interval = finish - start
                    py_times.append((num_dim, dim_size, interval))
                    print "python: ", interval

                    start = time.time()
                    TestSpecializedAStar.get_a_star_cost(*grid)
                    finish = time.time()
                    interval = finish - start
                    ctree_times.append((num_dim, dim_size, interval))
                    print "ctree: ", interval

        print "saving..."
        with open('astar_times_python.csv', 'wb') as csvfile:
            csvwriter = csv.writer(csvfile)
            for sample in py_times:
                csvwriter.writerow(sample)

        with open('astar_times_ctree.csv', 'wb') as csvfile:
            csvwriter = csv.writer(csvfile)
            for sample in ctree_times:
                csvwriter.writerow(sample)

if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TimeComparison('test_growing_size'))

    unittest.TextTestRunner(verbosity=1).run(suite)
