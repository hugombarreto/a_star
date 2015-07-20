"""
specializer AStarSpecializer
"""
from a_star.specializers import AStarSpecializer


def mul_3(i):
    return i*3


if __name__ == '__main__':
    c_mul_3 = AStarSpecializer.from_function(mul_3, 'Translator')

    print mul_3(2)
    print c_mul_3(2)

