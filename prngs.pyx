from libc.stdlib cimport srand
from numpy.random import randint, seed as nseed


def seed_prngs(seed=None):
    """
    Initializes NumPy's RandomState and C's stdlib random generators, and
    returns the seed (a urandom or clock-based one if none is given).

    In critical sections you can find some pretty biased usage of rand().
    It had better not been used at all, but is quite a bit faster than other
    options and likely sufficient in the cases it is used.
    """
    if seed is None:
        # We want to store the actual seed, so need to explicitly generate it.
        nseed()
        seed = randint(2 ** 32)
    nseed(seed)
    srand(randint(2 ** 32))
    return seed
