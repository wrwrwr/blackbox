cdef class BaseTrainer:
    cdef:
        dict level, dists
        tuple emphases
        tuple seeds
        int runs

    cpdef tuple train(self)
