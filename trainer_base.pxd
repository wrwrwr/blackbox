cdef class BaseTrainer:
    cdef:
        dict level, dists
        tuple config, emphases
        tuple seeds
        int runs

    cpdef tuple train(self)
