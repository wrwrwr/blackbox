cdef class BaseProcessor:
    cdef:
        tuple data
        int max_features, max_actions, max_steps

    cpdef tuple results(self, tuple)
    cpdef object process(self)
