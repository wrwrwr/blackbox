cdef class BaseProcessor:
    cdef:
        tuple data
        int max_steps, max_actions, max_features

    cpdef tuple results(self, tuple)
    cpdef object process(self)
