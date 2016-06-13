cdef class BaseProcessor:
    cdef:
        tuple data, records, meta, levels, bots
        int max_steps, max_actions, max_features

    cpdef tuple results(self, tuple)
    cpdef object process(self)
