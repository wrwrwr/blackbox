from bot_base cimport BaseBot


cdef class BaseCollector:
    cdef:
        dict level
        BaseBot bot
        list checkpoints

    cdef int create_checkpoint(self)
    cdef void load_checkpoint(self, int checkpoint)
    cdef void clear_checkpoints(self)
    cpdef dict collect(self)
