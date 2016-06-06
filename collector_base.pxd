from bot_base cimport BaseBot


cdef class BaseCollector:
    cdef:
        dict level
        BaseBot bot
        list checkpoints

    cpdef int create_checkpoint(self)
    cpdef void load_checkpoint(self, int checkpoint)
    cpdef void clear_checkpoints(self)
    cpdef dict collect(self)
