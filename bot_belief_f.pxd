from bot_base cimport BaseBot


cdef class Bot(BaseBot):
    cdef:
        float* beliefs0
        float* beliefs0t
