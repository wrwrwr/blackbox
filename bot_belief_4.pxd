from bot_base cimport BaseBot


cdef class Bot(BaseBot):
    cdef:
        float[4] beliefs
