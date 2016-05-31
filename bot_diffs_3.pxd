from bot_base cimport BaseBot


cdef class Bot(BaseBot):
    cdef:
        float* state1
        float* state2
        float* state3
