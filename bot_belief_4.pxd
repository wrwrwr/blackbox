from bot_base cimport BaseBot


cdef class Bot(BaseBot):
    cdef:
        float belief0, belief1, belief2, belief3
