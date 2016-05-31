cdef class BaseBot:
    cdef:
        dict level, param_shapes, params
        int param_choices
        int[:] choices
        int last_action

    cdef BaseBot clone(self, bint state=?)
    cdef dict new_params(self, dict, tuple)
    cdef void vary_param(self, dict, tuple, float)
    cdef void act(self, int)
