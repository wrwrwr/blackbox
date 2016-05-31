cdef class BaseBot:
    cdef:
        dict level
        dict param_shapes
        dict param_sizes 
        int param_entries
        dict param_multipliers
        int param_choices
        int[:] choices
        readonly dict params
        int last_action

    cdef BaseBot clone(self, bint state=?)
    cdef dict new_params(self, dict, tuple)
    cdef void vary_params(self, dict, tuple, float, int)
    cdef void vary_param(self, dict, tuple, float)
    cpdef float evaluate(self, int)
    cdef void act(self, int)
