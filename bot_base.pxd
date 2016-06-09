cdef class BaseBot:
    cdef:
        dict level
        dict param_shapes
        dict param_sizes
        int param_entries
        dict param_multipliers
        int param_choices
        int[::1] choices
        readonly dict params
        int last_action

    cpdef BaseBot clone(self, bint state=?)
    cpdef dict new_params(self, dict, tuple)
    cpdef void vary_params(self, dict, tuple, float, int)
    cpdef void vary_param(self, dict, tuple, float)
    cpdef float evaluate(self, int)
    cpdef void act(self, int)
