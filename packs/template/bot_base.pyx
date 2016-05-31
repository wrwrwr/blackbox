"""
Limited bot base, just for evaluation.
"""
from numpy import array, empty


cdef class BaseBot:
    def __init__(self, dict level, dict params):
        cdef:
            int steps = level['steps'], \
                phase, step
            float phase_end

        self.level = level

        phases = params.get('_phases', array([1.], dtype='f4'))
        if len(phases) == 1:
            self.param_choices = 1
            self.choices = None
        else:
            self.param_choices = len(phases)
            self.choices = empty(steps, dtype='i4')
            step = 0
            for phase, phase_end in enumerate(phases):
                while step < phase_end * steps:
                    self.choices[step] = phase
                    step += 1

        self.params = params

    cdef BaseBot clone(self, bint state=True):
        raise NotImplementedError()

    cdef dict new_params(self, dict dists, tuple emphases):
        raise NotImplementedError()

    cdef void vary_param(self, dict dists, tuple emphases, float change):
        raise NotImplementedError()

    cdef void act(self, int steps):
        raise NotImplementedError()
