"""
Acts randomly, independently of state, with almost equal action probabilities.

This is the default bot used by collectors.
"""
from libc.stdlib cimport rand

from interface cimport c_do_action


cdef class Bot(BaseBot):
    def __cinit__(self, level, *args, **kwargs):
        self.param_shapes = {}

    cdef void act(self, int steps):
        cdef:
            int actions = self.level['actions'], \
                step, action = -1

        for step in range(steps):
            action = rand() % actions
            c_do_action(action)

        self.last_action = action
