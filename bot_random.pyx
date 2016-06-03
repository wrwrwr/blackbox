"""
Acts randomly, independently of state, with almost equal action probabilities.

This is the default bot used by collectors.
"""
from cython import cclass, cfunc, locals, returns
from libc.stdlib cimport rand

from bot_base cimport BaseBot

from interface cimport c_do_action


@cclass
class Bot(BaseBot):
    def __cinit__(self, level, *args, **kwargs):
        self.param_shapes = {}

    @cfunc
    @returns('void')
    @locals(steps='int', step='int', actions='int', action='int')
    def act(self, steps):
        actions = self.level['actions']
        action = -1

        for step in range(steps):
            action = rand() % actions
            c_do_action(action)

        self.last_action = action
