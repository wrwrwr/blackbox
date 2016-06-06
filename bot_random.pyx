"""
Acts randomly, independently of the state, with equal action probabilities.

This is the default bot used by collectors. The probabilities are only almost
equal due to a slight random-number generation bias (with correctness given up
for speed).
"""
from cython import ccall, cclass, locals, returns
from libc.stdlib cimport rand

from bot_base cimport BaseBot

from interface cimport c_do_action


@cclass
class Bot(BaseBot):
    def __cinit__(self, level, *args, **kwargs):
        self.param_shapes = {}

    @ccall
    @returns('void')
    @locals(steps='int', step='int', actions='int', action='int')
    def act(self, steps):
        actions = self.level['actions']
        action = -1

        for step in range(steps):
            action = rand() % actions
            c_do_action(action)

        self.last_action = action
