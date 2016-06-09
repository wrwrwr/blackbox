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
    @staticmethod
    def shapes(steps, actions, features):
        return {}

    @ccall
    @returns('BaseBot')
    @locals(state='bint', bot='BaseBot')
    def clone(self, state=True):
        assert state  # Otherwise, we'd risk confusing last actions.
        return self

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
