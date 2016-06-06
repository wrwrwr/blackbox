"""
Sums weighted components of the state and adds a free factor, with a different
vector of weights and factor for each action, then chooses the action for which
the sum is the biggest.

Assumes 4 actions.
"""
from cython import ccall, cclass, locals, returns

from bot_base cimport BaseBot

from interface cimport c_do_action, c_get_state


@cclass
class Bot(BaseBot):
    def __cinit__(self, level, *args, **kwargs):
        self.param_shapes = {
            'free': (level['actions'],),
            'state0l': (level['actions'], level['features'])
        }

    @ccall
    @returns('void')
    @locals(steps='int', step='int', action='int',
            features='int', feature='int',
            free='float[4]', state0l='float[:, :]',
            values='float[4]', state0='float*', state0f='float')
    def act(self, steps):
        features = self.level['features']
        free = self.params['free']
        state0l = self.params['state0l']
        action = -1

        for step in range(steps):
            values = free[:]
            state0 = c_get_state()
            for feature in range(features):
                state0f = state0[feature]
                values[0] += state0l[0, feature] * state0f
                values[1] += state0l[1, feature] * state0f
                values[2] += state0l[2, feature] * state0f
                values[3] += state0l[3, feature] * state0f
            action = (((0 if values[0] > values[3] else 3)
                                if values[0] > values[2] else
                                        (2 if values[2] > values[3] else 3))
                                if values[0] > values[1] else
                        ((1 if values[1] > values[3] else 3)
                                if values[1] > values[2] else
                                        (2 if values[2] > values[3] else 3)))
            c_do_action(action)

        self.last_action = action
