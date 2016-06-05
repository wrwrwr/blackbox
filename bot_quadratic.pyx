"""
Calculates a (multi-variable) quadratic polynomial over the state for each
action and chooses the one giving the highest value.

Assumes 4 actions.
"""
from cython import cclass, cfunc, locals, returns

from bot_base cimport BaseBot

from interface cimport c_do_action, c_get_state


@cclass
class Bot(BaseBot):
    def __cinit__(self, level, *args, **kwargs):
        self.param_shapes = {
            'free': (level['actions'],),
            'state0l': (level['actions'], level['features']),
            'state0q': (level['actions'], level['features'], level['features'])
        }

    @cfunc
    @returns('void')
    @locals(steps='int', step='int', action='int',
            features='int', feature0='int', feature1='int',
            free='float[4]', state0l='float[:, :]', state0q='float[:, :, :]',
            state0q0='float[:, :]', state0q1='float[:, :]',
            state0q2='float[:, :]', state0q3='float[:, :]',
            values='float[4]',
            state0='float*', state0f0='float', state0f01='float')
    def act(self, steps):
        features = self.level['features']
        free = self.params['free']
        state0l = self.params['state0l']
        state0q = self.params['state0q']
        state0q0 = state0q[0]
        state0q1 = state0q[1]
        state0q2 = state0q[2]
        state0q3 = state0q[3]
        action = -1

        for step in range(steps):
            values = free[:]
            state0 = c_get_state()
            for feature0 in range(features):
                state0f0 = state0[feature0]
                values[0] += state0l[0, feature0] * state0f0
                values[1] += state0l[1, feature0] * state0f0
                values[2] += state0l[2, feature0] * state0f0
                values[3] += state0l[3, feature0] * state0f0
                for feature1 in range(features):
                    state0f01 = state0f0 * state0[feature1]
                    values[0] += state0q0[feature0, feature1] * state0f01
                    values[1] += state0q1[feature0, feature1] * state0f01
                    values[2] += state0q2[feature0, feature1] * state0f01
                    values[3] += state0q3[feature0, feature1] * state0f01
            action = (((0 if values[0] > values[3] else 3)
                                if values[0] > values[2] else
                                        (2 if values[2] > values[3] else 3))
                                if values[0] > values[1] else
                        ((1 if values[1] > values[3] else 3)
                                if values[1] > values[2] else
                                        (2 if values[2] > values[3] else 3)))
            c_do_action(action)

        self.last_action = action
