"""
Linear regression with one set of coefficients for the first half of the level
and another set for the second half.

The phases meta-parameter holds the break-point between the two coefficient
sets (by time fraction, where 0 is the beginning and 1 the end of a level).

In place of this bot consider using linear_m in combination with the --phases
option, or better yet with the comb trainer.

Assumes 4 actions.
"""
from cython import cclass, cfunc, locals, returns
from numpy import array

from bot_base cimport BaseBot

from interface cimport c_do_action, c_get_state, c_get_time


@cclass
class Bot(BaseBot):
    def __cinit__(self, level, *args, **kwargs):
        self.param_shapes = {
            'freep0': (level['actions'],),
            'freep1': (level['actions'],),
            'state0p0l': (level['actions'], level['features']),
            'state0p1l': (level['actions'], level['features'])
        }

    def __init__(self, level, *args, **kwargs):
        super().__init__(level, *args, **kwargs)
        self.params['phases'] = array([level['steps'] // 2])

    @cfunc
    @returns('void')
    @locals(steps='int', step='int', action='int', phase1='int',
            features='int', feature='int',
            free='float[4]', state0l='float[:, :]',
            values='float[4]', state0='float*', state0f='float')
    def act(self, steps):
        features = self.level['features']
        phase1 = self.params['phases'][0]
        action = -1

        if c_get_time() < phase1:
            free = self.params['freep0']
            state0l = self.params['state0p0l']
        else:
            free = self.params['freep1']
            state0l = self.params['state0p1l']

        for step in range(c_get_time(), c_get_time() + steps):
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
            if step == phase1:
                free = self.params['freep1']
                state0l = self.params['state0p1l']

        self.last_action = action
