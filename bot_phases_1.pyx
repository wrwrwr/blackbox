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
    @locals(steps='int', step='int', action='int', time='int', phase1='int',
            features='int', feature='int',
            free='float[:]', state0l='float[:, :]',
            free0='float', free1='float',
            free2='float', free3='float',
            state0l0='float[:]', state0l1='float[:]',
            state0l2='float[:]', state0l3='float[:]',
            value0='float', value1='float', value2='float', value3='float',
            state0='float*', state0f='float')
    def act(self, steps):
        features = self.level['features']
        time = c_get_time()
        phase1 = self.params['phases'][0]
        action = -1

        if time < phase1:
            free = self.params['freep0']
            state0l = self.params['state0p0l']
        else:
            free = self.params['freep1']
            state0l = self.params['state0p1l']

        free0, free1, free2, free3 = free
        state0l0, state0l1, state0l2, state0l3 = state0l

        for step in range(steps):
            value0 = free0
            value1 = free1
            value2 = free2
            value3 = free3
            state0 = c_get_state()
            for feature in range(features):
                state0f = state0[feature]
                value0 += state0l0[feature] * state0f
                value1 += state0l1[feature] * state0f
                value2 += state0l2[feature] * state0f
                value3 += state0l3[feature] * state0f
            action = (((0 if value0 > value3 else 3)
                                    if value0 > value2 else
                                                (2 if value2 > value3 else 3))
                                if value0 > value1 else
                        ((1 if value1 > value3 else 3)
                                    if value1 > value2 else
                                                (2 if value2 > value3 else 3)))
            c_do_action(action)
            time += 1
            if time == phase1:
                free = self.params['freep1']
                state0l = self.params['state0p1l']
                free0, free1, free2, free3 = free
                state0l0, state0l1, state0l2, state0l3 = state0l

        self.last_action = action
