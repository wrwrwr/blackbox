"""
Linear regression with one set of coefficients for the first half of the level
and another set for the second half.

The phases parameter holds the breaks-points between coefficient sets (times).
For this simple bot it is fixed with a single entry at the middle of the level.

Assumes 4 actions.
"""
from numpy import array

from interface cimport c_do_action, c_get_state, c_get_time


cdef class Bot(BaseBot):
    def __cinit__(self, level, *args, **kwargs):
        self.param_shapes = {
            'constantp0': (level['actions'],),
            'constantp1': (level['actions'],),
            'state0p0l': (level['actions'], level['features']),
            'state0p1l': (level['actions'], level['features'])
        }

    def __init__(self, level, *args, **kwargs):
        super().__init__(level, *args, **kwargs)
        self.params['phases'] = array([level['steps'] // 2])

    cdef void act(self, int steps):
        cdef:
            int features = self.level['features'], \
                phase1 = self.params['phases'][0], \
                time, step, feature, action = -1
            float[:, :] state0l
            float[:] state0l0, state0l1, state0l2, state0l3, constant
            float constant0, constant1, constant2, constant3, \
                  value0, value1, value2, value3, state0f
            float* state0

        time = c_get_time()
        if time < phase1:
            constant = self.params['constantp0']
            state0l = self.params['state0p0l']
        else:
            constant = self.params['constantp1']
            state0l = self.params['state0p1l']

        constant0 = constant[0]
        constant1 = constant[1]
        constant2 = constant[2]
        constant3 = constant[3]
        state0l0 = state0l[0]
        state0l1 = state0l[1]
        state0l2 = state0l[2]
        state0l3 = state0l[3]

        for step in range(steps):
            value0 = constant0
            value1 = constant1
            value2 = constant2
            value3 = constant3
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
                constant = self.params['constantp1']
                state0l = self.params['state0p1l']
                constant0 = constant[0]
                constant1 = constant[1]
                constant2 = constant[2]
                constant3 = constant[3]
                state0l0 = state0l[0]
                state0l1 = state0l[1]
                state0l2 = state0l[2]
                state0l3 = state0l[3]

        self.last_action = action
