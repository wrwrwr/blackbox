"""
Double-state linear regression.

Multiplies each component of the state and the previous state by a weight and
chooses the action for which the sum plus a constant factor is the biggest.

Assumes 4 actions.
"""
from libc.stdlib cimport calloc, free
from libc.string cimport memcpy

from interface cimport c_do_action, c_get_state


cdef class Bot(BaseBot):
    def __cinit__(self, level, *args, **kwargs):
        self.param_shapes = {
            'constant': (level['actions'],),
            'state0l': (level['actions'], level['features']),
            'state1l': (level['actions'], level['features'])
        }
        self.state1 = <float*>calloc(level['features'], sizeof(float))

    def __dealloc__(self):
        free(self.state1)

    cdef Bot clone(self, bint state=True):
        cdef:
            Bot bot = BaseBot.clone(self, state)
            int state_size

        if state:
            state_size = self.level['features'] * sizeof(float)
            memcpy(bot.state1, self.state1, state_size)
        return bot

    cdef void act(self, int steps):
        cdef:
            int features = self.level['features'], \
                state_size = self.level['features'] * sizeof(float), \
                step, feature, action = -1
            float[:, :] state0l = self.params['state0l'], \
                        state1l = self.params['state1l']
            float[:] state0l0 = state0l[0], state0l1 = state0l[1], \
                     state0l2 = state0l[2], state0l3 = state0l[3], \
                     state1l0 = state1l[0], state1l1 = state1l[1], \
                     state1l2 = state1l[2], state1l3 = state1l[3], \
                     constant = self.params['constant']
            float constant0 = constant[0], constant1 = constant[1], \
                  constant2 = constant[2], constant3 = constant[3], \
                  value0, value1, value2, value3, state0f, state1f
            float* state0
            float* state1 = self.state1

        for step in range(steps):
            value0 = constant0
            value1 = constant1
            value2 = constant2
            value3 = constant3
            state0 = c_get_state()
            for feature in range(features):
                state0f = state0[feature]
                state1f = state1[feature]
                value0 += (state0l0[feature] * state0f +
                                                state1l0[feature] * state1f)
                value1 += (state0l1[feature] * state0f +
                                                state1l1[feature] * state1f)
                value2 += (state0l2[feature] * state0f +
                                                state1l2[feature] * state1f)
                value3 += (state0l3[feature] * state0f +
                                                state1l3[feature] * state1f)
            action = (((0 if value0 > value3 else 3)
                                    if value0 > value2 else
                                                (2 if value2 > value3 else 3))
                                if value0 > value1 else
                        ((1 if value1 > value3 else 3)
                                    if value1 > value2 else
                                                (2 if value2 > value3 else 3)))
            c_do_action(action)
            memcpy(state1, state0, state_size)

        self.last_action = action
