"""
Linear regression with the state and three backward finite differences.

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
            'diffs0l': (level['actions'], level['features']),
            'diffs1l': (level['actions'], level['features']),
            'diffs2l': (level['actions'], level['features'])
        }
        self.state1 = <float*>calloc(level['features'], sizeof(float))
        self.state2 = <float*>calloc(level['features'], sizeof(float))
        self.state3 = <float*>calloc(level['features'], sizeof(float))

    def __dealloc__(self):
        free(self.state3)
        free(self.state2)
        free(self.state1)

    cdef Bot clone(self, bint state=True):
        cdef:
            Bot bot = BaseBot.clone(self, state)
            int state_size

        if state:
            state_size = self.level['features'] * sizeof(float)
            memcpy(bot.state1, self.state1, state_size)
            memcpy(bot.state2, self.state2, state_size)
            memcpy(bot.state3, self.state3, state_size)
        return bot

    cdef void act(self, int steps):
        cdef:
            int features = self.level['features'], \
                state_size = features * sizeof(float), \
                step, feature, action = -1
            float[:, :] state0l = self.params['state0l'], \
                        diffs0l = self.params['diffs0l'], \
                        diffs1l = self.params['diffs1l'], \
                        diffs2l = self.params['diffs2l']
            float[:] state0l0 = state0l[0], state0l1 = state0l[1], \
                     state0l2 = state0l[2], state0l3 = state0l[3], \
                     diffs0l0 = diffs0l[0], diffs0l1 = diffs0l[1], \
                     diffs0l2 = diffs0l[2], diffs0l3 = diffs0l[3], \
                     diffs1l0 = diffs1l[0], diffs1l1 = diffs1l[1], \
                     diffs1l2 = diffs1l[2], diffs1l3 = diffs1l[3], \
                     diffs2l0 = diffs2l[0], diffs2l1 = diffs2l[1], \
                     diffs2l2 = diffs2l[2], diffs2l3 = diffs2l[3], \
                     constant = self.params['constant']
            float constant0 = constant[0], constant1 = constant[1], \
                  constant2 = constant[2], constant3 = constant[3], \
                  value0, value1, value2, value3, \
                  state0f, state1f, state2f, diffs0f, diffs1f, diffs2f
            float* state0
            float* state1 = self.state1
            float* state2 = self.state2
            float* state3 = self.state3

        for step in range(steps):
            value0 = constant0
            value1 = constant1
            value2 = constant2
            value3 = constant3
            state0 = c_get_state()
            for feature in range(features):
                state0f = state0[feature]
                state1f = state1[feature]
                state2f = state2[feature]
                diffs0f = state0f - state1f
                diffs1f = diffs0f - state1f + state2f
                diffs2f = state0f - 3 * (state1f - state2f) - state3[feature]
                value0 += (state0l0[feature] * state0f +
                                                diffs0l0[feature] * diffs0f +
                                                diffs1l0[feature] * diffs1f +
                                                diffs2l0[feature] * diffs2f)
                value1 += (state0l1[feature] * state0f +
                                                diffs0l1[feature] * diffs0f +
                                                diffs1l1[feature] * diffs1f +
                                                diffs2l1[feature] * diffs2f)
                value2 += (state0l2[feature] * state0f +
                                                diffs0l2[feature] * diffs0f +
                                                diffs1l2[feature] * diffs1f +
                                                diffs2l2[feature] * diffs2f)
                value3 += (state0l3[feature] * state0f +
                                                diffs0l3[feature] * diffs0f +
                                                diffs1l3[feature] * diffs1f +
                                                diffs2l3[feature] * diffs2f)
            action = (((0 if value0 > value3 else 3)
                                    if value0 > value2 else
                                                (2 if value2 > value3 else 3))
                                if value0 > value1 else
                        ((1 if value1 > value3 else 3)
                                    if value1 > value2 else
                                                (2 if value2 > value3 else 3)))
            c_do_action(action)
            state3, state2, state1 = state2, state1, state3
            memcpy(state1, state0, state_size)

        self.state1 = state1
        self.state2 = state2
        self.state3 = state3
        self.last_action = action
