"""
Linear regression with the state and its backward finite differences.

Multiplies each component of the current state and the difference between it
and the last state by a weight and chooses the action for which the sum plus
a constant is the biggest. Similar to double-state linear, but with different
training accents.

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
            'diffs0l': (level['actions'], level['features'])
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
                state_size = features * sizeof(float), \
                step, feature, action = -1
            float[:, :] state0l = self.params['state0l'], \
                        diffs0l = self.params['diffs0l']
            float[:] state0l0 = state0l[0], state0l1 = state0l[1], \
                     state0l2 = state0l[2], state0l3 = state0l[3], \
                     diffs0l0 = diffs0l[0], diffs0l1 = diffs0l[1], \
                     diffs0l2 = diffs0l[2], diffs0l3 = diffs0l[3], \
                     constant = self.params['constant']
            float constant0 = constant[0], constant1 = constant[1], \
                  constant2 = constant[2], constant3 = constant[3], \
                  value0, value1, value2, value3, state0f, diffs0f
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
                diffs0f = state0f - state1[feature]
                value0 += (state0l0[feature] * state0f +
                                                diffs0l0[feature] * diffs0f)
                value1 += (state0l1[feature] * state0f +
                                                diffs0l1[feature] * diffs0f)
                value2 += (state0l2[feature] * state0f +
                                                diffs0l2[feature] * diffs0f)
                value3 += (state0l3[feature] * state0f +
                                                diffs0l3[feature] * diffs0f)
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
