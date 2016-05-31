"""
Diffs 4 with support for multiple parameter sets.

Assumes 4 actions.
"""
from libc.stdlib cimport calloc, free
from libc.string cimport memcpy

from interface cimport c_do_action, c_get_state, c_get_time


cdef class Bot(BaseBot):
    def __cinit__(self, level, *args, **kwargs):
        self.param_shapes = {
            'constant': (level['actions'],),
            'state0l': (level['actions'], level['features']),
            'diffs0l': (level['actions'], level['features']),
            'diffs1l': (level['actions'], level['features']),
            'diffs2l': (level['actions'], level['features']),
            'diffs3l': (level['actions'], level['features'])
        }
        self.state1 = <float*>calloc(level['features'], sizeof(float))
        self.state2 = <float*>calloc(level['features'], sizeof(float))
        self.state3 = <float*>calloc(level['features'], sizeof(float))
        self.state4 = <float*>calloc(level['features'], sizeof(float))

    def __dealloc__(self):
        free(self.state4)
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
            memcpy(bot.state4, self.state4, state_size)
        return bot

    cdef void act(self, int steps):
        cdef:
            int features = self.level['features'], \
                state_size = features * sizeof(float), \
                time, step, choice, feature, action = -1
            int[:] choices = self.choices
            float[:, :, :] state0l = self.params['state0l'], \
                           diffs0l = self.params['diffs0l'], \
                           diffs1l = self.params['diffs1l'], \
                           diffs2l = self.params['diffs2l'], \
                           diffs3l = self.params['diffs3l']
            float[:, :] state0l0 = state0l[0], state0l1 = state0l[1], \
                        state0l2 = state0l[2], state0l3 = state0l[3], \
                        diffs0l0 = diffs0l[0], diffs0l1 = diffs0l[1], \
                        diffs0l2 = diffs0l[2], diffs0l3 = diffs0l[3], \
                        diffs1l0 = diffs1l[0], diffs1l1 = diffs1l[1], \
                        diffs1l2 = diffs1l[2], diffs1l3 = diffs1l[3], \
                        diffs2l0 = diffs2l[0], diffs2l1 = diffs2l[1], \
                        diffs2l2 = diffs2l[2], diffs2l3 = diffs2l[3], \
                        diffs3l0 = diffs3l[0], diffs3l1 = diffs3l[1], \
                        diffs3l2 = diffs3l[2], diffs3l3 = diffs3l[3], \
                        constant = self.params['constant']
            float[:] constant0 = constant[0], constant1 = constant[1], \
                     constant2 = constant[2], constant3 = constant[3]
            float value0, value1, value2, value3, \
                  state0f, state1f, state2f, state3f, \
                  diffs0f, diffs1f, diffs2f, diffs3f
            float* state0
            float* state1 = self.state1
            float* state2 = self.state2
            float* state3 = self.state3
            float* state4 = self.state4

        time = c_get_time()

        for step in range(steps):
            choice = choices[time]
            value0 = constant0[choice]
            value1 = constant1[choice]
            value2 = constant2[choice]
            value3 = constant3[choice]
            state0 = c_get_state()
            for feature in range(features):
                state0f = state0[feature]
                state1f = state1[feature]
                state2f = state2[feature]
                state3f = state3[feature]
                diffs0f = state0f - state1f
                diffs1f = diffs0f - state1f + state2f
                diffs2f = state0f - 3 * (state1f - state2f) - state3f
                diffs3f = (state0f - 4 * (state1f + state3f) +
                                                6 * state2f + state4[feature])
                value0 += (state0l0[feature, choice] * state0f +
                                        diffs0l0[feature, choice] * diffs0f +
                                        diffs1l0[feature, choice] * diffs1f +
                                        diffs2l0[feature, choice] * diffs2f +
                                        diffs3l0[feature, choice] * diffs3f)
                value1 += (state0l1[feature, choice] * state0f +
                                        diffs0l1[feature, choice] * diffs0f +
                                        diffs1l1[feature, choice] * diffs1f +
                                        diffs2l1[feature, choice] * diffs2f +
                                        diffs3l1[feature, choice] * diffs3f)
                value2 += (state0l2[feature, choice] * state0f +
                                        diffs0l2[feature, choice] * diffs0f +
                                        diffs1l2[feature, choice] * diffs1f +
                                        diffs2l2[feature, choice] * diffs2f +
                                        diffs3l2[feature, choice] * diffs3f)
                value3 += (state0l3[feature, choice] * state0f +
                                        diffs0l3[feature, choice] * diffs0f +
                                        diffs1l3[feature, choice] * diffs1f +
                                        diffs2l3[feature, choice] * diffs2f +
                                        diffs3l3[feature, choice] * diffs3f)
            action = (((0 if value0 > value3 else 3)
                                    if value0 > value2 else
                                                (2 if value2 > value3 else 3))
                                if value0 > value1 else
                        ((1 if value1 > value3 else 3)
                                    if value1 > value2 else
                                                (2 if value2 > value3 else 3)))
            c_do_action(action)
            memcpy(state4, state3, state_size)
            memcpy(state3, state2, state_size)
            memcpy(state2, state1, state_size)
            memcpy(state1, state0, state_size)
            time += 1

        self.last_action = action