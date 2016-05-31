"""
Diffs 1 with support for multiple parameter sets.
"""
from libc.stdlib cimport calloc, free
from libc.string cimport memcpy

from interface cimport c_do_action, c_get_state, c_get_time


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
                time, step, choice, feature, action = -1
            int[:] choices = self.choices
            float[:, :, :] state0l = self.params['state0l'], \
                           diffs0l = self.params['diffs0l']
            float[:, :] state0l0 = state0l[0], state0l1 = state0l[1], \
                        state0l2 = state0l[2], state0l3 = state0l[3], \
                        diffs0l0 = diffs0l[0], diffs0l1 = diffs0l[1], \
                        diffs0l2 = diffs0l[2], diffs0l3 = diffs0l[3], \
                        constant = self.params['constant']
            float[:] constant0 = constant[0], constant1 = constant[1], \
                     constant2 = constant[2], constant3 = constant[3]
            float value0, value1, value2, value3, state0f, diffs0f
            float* state0
            float* state1 = self.state1

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
                diffs0f = state0f - state1[feature]
                value0 += (state0l0[feature, choice] * state0f +
                                        diffs0l0[feature, choice] * diffs0f)
                value1 += (state0l1[feature, choice] * state0f +
                                        diffs0l1[feature, choice] * diffs0f)
                value2 += (state0l2[feature, choice] * state0f +
                                        diffs0l2[feature, choice] * diffs0f)
                value3 += (state0l3[feature, choice] * state0f +
                                        diffs0l3[feature, choice] * diffs0f)
            action = (((0 if value0 > value3 else 3)
                                    if value0 > value2 else
                                                (2 if value2 > value3 else 3))
                                if value0 > value1 else
                        ((1 if value1 > value3 else 3)
                                    if value1 > value2 else
                                                (2 if value2 > value3 else 3)))
            c_do_action(action)
            memcpy(state1, state0, state_size)
            time += 1

        self.last_action = action
