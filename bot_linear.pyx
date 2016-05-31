"""
Multiplies each component of the state by a weight and chooses the action
for which the sum plus a constant factor is the biggest.

Assumes 4 actions.
"""
from interface cimport c_do_action, c_get_state


cdef class Bot(BaseBot):
    def __cinit__(self, level, *args, **kwargs):
        self.param_shapes = {
            'constant': (level['actions'],),
            'state0l': (level['actions'], level['features'])
        }

    cdef void act(self, int steps):
        cdef:
            int features = self.level['features'], \
                step, feature, action = -1
            float[:, :] state0l = self.params['state0l']
            float[:] state0l0 = state0l[0], state0l1 = state0l[1], \
                     state0l2 = state0l[2], state0l3 = state0l[3], \
                     constant = self.params['constant']
            float constant0 = constant[0], constant1 = constant[1], \
                  constant2 = constant[2], constant3 = constant[3], \
                  value0, value1, value2, value3, state0f
            float* state0

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

        self.last_action = action
