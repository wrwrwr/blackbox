"""
Calculates a (multivariable) quadratic polynomial over the state for each
action and chooses the one giving the highest value.

Assumes 4 actions.
"""
from interface cimport c_do_action, c_get_state


cdef class Bot(BaseBot):
    def __cinit__(self, level, *args, **kwargs):
        self.param_shapes = {
            'constant': (level['actions'],),
            'state0l': (level['actions'], level['features']),
            'state0q': (level['actions'], level['features'] ** 2)
        }

    cdef void act(self, int steps):
        cdef:
            int features = self.level['features'], \
                step, feature0, feature1, action = -1
            float[:, :, :] state0q = self.params['state0q']
            float[:, :] state0q0 = state0q[0], state0q1 = state0q[1], \
                        state0q2 = state0q[2], state0q3 = state0q[3], \
                        state0l = self.params['state0l']
            float[:] state0l0 = state0l[0], state0l1 = state0l[1], \
                     state0l2 = state0l[2], state0l3 = state0l[3], \
                     constant = self.params['constant']
            float constant0 = constant[0], constant1 = constant[1], \
                  constant2 = constant[2], constant3 = constant[3], \
                  value0, value1, value2, value3, state0f0, state0f01
            float* state0

        for step in range(steps):
            value0 = constant0
            value1 = constant1
            value2 = constant2
            value3 = constant3
            state0 = c_get_state()
            for feature0 in range(features):
                state0f0 = state0[feature0]
                value0 += state0l0[feature0] * state0f0
                value1 += state0l1[feature0] * state0f0
                value2 += state0l2[feature0] * state0f0
                value3 += state0l3[feature0] * state0f0
                for feature1 in range(features):
                    state0f01 = state0[feature0] * state0[feature1]
                    value0 += state0q0[feature0, feature1] * state0f01
                    value1 += state0q1[feature0, feature1] * state0f01
                    value2 += state0q2[feature0, feature1] * state0f01
                    value3 += state0q3[feature0, feature1] * state0f01
            action = (((0 if value0 > value3 else 3)
                                    if value0 > value2 else
                                                (2 if value2 > value3 else 3))
                                if value0 > value1 else
                        ((1 if value1 > value3 else 3)
                                    if value1 > value2 else
                                                (2 if value2 > value3 else 3)))
            c_do_action(action)

        self.last_action = action