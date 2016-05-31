"""
Holds a belief for a hidden state, of the same length as the visible state.

The belief is updated linearly, by multiplying the current state and belief
by a matrix and adding a constant vector. Actions are also evaluated linearly,
weighting each component of the state and each component of the belief and
adding a constant.

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
            'belief0l': (level['actions'], level['features']),
            'belief_constant': (level['features'],),
            'belief_state0l': (level['features'], level['features']),
            'belief_belief0l': (level['features'], level['features'])
        }
        self.beliefs0 = <float*>calloc(level['features'], sizeof(float))
        self.beliefs0t = <float*>calloc(level['features'], sizeof(float))

    def __dealloc__(self):
        free(self.beliefs0t)
        free(self.beliefs0)

    cdef Bot clone(self, bint state=True):
        cdef:
            Bot bot = BaseBot.clone(self, state)
            int beliefs_size

        if state:
            beliefs_size = self.level['features'] * sizeof(float)
            memcpy(bot.beliefs0, self.beliefs0, beliefs_size)
            memcpy(bot.beliefs0t, self.beliefs0t, beliefs_size)
        return bot

    cdef dict new_params(self, dict dists, tuple emphases):
        cdef:
            float belief_trust = dists['unit'].rvs(), \
                  belief_lag = dists['unit'].rvs()
            dict multipliers = self.param_multipliers, \
                 params

        multipliers['belief0l'] = belief_trust
        multipliers['belief_state0l'] = belief_lag
        multipliers['belief_belief0l'] = belief_trust * belief_lag
        params = BaseBot.new_params(self, dists, emphases)
        params['_belief_trust'] = belief_trust
        params['_belief_lag'] = belief_lag
        return params

    cdef void act(self, int steps):
        cdef:
            int features = self.level['features'], \
                step, feature, featureb, action = -1
            float[:, :] state0l = self.params['state0l'], \
                        belief0l = self.params['belief0l'], \
                        belief_state0l = self.params['belief_state0l'], \
                        belief_belief0l = self.params['belief_belief0l']
            float[:] state0l0 = state0l[0], state0l1 = state0l[1], \
                     state0l2 = state0l[2], state0l3 = state0l[3], \
                     belief0l0 = belief0l[0], belief0l1 = belief0l[1], \
                     belief0l2 = belief0l[2], belief0l3 = belief0l[3], \
                     constant = self.params['constant'], \
                     belief_constant = self.params['belief_constant']
            float constant0 = constant[0], constant1 = constant[1], \
                  constant2 = constant[2], constant3 = constant[3], \
                  value0, value1, value2, value3, state0f, beliefs0f, belieft
            float* state0
            float* beliefs0 = self.beliefs0
            float* beliefs0t = self.beliefs0t

        for step in range(steps):
            value0 = constant0
            value1 = constant1
            value2 = constant2
            value3 = constant3
            state0 = c_get_state()
            for feature in range(features):
                state0f = state0[feature]
                beliefs0f = beliefs0[feature]
                value0 += (state0l0[feature] * state0f +
                                            belief0l0[feature] * beliefs0f)
                value1 += (state0l1[feature] * state0f +
                                            belief0l1[feature] * beliefs0f)
                value2 += (state0l2[feature] * state0f +
                                            belief0l2[feature] * beliefs0f)
                value3 += (state0l3[feature] * state0f +
                                            belief0l3[feature] * beliefs0f)
            action = (((0 if value0 > value3 else 3)
                                    if value0 > value2 else
                                                (2 if value2 > value3 else 3))
                                if value0 > value1 else
                        ((1 if value1 > value3 else 3)
                                    if value1 > value2 else
                                                (2 if value2 > value3 else 3)))
            c_do_action(action)
            for featureb in range(features):
                belieft = belief_constant[featureb]
                for feature in range(features):
                    belieft += (belief_state0l[featureb, feature] *
                                                            state0[feature] +
                                belief_belief0l[featureb, feature] *
                                                            beliefs0[feature])
                beliefs0t[featureb] = belieft
            beliefs0, beliefs0t = beliefs0t, beliefs0

        self.beliefs0 = beliefs0
        self.beliefs0t = beliefs0t
        self.last_action = action
