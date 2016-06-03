"""
Holds a belief for a hidden state, of the same length as the visible state.

The belief is updated linearly, by multiplying the current state and belief
by a matrix and adding a constant vector. Actions are also evaluated linearly,
weighting each component of the state and each component of the belief and
adding a constant.

Assumes 4 actions.
"""
from cython import cast, cclass, cfunc, locals, returns, sizeof
from libc.stdlib cimport calloc, free
from libc.string cimport memcpy

from bot_base cimport BaseBot

from interface cimport c_do_action, c_get_state


@cclass
class Bot(BaseBot):
    def __cinit__(self, level, *args, **kwargs):
        self.param_shapes = {
            'constant': (level['actions'],),
            'state0l': (level['actions'], level['features']),
            'belief0l': (level['actions'], level['features']),
            'belief_constant': (level['features'],),
            'belief_state0l': (level['features'], level['features']),
            'belief_belief0l': (level['features'], level['features'])
        }
        self.beliefs0 = cast('float*', calloc(level['features'],
                                                        sizeof(float)))
        self.beliefs0t = cast('float*', calloc(level['features'],
                                                        sizeof(float)))

    def __dealloc__(self):
        free(self.beliefs0t)
        free(self.beliefs0)

    @cfunc
    @returns('Bot')
    @locals(state='bint', bot='Bot', beliefs_size='int')
    def clone(self, state=True):
        bot = BaseBot.clone(self, state)
        if state:
            beliefs_size = self.level['features'] * sizeof(float)
            memcpy(bot.beliefs0, self.beliefs0, beliefs_size)
            memcpy(bot.beliefs0t, self.beliefs0t, beliefs_size)
        return bot

    @cfunc
    @returns('dict')
    @locals(dists='dict', emphases='tuple',
            belief_trust='float', belief_lag='float',
            multipliers='dict', params='dict')
    def new_params(self, dists, emphases):
        belief_trust = dists['unit'].rvs()
        belief_lag = dists['unit'].rvs()
        multipliers = self.param_multipliers
        multipliers['belief0l'] = belief_trust
        multipliers['belief_state0l'] = belief_lag
        multipliers['belief_belief0l'] = belief_trust * belief_lag
        params = BaseBot.new_params(self, dists, emphases)
        params['_belief_trust'] = belief_trust
        params['_belief_lag'] = belief_lag
        return params

    @cfunc
    @returns('void')
    @locals(steps='int', step='int', action='int',
            features='int', feature='int', featureb='int',
            constant0='float', constant1='float',
            constant2='float', constant3='float',
            state0l0='float[:]', state0l1='float[:]',
            state0l2='float[:]', state0l3='float[:]',
            belief0l0='float[:]', belief0l1='float[:]',
            belief0l2='float[:]', belief0l3='float[:]',
            belief_constant='float[:]',
            belief_state0l='float[:, :]',
            belief_belief0l='float[:, :]',
            beliefs0='float*', beliefs0t='float*',
            value0='float', value1='float', value2='float', value3='float',
            state0='float*', state0f='float',
            beliefs0f='float', belieft='float')
    def act(self, steps):
        features = self.level['features']
        constant0, constant1, constant2, constant3 = self.params['constant']
        state0l0, state0l1, state0l2, state0l3 = self.params['state0l']
        belief0l0, belief0l1, belief0l2, belief0l3 = self.params['belief0l']
        belief_constant = self.params['belief_constant']
        belief_state0l = self.params['belief_state0l']
        belief_belief0l = self.params['belief_belief0l']
        beliefs0 = self.beliefs0
        beliefs0t = self.beliefs0t
        action = -1

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
