"""
Holds a belief for a hidden state, of the same length as the visible state.

Assumes 4 actions.
"""
from cython import cast, ccall, cclass, locals, returns, sizeof
from libc.stdlib cimport calloc, free
from libc.string cimport memcpy

from bot_base cimport BaseBot

from interface cimport c_do_action, c_get_state


@cclass
class Bot(BaseBot):
    @staticmethod
    def shapes(features, actions):
        return {
            'free': (actions,),
            'state0l': (actions, features),
            'belief0l': (actions, features),
            'belief_free': (features,),
            'belief_state0l': (features, features),
            'belief_belief0l': (features, features)
        }

    def __cinit__(self, level, *args, **kwargs):
        features = level['features']
        self.beliefs0 = cast('float*', calloc(features, sizeof(float)))
        self.beliefs0t = cast('float*', calloc(features, sizeof(float)))

    def __dealloc__(self):
        free(self.beliefs0t)
        free(self.beliefs0)

    @ccall
    @returns('Bot')
    @locals(state='bint', bot='Bot', beliefs_size='int')
    def clone(self, state=True):
        bot = BaseBot.clone(self, state)
        if state:
            beliefs_size = self.level['features'] * sizeof(float)
            memcpy(bot.beliefs0, self.beliefs0, beliefs_size)
            memcpy(bot.beliefs0t, self.beliefs0t, beliefs_size)
        return bot

    @ccall
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

    @ccall
    @returns('void')
    @locals(steps='int', step='int', action='int',
            features='int', feature='int', featureb='int',
            free='float[4]', state0l='float[:, :]', belief0l='float[:, :]',
            belief_free='float[:]', belief_state0l='float[:, :]',
            belief_belief0l='float[:, :]',
            beliefs0='float*', beliefs0t='float*',
            values='float[4]', state0='float*', state0f='float',
            beliefs0f='float', belieft='float')
    def act(self, steps):
        features = self.level['features']
        free = self.params['free']
        state0l = self.params['state0l']
        belief0l = self.params['belief0l']
        belief_free = self.params['belief_free']
        belief_state0l = self.params['belief_state0l']
        belief_belief0l = self.params['belief_belief0l']
        beliefs0 = self.beliefs0
        beliefs0t = self.beliefs0t
        action = -1

        for step in range(steps):
            values = free[:]
            state0 = c_get_state()
            for feature in range(features):
                state0f = state0[feature]
                beliefs0f = beliefs0[feature]
                values[0] += (state0l[0, feature] * state0f +
                                            belief0l[0, feature] * beliefs0f)
                values[1] += (state0l[1, feature] * state0f +
                                            belief0l[1, feature] * beliefs0f)
                values[2] += (state0l[2, feature] * state0f +
                                            belief0l[2, feature] * beliefs0f)
                values[3] += (state0l[3, feature] * state0f +
                                            belief0l[3, feature] * beliefs0f)
            action = (((0 if values[0] > values[3] else 3)
                                if values[0] > values[2] else
                                        (2 if values[2] > values[3] else 3))
                                if values[0] > values[1] else
                        ((1 if values[1] > values[3] else 3)
                                if values[1] > values[2] else
                                        (2 if values[2] > values[3] else 3)))
            c_do_action(action)
            for featureb in range(features):
                belieft = belief_free[featureb]
                for feature in range(features):
                    belieft += (
                        belief_state0l[featureb, feature] * state0[feature] +
                        belief_belief0l[featureb, feature] * beliefs0[feature])
                beliefs0t[featureb] = belieft
            beliefs0, beliefs0t = beliefs0t, beliefs0

        self.beliefs0 = beliefs0
        self.beliefs0t = beliefs0t
        self.last_action = action
