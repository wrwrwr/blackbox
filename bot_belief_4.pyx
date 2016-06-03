"""
Holds a belief for 4 float values.

The beliefs are updated linearly, by multiplying the current state and beliefs
by a matrix and adding a vector. Actions are also evaluated linearly, weighting
each component of the state and each belief and adding a constant.

Assumes 4 actions.
"""
from cython import cclass, cfunc, locals, returns

from bot_base cimport BaseBot

from interface cimport c_do_action, c_get_state


@cclass
class Bot(BaseBot):
    def __cinit__(self, level, *args, **kwargs):
        self.param_shapes = {
            'constant': (level['actions'],),
            'state0l': (level['actions'], level['features']),
            'belief0l': (level['actions'], 4),
            'belief_constant': (4,),
            'belief_state0l': (4, level['features']),
            'belief_belief0l': (4, 4)
        }
        self.belief0 = self.belief1 = self.belief2 = self.belief3 = 0

    @cfunc
    @returns('Bot')
    @locals(state='bint', bot='Bot')
    def clone(self, state=True):
        bot = BaseBot.clone(self, state)
        if state:
            bot.belief0 = self.belief0
            bot.belief1 = self.belief1
            bot.belief2 = self.belief2
            bot.belief3 = self.belief3
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
            features='int', feature='int',
            constant0='float', constant1='float',
            constant2='float', constant3='float',
            state0l0='float[:]', state0l1='float[:]',
            state0l2='float[:]', state0l3='float[:]',
            belief0l0='float[:]', belief0l1='float[:]',
            belief0l2='float[:]', belief0l3='float[:]',
            belief_constant0='float', belief_constant1='float',
            belief_constant2='float', belief_constant3='float',
            belief_state0l0='float[:]', belief_state0l1='float[:]',
            belief_state0l2='float[:]', belief_state0l3='float[:]',
            belief_belief0l0='float[:]', belief_belief0l1='float[:]',
            belief_belief0l2='float[:]', belief_belief0l3='float[:]',
            belief0='float', belief0t='float',
            belief1='float', belief1t='float',
            belief2='float', belief2t='float',
            belief3='float', belief3t='float',
            value0='float', value1='float', value2='float', value3='float',
            state0='float*', state0f='float')
    def act(self, steps):
        features = self.level['features']
        constant0, constant1, constant2, constant3 = self.params['constant']
        state0l0, state0l1, state0l2, state0l3 = self.params['state0l']
        belief0l0, belief0l1, belief0l2, belief0l3 = self.params['belief0l']
        (belief_constant0, belief_constant1, belief_constant2,
                            belief_constant3) = self.params['belief_constant']
        (belief_state0l0, belief_state0l1, belief_state0l2,
                            belief_state0l3) = self.params['belief_state0l']
        (belief_belief0l0, belief_belief0l1, belief_belief0l2,
                            belief_belief0l3) = self.params['belief_belief0l']
        belief0 = self.belief0
        belief1 = self.belief1
        belief2 = self.belief2
        belief3 = self.belief3
        action = -1

        for step in range(steps):
            value0 = (constant0 + belief0l0[0] * belief0 +
                                  belief0l0[1] * belief1 +
                                  belief0l0[2] * belief2 +
                                  belief0l0[3] * belief3)
            value1 = (constant1 + belief0l1[0] * belief0 +
                                  belief0l1[1] * belief1 +
                                  belief0l1[2] * belief2 +
                                  belief0l1[3] * belief3)
            value2 = (constant2 + belief0l2[0] * belief0 +
                                  belief0l2[1] * belief1 +
                                  belief0l2[2] * belief2 +
                                  belief0l2[3] * belief3)
            value3 = (constant3 + belief0l3[0] * belief0 +
                                  belief0l3[1] * belief1 +
                                  belief0l3[2] * belief2 +
                                  belief0l3[3] * belief3)
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
            belief0t = (belief_constant0 + belief_belief0l0[0] * belief0 +
                                           belief_belief0l0[1] * belief1 +
                                           belief_belief0l0[2] * belief2 +
                                           belief_belief0l0[3] * belief3)
            belief1t = (belief_constant1 + belief_belief0l1[0] * belief0 +
                                           belief_belief0l1[1] * belief1 +
                                           belief_belief0l1[2] * belief2 +
                                           belief_belief0l1[3] * belief3)
            belief2t = (belief_constant2 + belief_belief0l2[0] * belief0 +
                                           belief_belief0l2[1] * belief1 +
                                           belief_belief0l2[2] * belief2 +
                                           belief_belief0l2[3] * belief3)
            belief3t = (belief_constant3 + belief_belief0l3[0] * belief0 +
                                           belief_belief0l3[1] * belief1 +
                                           belief_belief0l3[2] * belief2 +
                                           belief_belief0l3[3] * belief3)
            for feature in range(features):
                state0f = state0[feature]
                belief0t += belief_state0l0[feature] * state0f
                belief1t += belief_state0l1[feature] * state0f
                belief2t += belief_state0l2[feature] * state0f
                belief3t += belief_state0l3[feature] * state0f
            belief0 = belief0t
            belief1 = belief1t
            belief2 = belief2t
            belief3 = belief3t

        self.belief0 = belief0
        self.belief1 = belief1
        self.belief2 = belief2
        self.belief3 = belief3
        self.last_action = action
