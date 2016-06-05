"""
Holds a belief for a single float value.

The belief is updated linearly, as a function of the current state and the last
belief. Actions are also evaluated linearly, with a coefficient for the belief.

The Belief lag and belief trust meta-parameters influence the generation of
the coefficients. Belief lag weakens the dependence of the belief on the state,
a value of .1 would make the belief update based on the state 10 times smaller
than the belief update based on the previous belief. Similarly, belief trust
weakens the impact of the belief on action evaluation. Both are drawn from the
"unit" distribution when generating new parameters.

Assumes 4 actions.
"""
from cython import cclass, cfunc, locals, returns

from bot_base cimport BaseBot

from interface cimport c_do_action, c_get_state


@cclass
class Bot(BaseBot):
    def __cinit__(self, level, *args, **kwargs):
        self.param_shapes = {
            'free': (level['actions'],),
            'state0l': (level['actions'], level['features']),
            'belief0l': (level['actions'],),
            'belief_free': (1,),
            'belief_state0l': (level['features'],),
            'belief_belief0l': (1,)
        }
        self.belief = 0

    @cfunc
    @returns('Bot')
    @locals(state='bint', bot='Bot')
    def clone(self, state=True):
        bot = BaseBot.clone(self, state)
        if state:
            bot.belief = self.belief
        else:
            bot.belief = 0
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
            free='float[4]', state0l='float[:, :]', belief0l='float[4]',
            belief_free='float', belief_state0l='float[:]',
            belief_belief0l='float', belief='float',
            values='float[4]', state0='float*', state0f='float')
    def act(self, steps):
        features = self.level['features']
        free = self.params['free']
        state0l = self.params['state0l']
        belief0l = self.params['belief0l']
        belief_free = self.params['belief_free']
        belief_state0l = self.params['belief_state0l']
        belief_belief0l = self.params['belief_belief0l']
        belief = self.belief
        action = -1

        for step in range(steps):
            values = free[:]
            values[0] += belief0l[0] * belief
            values[1] += belief0l[1] * belief
            values[2] += belief0l[2] * belief
            values[3] += belief0l[3] * belief
            state0 = c_get_state()
            for feature in range(features):
                state0f = state0[feature]
                values[0] += state0l[0, feature] * state0f
                values[1] += state0l[1, feature] * state0f
                values[2] += state0l[2, feature] * state0f
                values[3] += state0l[3, feature] * state0f
            action = (((0 if values[0] > values[3] else 3)
                                if values[0] > values[2] else
                                        (2 if values[2] > values[3] else 3))
                                if values[0] > values[1] else
                        ((1 if values[1] > values[3] else 3)
                                if values[1] > values[2] else
                                        (2 if values[2] > values[3] else 3)))
            c_do_action(action)
            belief = belief_free + belief_belief0l * belief
            for feature in range(features):
                belief += belief_state0l[feature] * state0[feature]

        self.belief = belief
        self.last_action = action
