"""
Holds a belief for a single float value.

The belief is updated linearly, as a function of the current state and the last
belief. Actions are also evaluated linearly, with a coefficient for the belief.

The Belief lag and belief trust meta-parameters influence the generation of
the coefficients. Belief lag weakens the dependence of the belief on the state,
a value of .1 would make the belief update based on the state 10 times smaller
than the belief update based on the previous belief. Similarly, belief trust
weakens the impact of the belief on actions evaluation. Both are drawn from
the "unit" distribution when generating new parameters.

Assumes 4 actions.
"""
from interface cimport c_do_action, c_get_state


cdef class Bot(BaseBot):
    def __cinit__(self, level, *args, **kwargs):
        self.param_shapes = {
            'constant': (level['actions'],),
            'state0l': (level['actions'], level['features']),
            'belief0l': (level['actions'],),
            'belief_constant': (1,),
            'belief_state0l': (level['features'],),
            'belief_belief0l': (1,)
        }
        self.belief = 0

    cdef Bot clone(self, bint state=True):
        cdef Bot bot = BaseBot.clone(self, state)
        if state:
            bot.belief = self.belief
        else:
            bot.belief = 0
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
                step, feature, action = -1
            float[:, :] state0l = self.params['state0l']
            float[:] state0l0 = state0l[0], state0l1 = state0l[1], \
                     state0l2 = state0l[2], state0l3 = state0l[3], \
                     constant = self.params['constant'], \
                     belief0l = self.params['belief0l'], \
                     belief_state0l = self.params['belief_state0l']
            float constant0 = constant[0], constant1 = constant[1], \
                  constant2 = constant[2], constant3 = constant[3], \
                  belief0l0 = belief0l[0], belief0l1 = belief0l[1], \
                  belief0l2 = belief0l[2], belief0l3 = belief0l[3], \
                  belief_constant = self.params['belief_constant'], \
                  belief_belief0l = self.params['belief_belief0l'], \
                  belief = self.belief, \
                  value0, value1, value2, value3, state0f
            float* state0

        for step in range(steps):
            value0 = constant0 + belief0l0 * belief
            value1 = constant1 + belief0l1 * belief
            value2 = constant2 + belief0l2 * belief
            value3 = constant3 + belief0l3 * belief
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
            belief = belief_constant + belief_belief0l * belief
            for feature in range(features):
                belief += belief_state0l[feature] * state0[feature]

        self.belief = belief
        self.last_action = action
