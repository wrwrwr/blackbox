"""
Makes sequences of N steps randomly, with a set of probabilities for each step.

Assumes 4 actions.
"""
from cython import cast, ccall, cclass, declare, locals, returns
from libc.stdlib cimport rand, RAND_MAX
from numpy.random import randint

from bot_base cimport BaseBot

from interface cimport c_do_action, c_get_time

n = declare('int', 100)


@cclass
class Bot(BaseBot):
    @staticmethod
    def shapes(steps, actions, features):
        return {'probs': (n, actions - 1)}

    @ccall
    @returns('dict')
    @locals(dists='dict', emphases='tuple')
    def new_params(self, dists, emphases):
        probs = dists['unit'].rvs(size=self.param_shapes['probs']).astype('f4')
        probs.sort()
        return {'_n': n, 'probs': probs}

    @ccall
    @returns('void')
    @locals(dists='dict', emphases='tuple', change='float',
            step='int', actions='int', action='int', probs='float[:, ::1]',
            prob='float', min_prob='float', max_prob='float')
    def vary_param(self, dists, emphases, change):
        actions = self.level['actions']
        probs = self.params['probs']
        step = randint(n)
        action = randint(actions - 1)
        prob = probs[step][action]
        prob += change * (dists['unit'].rvs() - prob)
        min_prob = 0 if action == 0 else probs[step][action - 1]
        max_prob = 1 if action == actions - 2 else probs[step][action + 1]
        probs[step, action] = (min_prob if prob < min_prob else
                                    (max_prob if prob > max_prob else prob))

    @ccall
    @returns('void')
    @locals(steps='int', step='int', action='int', remainder='int',
            probs='float[:, ::1]', random='float')
    def act(self, steps):
        probs = self.params['probs']
        action = -1

        for step in range(c_get_time(), c_get_time() + steps):
            remainder = step % n
            random = cast('float', rand() / RAND_MAX)
            action = ((0 if random < probs[remainder, 0] else 1)
                            if random < probs[remainder, 1] else
                                (2 if random < probs[remainder, 2] else 3))
            c_do_action(action)

        self.last_action = action
