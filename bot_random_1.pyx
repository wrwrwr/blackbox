"""
Acts randomly, independently of the state, with fixed frequencies of actions.

Assumes 4 actions.
"""
from cython import cast, cclass, cfunc, locals, returns
from libc.stdlib cimport rand, RAND_MAX
from numpy.random import randint

from bot_base cimport BaseBot

from interface cimport c_do_action


@cclass
class Bot(BaseBot):
    def __cinit__(self, level, *args, **kwargs):
        self.param_shapes = {
            'probs': (level['actions'] - 1,)
        }

    @cfunc
    @returns('dict')
    @locals(dists='dict', emphases='tuple')
    def new_params(self, dists, emphases):
        probs = dists['unit'].rvs(size=self.param_shapes['probs']).astype('f4')
        probs.sort()
        return {'probs': probs}

    @cfunc
    @returns('void')
    @locals(dists='dict', emphases='tuple', change='float',
            actions='int', action='int', probs='float[:]',
            prob='float', min_prob='float', max_prob='float')
    def vary_param(self, dists, emphases, change):
        actions = self.level['actions']
        probs = self.params['probs']
        action = randint(actions - 1)
        prob = probs[action]
        prob += change * (dists['unit'].rvs() - prob)
        min_prob = 0 if action == 0 else probs[action - 1]
        max_prob = 1 if action == actions - 2 else probs[action + 1]
        probs[action] = (min_prob if prob < min_prob else
                                (max_prob if prob > max_prob else prob))

    @cfunc
    @returns('void')
    @locals(steps='int', step='int', action='int',
            prob0='float', prob1='float', prob2='float', random='float')
    def act(self, steps):
        prob0, prob1, prob2 = self.params['probs']
        action = -1

        for step in range(steps):
            random = cast('float', rand() / RAND_MAX)
            action = ((0 if random < prob0 else 1)
                                if random < prob1 else
                                        (2 if random < prob2 else 3))
            c_do_action(action)

        self.last_action = action
