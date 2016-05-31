"""
Acts randomly, independently of state, with fixed frequencies of actions.

Assumes 4 actions.
"""
from libc.stdlib cimport rand, RAND_MAX

from numpy.random import randint

from interface cimport c_do_action


cdef class Bot(BaseBot):
    def __cinit__(self, level, *args, **kwargs):
        self.param_shapes = {
            'probs': (level['actions'] - 1,)
        }

    cdef dict new_params(self, dict dists, tuple emphases):
        cdef:
            float[:] probs

        probs = dists['unit'].rvs(size=self.level['actions'] - 1)
        probs.sort()
        return {'probs': probs.astype('f4')}

    cdef void vary_param(self, dict dists, tuple emphases, float change):
        cdef:
            int actions = self.level['actions'], \
                action
            float[:] probs = self.params['probs']
            float prob, min_prob, max_prob

        action = randint(actions - 1)
        prob = probs[action]
        prob += change * (dists['unit'].rvs() - prob)
        min_prob = 0 if action == 0 else probs[action - 1]
        max_prob = 1 if action == actions - 2 else probs[action + 1]
        probs[action] = (min_prob if prob < min_prob else
                                (max_prob if prob > max_prob else prob))

    cdef void act(self, int steps):
        cdef:
            int step, action = -1
            float[:] probs = self.params['probs']
            float prob0 = probs[0], prob1 = probs[1], prob2 = probs[2], \
                  random

        for step in range(steps):
            random = <float>rand() / RAND_MAX
            action = ((0 if random < prob0 else 1)
                                if random < prob1 else
                                        (2 if random < prob2 else 3))
            c_do_action(action)

        self.last_action = action
