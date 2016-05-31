"""
Makes sequences of N steps randomly, with a set of probabilities for each step.

Assumes 4 actions.
"""
from libc.stdlib cimport rand, RAND_MAX

from numpy.random import randint

from interface cimport c_do_action, c_get_time

cdef int n = 100


cdef class Bot(BaseBot):
    def __cinit__(self, level, *args, **kwargs):
        self.param_shapes = {
            'probs': (n, level['actions'] - 1)
        }

    cdef dict new_params(self, dict dists, tuple emphases):
        cdef:
            float[:] probs

        probs = dists['unit'].rvs(size=(n, self.level['actions'] - 1))
        probs.sort()
        return {'_n': n, 'probs': probs.astype('f4')}

    cdef void vary_param(self, dict dists, tuple emphases, float change):
        cdef:
            int actions = self.level['actions'], \
                step, action
            float[:, :] probs = self.params['probs']
            float prob, min_prob, max_prob

        step = randint(n)
        action = randint(actions - 1)
        prob = probs[step][action]
        prob += change * (dists['unit'].rvs() - prob)
        min_prob = 0 if action == 0 else probs[step][action - 1]
        max_prob = 1 if action == actions - 2 else probs[step][action + 1]
        probs[step][action] = (min_prob if prob < min_prob else
                                    (max_prob if prob > max_prob else prob))

    cdef void act(self, int steps):
        cdef:
            int time, step, action = -1
            float[:, :] probs = self.params['probs']
            float[:] probsn
            float prob0, prob1, prob2, random

        time = c_get_time()

        for step in range(time, time + steps):
            probsn = probs[step % n] 
            prob0 = probsn[0]
            prob1 = probsn[1]
            prob2 = probsn[2]
            random = <float>rand() / RAND_MAX
            action = ((0 if random < prob0 else 1)
                                if random < prob1 else
                                        (2 if random < prob2 else 3))
            c_do_action(action)

        self.last_action = action
