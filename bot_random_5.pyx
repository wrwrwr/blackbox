"""
Makes sequences of 5 steps randomly, with a set of probabilities for each step.

Assumes 4 actions. Cannot make less than 5 steps at a time.
Some 10-15% faster than the N implementation.
"""
from libc.stdlib cimport rand, RAND_MAX

from numpy cimport ndarray
from numpy.random import randint

from interface cimport c_do_action, c_get_time


cdef class Bot(BaseBot):
    def __cinit__(self, level, *args, **kwargs):
        self.param_shapes = {
            'probs': (5, level['actions'] - 1)
        }

    cdef dict new_params(self, dict dists, tuple emphases):
        cdef:
            ndarray[float, ndim=2] probs

        probs = dists['unit'].rvs(size=self.param_shapes['probs']).astype('f4')
        probs.sort()
        return {'probs': probs}

    cdef void vary_param(self, dict dists, tuple emphases, float change):
        cdef:
            int actions = self.level['actions'], \
                action, step
            float[:, :] probs = self.params['probs']
            float prob, min_prob, max_prob

        step = randint(5)
        action = randint(actions - 1)
        prob = probs[step][action]
        prob += change * (dists['unit'].rvs() - prob)
        min_prob = 0 if action == 0 else probs[step][action - 1]
        max_prob = 1 if action == actions - 2 else probs[step][action + 1]
        probs[step][action] = (min_prob if prob < min_prob else
                                    (max_prob if prob > max_prob else prob))

    cdef void act(self, int steps):
        cdef:
            int rest, step, action = -1
            float[:, :] probs = self.params['probs']
            float prob00 = probs[0, 0], prob01 = probs[0, 1], \
                  prob02 = probs[0, 2], \
                  prob10 = probs[1, 0], prob11 = probs[1, 1], \
                  prob12 = probs[1, 2], \
                  prob20 = probs[2, 0], prob21 = probs[2, 1], \
                  prob22 = probs[2, 2], \
                  prob30 = probs[3, 0], prob31 = probs[3, 1], \
                  prob32 = probs[3, 2], \
                  prob40 = probs[4, 0], prob41 = probs[4, 1], \
                  prob42 = probs[4, 2], \
                  random

        assert steps >= 5
        rest = c_get_time() % 5

        if rest != 0:
            if rest != 4:
                if rest != 3:
                    if rest != 2:
                        random = <float>rand() / RAND_MAX
                        action = ((0 if random < prob10 else 1)
                                            if random < prob11 else
                                                (2 if random < prob12 else 3))
                        c_do_action(action)
                    random = <float>rand() / RAND_MAX
                    action = ((0 if random < prob20 else 1)
                                        if random < prob21 else
                                                (2 if random < prob22 else 3))
                    c_do_action(action)
                random = <float>rand() / RAND_MAX
                action = ((0 if random < prob30 else 1)
                                    if random < prob31 else
                                            (2 if random < prob32 else 3))
                c_do_action(action)
            random = <float>rand() / RAND_MAX
            action = ((0 if random < prob40 else 1)
                                if random < prob41 else
                                        (2 if random < prob42 else 3))
            c_do_action(action)

        steps, rest = divmod(steps - rest, 5)

        for step in range(steps):
            random = <float>rand() / RAND_MAX
            action = ((0 if random < prob00 else 1)
                                if random < prob01 else
                                        (2 if random < prob02 else 3))
            c_do_action(action)
            random = <float>rand() / RAND_MAX
            action = ((0 if random < prob10 else 1)
                                if random < prob11 else
                                        (2 if random < prob12 else 3))
            c_do_action(action)
            random = <float>rand() / RAND_MAX
            action = ((0 if random < prob20 else 1)
                                if random < prob21 else
                                        (2 if random < prob22 else 3))
            c_do_action(action)
            random = <float>rand() / RAND_MAX
            action = ((0 if random < prob30 else 1)
                                if random < prob31 else
                                        (2 if random < prob32 else 3))
            c_do_action(action)
            random = <float>rand() / RAND_MAX
            action = ((0 if random < prob40 else 1)
                                if random < prob41 else
                                        (2 if random < prob42 else 3))
            c_do_action(action)

        if rest > 0:
            random = <float>rand() / RAND_MAX
            action = ((0 if random < prob00 else 1)
                                if random < prob01 else
                                        (2 if random < prob02 else 3))
            c_do_action(action)
            if rest > 1:
                random = <float>rand() / RAND_MAX
                action = ((0 if random < prob10 else 1)
                                    if random < prob11 else
                                            (2 if random < prob12 else 3))
                c_do_action(action)
                if rest > 2:
                    random = <float>rand() / RAND_MAX
                    action = ((0 if random < prob20 else 1)
                                        if random < prob21 else
                                                (2 if random < prob22 else 3))
                    c_do_action(action)
                    if rest > 3:
                        random = <float>rand() / RAND_MAX
                        action = ((0 if random < prob30 else 1)
                                            if random < prob31 else
                                                (2 if random < prob32 else 3))
                        c_do_action(action)

        self.last_action = action
