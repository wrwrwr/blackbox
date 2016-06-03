"""
Makes sequences of 5 steps randomly, with a set of probabilities for each step.

Assumes 4 actions. Cannot make less than 5 steps at a time.
Some 10-15% faster than the N implementation.
"""
from cython import cast, cclass, cfunc, locals, returns
from libc.stdlib cimport rand, RAND_MAX
from numpy.random import randint

from bot_base cimport BaseBot

from interface cimport c_do_action, c_get_time


@cclass
class Bot(BaseBot):
    def __cinit__(self, level, *args, **kwargs):
        self.param_shapes = {
            'probs': (5, level['actions'] - 1)
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
            actions='int', action='int', step='int', probs='float[:, :]',
            prob='float', min_prob='float', max_prob='float')
    def vary_param(self, dists, emphases, change):
        actions = self.level['actions']
        probs = self.params['probs']
        step = randint(5)
        action = randint(actions - 1)
        prob = probs[step][action]
        prob += change * (dists['unit'].rvs() - prob)
        min_prob = 0 if action == 0 else probs[step][action - 1]
        max_prob = 1 if action == actions - 2 else probs[step][action + 1]
        probs[step][action] = (min_prob if prob < min_prob else
                                    (max_prob if prob > max_prob else prob))

    @cfunc
    @returns('void')
    @locals(steps='int', step='int', action='int', rest='int',
            probs='float[:, :]',
            prob00='float', prob01='float', prob02='float',
            prob10='float', prob11='float', prob12='float',
            prob20='float', prob21='float', prob22='float',
            prob30='float', prob31='float', prob32='float',
            prob40='float', prob41='float', prob42='float',
            random='float')
    def act(self, steps):
        assert steps >= 5
        probs = self.params['probs']
        prob00, prob01, prob02 = probs[0]
        prob10, prob11, prob12 = probs[1]
        prob20, prob21, prob22 = probs[2]
        prob30, prob31, prob32 = probs[3]
        prob40, prob41, prob42 = probs[4]
        rest = c_get_time() % 5
        action = -1

        if rest != 0:
            if rest != 4:
                if rest != 3:
                    if rest != 2:
                        random = cast('float', rand() / RAND_MAX)
                        action = ((0 if random < prob10 else 1)
                                            if random < prob11 else
                                                (2 if random < prob12 else 3))
                        c_do_action(action)
                    random = cast('float', rand() / RAND_MAX)
                    action = ((0 if random < prob20 else 1)
                                        if random < prob21 else
                                                (2 if random < prob22 else 3))
                    c_do_action(action)
                random = cast('float', rand() / RAND_MAX)
                action = ((0 if random < prob30 else 1)
                                    if random < prob31 else
                                            (2 if random < prob32 else 3))
                c_do_action(action)
            random = cast('float', rand() / RAND_MAX)
            action = ((0 if random < prob40 else 1)
                                if random < prob41 else
                                        (2 if random < prob42 else 3))
            c_do_action(action)

        steps, rest = divmod(steps - rest, 5)

        for step in range(steps):
            random = cast('float', rand() / RAND_MAX)
            action = ((0 if random < prob00 else 1)
                                if random < prob01 else
                                        (2 if random < prob02 else 3))
            c_do_action(action)
            random = cast('float', rand() / RAND_MAX)
            action = ((0 if random < prob10 else 1)
                                if random < prob11 else
                                        (2 if random < prob12 else 3))
            c_do_action(action)
            random = cast('float', rand() / RAND_MAX)
            action = ((0 if random < prob20 else 1)
                                if random < prob21 else
                                        (2 if random < prob22 else 3))
            c_do_action(action)
            random = cast('float', rand() / RAND_MAX)
            action = ((0 if random < prob30 else 1)
                                if random < prob31 else
                                        (2 if random < prob32 else 3))
            c_do_action(action)
            random = cast('float', rand() / RAND_MAX)
            action = ((0 if random < prob40 else 1)
                                if random < prob41 else
                                        (2 if random < prob42 else 3))
            c_do_action(action)

        if rest > 0:
            random = cast('float', rand() / RAND_MAX)
            action = ((0 if random < prob00 else 1)
                                if random < prob01 else
                                        (2 if random < prob02 else 3))
            c_do_action(action)
            if rest > 1:
                random = cast('float', rand() / RAND_MAX)
                action = ((0 if random < prob10 else 1)
                                    if random < prob11 else
                                            (2 if random < prob12 else 3))
                c_do_action(action)
                if rest > 2:
                    random = cast('float', rand() / RAND_MAX)
                    action = ((0 if random < prob20 else 1)
                                        if random < prob21 else
                                                (2 if random < prob22 else 3))
                    c_do_action(action)
                    if rest > 3:
                        random = cast('float', rand() / RAND_MAX)
                        action = ((0 if random < prob30 else 1)
                                            if random < prob31 else
                                                (2 if random < prob32 else 3))
                        c_do_action(action)

        self.last_action = action
