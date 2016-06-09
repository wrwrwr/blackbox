"""
Makes sequences of 5 steps randomly, with a set of probabilities for each step.

Assumes 4 actions. Cannot make less than 5 steps at a time.
Some 5% faster than the N implementation.
"""
from cython import cast, ccall, cclass, locals, returns
from libc.stdlib cimport rand, RAND_MAX
from numpy.random import randint

from bot_base cimport BaseBot

from interface cimport c_do_action, c_get_time


@cclass
class Bot(BaseBot):
    @staticmethod
    def shapes(steps, actions, features):
        return {'probs': (5, actions - 1)}

    @ccall
    @returns('dict')
    @locals(dists='dict', emphases='tuple')
    def new_params(self, dists, emphases):
        probs = dists['unit'].rvs(size=self.param_shapes['probs']).astype('f4')
        probs.sort()
        return {'probs': probs}

    @ccall
    @returns('void')
    @locals(dists='dict', emphases='tuple', change='float',
            step='int', actions='int', action='int', probs='float[:, :]',
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
        probs[step, action] = (min_prob if prob < min_prob else
                                    (max_prob if prob > max_prob else prob))

    @ccall
    @returns('void')
    @locals(steps='int', step='int', action='int', remainder='int',
            probs='float[5][3]', random='float')
    def act(self, steps):
        assert steps >= 5
        probs = self.params['probs']
        remainder = c_get_time() % 5
        action = -1

        if remainder != 0:
            if remainder != 4:
                if remainder != 3:
                    if remainder != 2:
                        random = cast('float', rand() / RAND_MAX)
                        action = ((0 if random < probs[1][0] else 1)
                                        if random < probs[1][1] else
                                            (2 if random < probs[1][2] else 3))
                        c_do_action(action)
                    random = cast('float', rand() / RAND_MAX)
                    action = ((0 if random < probs[2][0] else 1)
                                    if random < probs[2][1] else
                                        (2 if random < probs[2][2] else 3))
                    c_do_action(action)
                random = cast('float', rand() / RAND_MAX)
                action = ((0 if random < probs[3][0] else 1)
                                if random < probs[3][1] else
                                    (2 if random < probs[3][2] else 3))
                c_do_action(action)
            random = cast('float', rand() / RAND_MAX)
            action = ((0 if random < probs[4][0] else 1)
                            if random < probs[4][1] else
                                (2 if random < probs[4][2] else 3))
            c_do_action(action)

        steps, remainder = divmod(steps - remainder, 5)

        for step in range(steps):
            random = cast('float', rand() / RAND_MAX)
            action = ((0 if random < probs[0][0] else 1)
                            if random < probs[0][1] else
                                (2 if random < probs[0][2] else 3))
            c_do_action(action)
            random = cast('float', rand() / RAND_MAX)
            action = ((0 if random < probs[1][0] else 1)
                            if random < probs[1][1] else
                                (2 if random < probs[1][2] else 3))
            c_do_action(action)
            random = cast('float', rand() / RAND_MAX)
            action = ((0 if random < probs[2][0] else 1)
                            if random < probs[2][1] else
                                (2 if random < probs[2][2] else 3))
            c_do_action(action)
            random = cast('float', rand() / RAND_MAX)
            action = ((0 if random < probs[3][0] else 1)
                            if random < probs[3][1] else
                                (2 if random < probs[3][2] else 3))
            c_do_action(action)
            random = cast('float', rand() / RAND_MAX)
            action = ((0 if random < probs[4][0] else 1)
                            if random < probs[4][1] else
                                (2 if random < probs[4][2] else 3))
            c_do_action(action)

        if remainder > 0:
            random = cast('float', rand() / RAND_MAX)
            action = ((0 if random < probs[0][0] else 1)
                            if random < probs[0][1] else
                                (2 if random < probs[0][2] else 3))
            c_do_action(action)
            if remainder > 1:
                random = cast('float', rand() / RAND_MAX)
                action = ((0 if random < probs[1][0] else 1)
                                if random < probs[1][1] else
                                    (2 if random < probs[1][2] else 3))
                c_do_action(action)
                if remainder > 2:
                    random = cast('float', rand() / RAND_MAX)
                    action = ((0 if random < probs[2][0] else 1)
                                    if random < probs[2][1] else
                                        (2 if random < probs[2][2] else 3))
                    c_do_action(action)
                    if remainder > 3:
                        random = cast('float', rand() / RAND_MAX)
                        action = ((0 if random < probs[3][0] else 1)
                                        if random < probs[3][1] else
                                            (2 if random < probs[3][2] else 3))
                        c_do_action(action)

        self.last_action = action
