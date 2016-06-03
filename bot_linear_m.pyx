"""
Linear with support for multiple param sets.
"""
from cython import cclass, cfunc, locals, returns

from bot_base cimport BaseBot

from interface cimport c_do_action, c_get_state, c_get_time


@cclass
class Bot(BaseBot):
    def __cinit__(self, level, *args, **kwargs):
        self.param_shapes = {
            'constant': (level['actions'],),
            'state0l': (level['actions'], level['features'])
        }

    @cfunc
    @returns('void')
    @locals(steps='int', step='int', action='int', time='int',
            features='int', feature='int', choices='int[:]', choice='int',
            constant0='float[:]', constant1='float[:]',
            constant2='float[:]', constant3='float[:]',
            state0l0='float[:, :]', state0l1='float[:, :]',
            state0l2='float[:, :]', state0l3='float[:, :]',
            value0='float', value1='float', value2='float', value3='float',
            state0='float*', state0f='float')
    def act(self, steps):
        features = self.level['features']
        constant0, constant1, constant2, constant3 = self.params['constant']
        state0l0, state0l1, state0l2, state0l3 = self.params['state0l']
        choices = self.choices
        time = c_get_time()
        action = -1

        for step in range(steps):
            choice = choices[time]
            value0 = constant0[choice]
            value1 = constant1[choice]
            value2 = constant2[choice]
            value3 = constant3[choice]
            state0 = c_get_state()
            for feature in range(features):
                state0f = state0[feature]
                value0 += state0l0[feature, choice] * state0f
                value1 += state0l1[feature, choice] * state0f
                value2 += state0l2[feature, choice] * state0f
                value3 += state0l3[feature, choice] * state0f
            action = (((0 if value0 > value3 else 3)
                                    if value0 > value2 else
                                                (2 if value2 > value3 else 3))
                                if value0 > value1 else
                        ((1 if value1 > value3 else 3)
                                    if value1 > value2 else
                                                (2 if value2 > value3 else 3)))
            c_do_action(action)
            time += 1

        self.last_action = action
