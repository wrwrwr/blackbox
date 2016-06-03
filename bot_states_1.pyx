"""
Double-state linear regression.

Multiplies each component of the state and the previous state by a weight and
chooses the action for which the sum plus a constant factor is the biggest.

Assumes 4 actions.
"""
from cython import cast, cclass, cfunc, locals, returns, sizeof
from libc.stdlib cimport calloc, free
from libc.string cimport memcpy

from bot_base cimport BaseBot

from interface cimport c_do_action, c_get_state


@cclass
class Bot(BaseBot):
    def __cinit__(self, level, *args, **kwargs):
        self.param_shapes = {
            'constant': (level['actions'],),
            'state0l': (level['actions'], level['features']),
            'state1l': (level['actions'], level['features'])
        }
        self.state1 = cast('float*', calloc(level['features'], sizeof(float)))

    def __dealloc__(self):
        free(self.state1)

    @cfunc
    @returns('Bot')
    @locals(state='bint', bot='Bot', state_size='int')
    def clone(self, state=True):
        bot = BaseBot.clone(self, state)
        if state:
            state_size = self.level['features'] * sizeof(float)
            memcpy(bot.state1, self.state1, state_size)
        return bot

    @cfunc
    @returns('void')
    @locals(steps='int', step='int', action='int',
            features='int', feature='int', state_size='int',
            constant0='float', constant1='float',
            constant2='float', constant3='float',
            state0l0='float[:]', state0l1='float[:]',
            state0l2='float[:]', state0l3='float[:]',
            state1l0='float[:]', state1l1='float[:]',
            state1l2='float[:]', state1l3='float[:]',
            value0='float', value1='float', value2='float', value3='float',
            state0='float*', state0f='float', state1='float*', state1f='float')
    def act(self, steps):
        features = self.level['features']
        state_size = features * sizeof(float)
        constant0, constant1, constant2, constant3 = self.params['constant']
        state0l0, state0l1, state0l2, state0l3 = self.params['state0l']
        state1l0, state1l1, state1l2, state1l3 = self.params['state1l']
        state1 = self.state1
        action = -1

        for step in range(steps):
            value0 = constant0
            value1 = constant1
            value2 = constant2
            value3 = constant3
            state0 = c_get_state()
            for feature in range(features):
                state0f = state0[feature]
                state1f = state1[feature]
                value0 += (state0l0[feature] * state0f +
                                                state1l0[feature] * state1f)
                value1 += (state0l1[feature] * state0f +
                                                state1l1[feature] * state1f)
                value2 += (state0l2[feature] * state0f +
                                                state1l2[feature] * state1f)
                value3 += (state0l3[feature] * state0f +
                                                state1l3[feature] * state1f)
            action = (((0 if value0 > value3 else 3)
                                    if value0 > value2 else
                                                (2 if value2 > value3 else 3))
                                if value0 > value1 else
                        ((1 if value1 > value3 else 3)
                                    if value1 > value2 else
                                                (2 if value2 > value3 else 3)))
            c_do_action(action)
            memcpy(state1, state0, state_size)

        self.last_action = action
