"""
Double-state linear regression.

Sums weighted components of the state and the previous state and chooses the
action for which the sum plus a free factor is the biggest.

Assumes 4 actions.
"""
from cython import cast, ccall, cclass, locals, returns, sizeof
from libc.stdlib cimport calloc, free
from libc.string cimport memcpy

from bot_base cimport BaseBot

from interface cimport c_do_action, c_get_state


@cclass
class Bot(BaseBot):
    @staticmethod
    def shapes(features, actions):
        return {
            'free': (actions,),
            'state0l': (actions, features),
            'state1l': (actions, features)
        }

    def __cinit__(self, level, *args, **kwargs):
        self.state1 = cast('float*', calloc(level['features'], sizeof(float)))

    def __dealloc__(self):
        free(self.state1)

    @ccall
    @returns('Bot')
    @locals(state='bint', bot='Bot', state_size='int')
    def clone(self, state=True):
        bot = BaseBot.clone(self, state)
        if state:
            state_size = self.level['features'] * sizeof(float)
            memcpy(bot.state1, self.state1, state_size)
        return bot

    @ccall
    @returns('void')
    @locals(steps='int', step='int', action='int',
            features='int', feature='int', state_size='int',
            free0='float', free1='float',
            free2='float', free3='float',
            state0l0='float[:]', state0l1='float[:]',
            state0l2='float[:]', state0l3='float[:]',
            state1l0='float[:]', state1l1='float[:]',
            state1l2='float[:]', state1l3='float[:]',
            value0='float', value1='float', value2='float', value3='float',
            state0='float*', state0f='float', state1='float*', state1f='float')
    def act(self, steps):
        features = self.level['features']
        state_size = features * sizeof(float)
        free0, free1, free2, free3 = self.params['free']
        state0l0, state0l1, state0l2, state0l3 = self.params['state0l']
        state1l0, state1l1, state1l2, state1l3 = self.params['state1l']
        state1 = self.state1
        action = -1

        for step in range(steps):
            value0 = free0
            value1 = free1
            value2 = free2
            value3 = free3
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
