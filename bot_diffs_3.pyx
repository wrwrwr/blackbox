"""
Linear regression using the state and three backward finite differences.

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
    def shapes(steps, actions, features):
        return {
            'free': (actions,),
            'state0l': (actions, features),
            'diffs0l': (actions, features),
            'diffs1l': (actions, features),
            'diffs2l': (actions, features)
        }

    def __cinit__(self, level, *args, **kwargs):
        self.state1 = cast('float*', calloc(level['features'], sizeof(float)))
        self.state2 = cast('float*', calloc(level['features'], sizeof(float)))
        self.state3 = cast('float*', calloc(level['features'], sizeof(float)))

    def __dealloc__(self):
        free(self.state3)
        free(self.state2)
        free(self.state1)

    @ccall
    @returns('Bot')
    @locals(state='bint', bot='Bot', state_size='int')
    def clone(self, state=True):
        bot = BaseBot.clone(self, state)
        if state:
            state_size = self.level['features'] * sizeof(float)
            memcpy(bot.state1, self.state1, state_size)
            memcpy(bot.state2, self.state2, state_size)
            memcpy(bot.state3, self.state3, state_size)
        return bot

    @ccall
    @returns('void')
    @locals(steps='int', step='int', action='int',
            features='int', feature='int', state_size='int',
            free='float[4]', state0l='float[:, ::1]', diffs0l='float[:, ::1]',
            diffs1l='float[:, ::1]', diffs2l='float[:, ::1]',
            values='float[4]',
            state0='float*', state1='float*', state2='float*', state3='float*',
            state0f='float', state1f='float', state2f='float',
            diffs0f='float', diffs1f='float', diffs2f='float')
    def act(self, steps):
        features = self.level['features']
        state_size = features * sizeof(float)
        free = self.params['free']
        state0l = self.params['state0l']
        diffs0l = self.params['diffs0l']
        diffs1l = self.params['diffs1l']
        diffs2l = self.params['diffs2l']
        state1 = self.state1
        state2 = self.state2
        state3 = self.state3
        action = -1

        for step in range(steps):
            values = free[:]
            state0 = c_get_state()
            for feature in range(features):
                state0f = state0[feature]
                state1f = state1[feature]
                state2f = state2[feature]
                diffs0f = state0f - state1f
                diffs1f = diffs0f - state1f + state2f
                diffs2f = state0f - 3 * (state1f - state2f) - state3[feature]
                values[0] += (state0l[0, feature] * state0f +
                                                diffs0l[0, feature] * diffs0f +
                                                diffs1l[0, feature] * diffs1f +
                                                diffs2l[0, feature] * diffs2f)
                values[1] += (state0l[1, feature] * state0f +
                                                diffs0l[1, feature] * diffs0f +
                                                diffs1l[1, feature] * diffs1f +
                                                diffs2l[1, feature] * diffs2f)
                values[2] += (state0l[2, feature] * state0f +
                                                diffs0l[2, feature] * diffs0f +
                                                diffs1l[2, feature] * diffs1f +
                                                diffs2l[2, feature] * diffs2f)
                values[3] += (state0l[3, feature] * state0f +
                                                diffs0l[3, feature] * diffs0f +
                                                diffs1l[3, feature] * diffs1f +
                                                diffs2l[3, feature] * diffs2f)
            action = (((0 if values[0] > values[3] else 3)
                                if values[0] > values[2] else
                                        (2 if values[2] > values[3] else 3))
                                if values[0] > values[1] else
                        ((1 if values[1] > values[3] else 3)
                                if values[1] > values[2] else
                                        (2 if values[2] > values[3] else 3)))
            c_do_action(action)
            state3, state2, state1 = state2, state1, state3
            memcpy(state1, state0, state_size)

        self.state1 = state1
        self.state2 = state2
        self.state3 = state3
        self.last_action = action
