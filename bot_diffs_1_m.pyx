"""
Diffs 1 with support for multiple parameter sets.
"""
from cython import cast, ccall, cclass, declare, locals, returns, sizeof
from libc.stdlib cimport calloc, free
from libc.string cimport memcpy

from bot_base cimport BaseBot

from interface cimport c_do_action, c_get_state, c_get_time


@cclass
class Bot(BaseBot):
    def __cinit__(self, level, *args, **kwargs):
        self.param_shapes = {
            'free': (level['actions'],),
            'state0l': (level['actions'], level['features']),
            'diffs0l': (level['actions'], level['features'])
        }
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
            choices='int[:]', choice='int',
            free='float[:, :]', state0l='float[:, :, :]',
            diffs0l='float[:, :, :]',
            state0='float*', state1='float*', state0f='float', diffs0f='float')
    def act(self, steps):
        features = self.level['features']
        state_size = features * sizeof(float)
        free = self.params['free']
        state0l = self.params['state0l']
        diffs0l = self.params['diffs0l']
        choices = self.choices
        state1 = self.state1
        values = declare('float[4]')
        action = -1

        for step in range(c_get_time(), c_get_time() + steps):
            choice = choices[step]
            values[0] = free[0, choice]
            values[1] = free[1, choice]
            values[2] = free[2, choice]
            values[3] = free[3, choice]
            state0 = c_get_state()
            for feature in range(features):
                state0f = state0[feature]
                diffs0f = state0f - state1[feature]
                values[0] += (state0l[0, feature, choice] * state0f +
                                        diffs0l[0, feature, choice] * diffs0f)
                values[1] += (state0l[1, feature, choice] * state0f +
                                        diffs0l[1, feature, choice] * diffs0f)
                values[2] += (state0l[2, feature, choice] * state0f +
                                        diffs0l[2, feature, choice] * diffs0f)
                values[3] += (state0l[3, feature, choice] * state0f +
                                        diffs0l[3, feature, choice] * diffs0f)
            action = (((0 if values[0] > values[3] else 3)
                                if values[0] > values[2] else
                                        (2 if values[2] > values[3] else 3))
                                if values[0] > values[1] else
                        ((1 if values[1] > values[3] else 3)
                                if values[1] > values[2] else
                                        (2 if values[2] > values[3] else 3)))
            c_do_action(action)
            memcpy(state1, state0, state_size)

        self.last_action = action
