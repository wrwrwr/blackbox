"""
Diffs 4 with support for multiple parameter sets.

Assumes 4 actions.
"""
from cython import cast, cclass, cfunc, locals, returns, sizeof
from libc.stdlib cimport calloc, free
from libc.string cimport memcpy

from bot_base cimport BaseBot

from interface cimport c_do_action, c_get_state, c_get_time


@cclass
class Bot(BaseBot):
    def __cinit__(self, level, *args, **kwargs):
        self.param_shapes = {
            'constant': (level['actions'],),
            'state0l': (level['actions'], level['features']),
            'diffs0l': (level['actions'], level['features']),
            'diffs1l': (level['actions'], level['features']),
            'diffs2l': (level['actions'], level['features']),
            'diffs3l': (level['actions'], level['features'])
        }
        self.state1 = cast('float*', calloc(level['features'], sizeof(float)))
        self.state2 = cast('float*', calloc(level['features'], sizeof(float)))
        self.state3 = cast('float*', calloc(level['features'], sizeof(float)))
        self.state4 = cast('float*', calloc(level['features'], sizeof(float)))

    def __dealloc__(self):
        free(self.state4)
        free(self.state3)
        free(self.state2)
        free(self.state1)

    @cfunc
    @returns('Bot')
    @locals(state='bint', bot='Bot', state_size='int')
    def clone(self, state=True):
        bot = BaseBot.clone(self, state)
        if state:
            state_size = self.level['features'] * sizeof(float)
            memcpy(bot.state1, self.state1, state_size)
            memcpy(bot.state2, self.state2, state_size)
            memcpy(bot.state3, self.state3, state_size)
            memcpy(bot.state4, self.state4, state_size)
        return bot

    @cfunc
    @returns('void')
    @cfunc
    @returns('void')
    @locals(steps='int', step='int', action='int', time='int',
            features='int', feature='int', state_size='int',
            choices='int[:]', choice='int',
            constant0='float[:]', constant1='float[:]',
            constant2='float[:]', constant3='float[:]',
            state0l0='float[:, :]', state0l1='float[:, :]',
            state0l2='float[:, :]', state0l3='float[:, :]',
            diffs0l0='float[:, :]', diffs0l1='float[:, :]',
            diffs0l2='float[:, :]', diffs0l3='float[:, :]',
            diffs1l0='float[:, :]', diffs1l1='float[:, :]',
            diffs1l2='float[:, :]', diffs1l3='float[:, :]',
            diffs2l0='float[:, :]', diffs2l1='float[:, :]',
            diffs2l2='float[:, :]', diffs2l3='float[:, :]',
            diffs3l0='float[:, :]', diffs3l1='float[:, :]',
            diffs3l2='float[:, :]', diffs3l3='float[:, :]',
            value0='float', value1='float', value2='float', value3='float',
            state0='float*', state1='float*', state2='float*',
            state3='float*', state4='float*',
            state0f='float', state1f='float', state2f='float', state3f='float',
            diffs0f='float', diffs1f='float', diffs2f='float', diffs3f='float')
    def act(self, steps):
        features = self.level['features']
        state_size = features * sizeof(float)
        constant0, constant1, constant2, constant3 = self.params['constant']
        state0l0, state0l1, state0l2, state0l3 = self.params['state0l']
        diffs0l0, diffs0l1, diffs0l2, diffs0l3 = self.params['diffs0l']
        diffs1l0, diffs1l1, diffs1l2, diffs1l3 = self.params['diffs1l']
        diffs2l0, diffs2l1, diffs2l2, diffs2l3 = self.params['diffs2l']
        diffs3l0, diffs3l1, diffs3l2, diffs3l3 = self.params['diffs3l']
        state1, state2, state3, state4 = (self.state1, self.state2,
                                                    self.state3, self.state4)
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
                state1f = state1[feature]
                state2f = state2[feature]
                state3f = state3[feature]
                diffs0f = state0f - state1f
                diffs1f = diffs0f - state1f + state2f
                diffs2f = state0f - 3 * (state1f - state2f) - state3f
                diffs3f = (state0f - 4 * (state1f + state3f) +
                                                6 * state2f + state4[feature])
                value0 += (state0l0[feature, choice] * state0f +
                                        diffs0l0[feature, choice] * diffs0f +
                                        diffs1l0[feature, choice] * diffs1f +
                                        diffs2l0[feature, choice] * diffs2f +
                                        diffs3l0[feature, choice] * diffs3f)
                value1 += (state0l1[feature, choice] * state0f +
                                        diffs0l1[feature, choice] * diffs0f +
                                        diffs1l1[feature, choice] * diffs1f +
                                        diffs2l1[feature, choice] * diffs2f +
                                        diffs3l1[feature, choice] * diffs3f)
                value2 += (state0l2[feature, choice] * state0f +
                                        diffs0l2[feature, choice] * diffs0f +
                                        diffs1l2[feature, choice] * diffs1f +
                                        diffs2l2[feature, choice] * diffs2f +
                                        diffs3l2[feature, choice] * diffs3f)
                value3 += (state0l3[feature, choice] * state0f +
                                        diffs0l3[feature, choice] * diffs0f +
                                        diffs1l3[feature, choice] * diffs1f +
                                        diffs2l3[feature, choice] * diffs2f +
                                        diffs3l3[feature, choice] * diffs3f)
            action = (((0 if value0 > value3 else 3)
                                    if value0 > value2 else
                                                (2 if value2 > value3 else 3))
                                if value0 > value1 else
                        ((1 if value1 > value3 else 3)
                                    if value1 > value2 else
                                                (2 if value2 > value3 else 3)))
            c_do_action(action)
            state4, state3, state2, state1 = state3, state2, state1, state4
            memcpy(state1, state0, state_size)
            time += 1

        self.state1, self.state2, self.state3, self.state4 = (state1, state2,
                                                                state3, state4)
        self.last_action = action
