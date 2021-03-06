"""
A bot that remembers some previously seen states, actions and rewards and
decides by presuming that states similar to the past ones will give similar
rewards for the same action.

Assumes 4 actions.
"""
from cython import (address, cast, ccall, cclass, declare, locals, returns,
                    sizeof)
from libc.math cimport sqrt
from libc.stdlib cimport free, malloc, rand
from libc.string cimport memcpy

from bot_base cimport BaseBot

from interface cimport c_do_action, c_get_score, c_get_state, c_get_time


@cclass
class Bot(BaseBot):
    @staticmethod
    def shapes(steps, actions, features):
        return {}

    def __cinit__(self, level, *args, **kwargs):
        steps = level['steps']
        features = level['features']
        self.states = cast('float*', malloc(steps * features * sizeof(float)))
        self.scores = cast('float*', malloc(steps * sizeof(float)))
        self.actions = cast('int*', malloc(steps * sizeof(int)))

    def __dealloc__(self):
        free(self.states)
        free(self.scores)
        free(self.actions)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params['lookback'] = 100
        self.params['threshold'] = .01

    @ccall
    @returns('Bot')
    @locals(state='bint', bot='Bot')
    def clone(self, state=True):
        bot = BaseBot.clone(self, state)
        if state:
            memcpy(bot.states, self.states, self.level['steps'] *
                                        self.level['features'] * sizeof(float))
            memcpy(bot.scores, self.scores, self.level['steps'] *
                                                                sizeof(float))
            memcpy(bot.actions, self.actions, self.level['steps'] *
                                                                sizeof(int))
        return bot

    @ccall
    @returns('void')
    @locals(dists='dict', emphases='tuple', change='float')
    def vary_param(self, dists, emphases, change):
        raise NotImplementedError()

    @ccall
    @returns('void')
    @locals(steps='int', step='int', pstep='int', action='int',
            features='int', feature='int', state_size='int',
            lookback='int', threshold='float',
            states='float*', scores='float*', actions='int*',
            score='float', dissimi='float', diff='float',
            state='float*', pstate='float*')
    def act(self, steps):
        features = self.level['features']
        state_size = features * sizeof(float)
        lookback = self.params['lookback']
        threshold = self.params['threshold']
        states = self.states
        scores = self.scores
        actions = self.actions
        values = declare('float[4]')
        action = -1

        for step in range(c_get_time(), c_get_time() + steps):
            state = c_get_state()
            memcpy(address(states[step * features]), state, state_size)
            score = c_get_score()
            scores[step] = c_get_score()
            values[0] = values[1] = values[2] = values[3] = 0
            for pstep in range(max(0, step - lookback), step):
                dissimi = 0
                pstate = address(states[pstep * features])
                for feature in range(features):
                    diff = state[feature] - pstate[feature]
                    dissimi += diff * diff
                dissimi = sqrt(dissimi)
                values[actions[pstep]] += (score - scores[pstep]) / (
                                                    (step - pstep) * dissimi)
            action = (((0 if values[0] > values[3] else 3)
                                if values[0] > values[2] else
                                        (2 if values[2] > values[3] else 3))
                                if values[0] > values[1] else
                        ((1 if values[1] > values[3] else 3)
                                if values[1] > values[2] else
                                        (2 if values[2] > values[3] else 3)))
            if values[action] < threshold:
                action = rand() % 4
            c_do_action(action)
            actions[step] = action

        self.last_action = action
