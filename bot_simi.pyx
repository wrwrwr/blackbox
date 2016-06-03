"""
A bot that remembers some previously seen states, actions taken and rewards and
decides based on the similarity of the current state to each of the previous.

Assumes 4 actions.
"""
from cython import (address, cast, cclass, cfunc, declare, locals, returns,
                    sizeof)
from libc.math cimport sqrt
from libc.stdlib cimport free, malloc, rand
from libc.string cimport memcpy

from bot_base cimport BaseBot

from interface cimport c_do_action, c_get_score, c_get_state


@cclass
class Bot(BaseBot):
    def __cinit__(self, level, *args, **kwargs):
        self.param_shapes = {}
        self.states = cast('float*', malloc(level['steps'] *
                                            level['features'] * sizeof(float)))
        self.scores = cast('float*', malloc(level['steps'] * sizeof(float)))
        self.actions = cast('int*', malloc(level['steps'] * sizeof(int)))

    def __dealloc__(self):
        free(self.states)
        free(self.scores)
        free(self.actions)

    @cfunc
    @returns('Bot')
    @locals(state='bint', bot='Bot', features='int', steps='int')
    def clone(self, state=True):
        bot = BaseBot.clone(self, state)
        if state:
            features = self.level['features']
            steps = self.level['steps']
            memcpy(bot.states, self.states, steps * features * sizeof(float))
            memcpy(bot.scores, self.scores, steps * sizeof(float))
            memcpy(bot.actions, self.actions, steps * sizeof(int))
        return bot

    @cfunc
    @returns('dict')
    @locals(dists='dict', emphases='tuple')
    def new_params(self, dists, emphases):
        return {'lookback': 100, 'threshold': 0}

    @cfunc
    @returns('void')
    @locals(dists='dict', emphases='tuple', change='float')
    def vary_param(self, dists, emphases, change):
        raise NotImplementedError()

    @cfunc
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
        score = c_get_score()
        values = declare('float[4]')
        action = -1

        for step in range(steps):
            state = c_get_state()
            values[0] = values[1] = values[2] = values[3] = 0
            for pstep in range(max(0, step - lookback), step - 1):
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
            score = c_get_score()
            memcpy(address(states[step * features]), state, state_size)
            actions[step] = action
            scores[step] = score

        self.last_action = action
