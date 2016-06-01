"""
A bot that remembers some previously seen states, actions taken and rewards and
decides based on the similarity of the current state to each of the previous.

Assumes 4 actions.
"""
from libc.math cimport sqrt
from libc.stdlib cimport free, malloc, rand
from libc.string cimport memcpy

from interface cimport c_do_action, c_get_score, c_get_state


cdef class Bot(BaseBot):
    def __cinit__(self, level, *args, **kwargs):
        self.param_shapes = {}
        self.states = <float*>malloc(
                            level['steps'] * level['features'] * sizeof(float))
        self.scores = <float*>malloc(level['steps'] * sizeof(float))
        self.actions = <int*>malloc(level['steps'] * sizeof(int))

    def __dealloc__(self):
        free(self.states)
        free(self.scores)
        free(self.actions)

    cdef Bot clone(self, bint state=True):
        cdef:
            Bot bot = BaseBot.clone(self, state)
            int features = self.level['features'], steps = self.level['steps']

        if state:
            memcpy(bot.states, self.states, steps * features * sizeof(float))
            memcpy(bot.scores, self.scores, steps * sizeof(float))
            memcpy(bot.actions, self.actions, steps * sizeof(int))
        return bot

    cdef dict new_params(self, dict dists, tuple emphases):
        return {'lookback': 100, 'threshold': 0}

    cdef void vary_param(self, dict dists, tuple emphases, float change):
        raise NotImplementedError()

    cdef void act(self, int steps):
        cdef:
            int features = self.level['features'], \
                lookback = self.params['lookback'], \
                state_size = features * sizeof(float), \
                step, pstep, feature, action = -1
            float threshold = self.params['threshold'], \
                  score, dissimi, diff
            float* states = self.states
            float* scores = self.scores
            int* actions = self.actions
            float[4] values
            float* state
            float* pstate

        score = c_get_score()

        for step in range(steps):
            state = c_get_state()
            values[0] = values[1] = values[2] = values[3] = 0
            for pstep in range(max(0, step - lookback), step - 1):
                dissimi = 0
                pstate = &states[pstep * features]
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
            memcpy(&states[step * features], state, state_size)
            actions[step] = action
            scores[step] = score

        self.last_action = action
