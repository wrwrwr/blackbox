"""
Records a level playthrough, a sequence of (state, action and score) tuples.
(It shouldn't be too risky to assume that the initial score is always zero.)
"""
from numpy import empty

from interface cimport c_get_score, c_get_state


cdef class Collector(BaseCollector):
    cpdef dict collect(self):
        cdef:
            int features = self.level['features'], \
                steps = self.level['steps'], \
                step, action, feature
            float[:, :] states
            int[:] actions
            float[:] scores
            float* state

        states = empty((steps, features), dtype='f4')
        actions = empty(steps, dtype='i4')
        scores = empty(steps, dtype='f4')

        for step in range(steps):
            state = c_get_state()
            for feature in range(features):
                states[step, feature] = state[feature]
            self.bot.act(1)
            actions[step] = self.bot.last_action
            scores[step] = c_get_score()

        return {'states': states, 'actions': actions, 'scores': scores}
