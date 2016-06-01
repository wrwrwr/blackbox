"""
Records a level playthrough, sequences of states, scores and actions.

The n-th record in states and scores holds the values before the n-th action.
The final state and score are also stored (thus the actions array is one item
shorter than the arrays for states and scores).
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

        states = empty((steps + 1, features), dtype='f4')
        scores = empty(steps + 1, dtype='f4')
        actions = empty(steps, dtype='i4')

        for step in range(steps):
            state = c_get_state()
            for feature in range(features):
                states[step, feature] = state[feature]
            scores[step] = c_get_score()
            self.bot.act(1)
            actions[step] = self.bot.last_action

        state = c_get_state()
        for feature in range(features):
            states[steps, feature] = state[feature]
        scores[steps] = c_get_score()

        return {'states': states, 'scores': scores, 'actions': actions}
