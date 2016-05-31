"""
Collects (state, following states) tuples, trying each action in turn.
"""
from numpy import empty

from interface cimport c_do_action, c_get_state


cdef class Collector(BaseCollector):
    cpdef dict collect(self):
        cdef:
            int features = self.level['features'], \
                actions = self.level['actions'], \
                steps = self.level['steps'], \
                step, checkpoint, action, feature
            float[:, :] states
            float[:, :, :] following_states
            float* state

        states = empty((steps, features), dtype='f4')
        following_states = empty((steps, actions, features), dtype='f4')

        for step in range(steps):
            state = c_get_state()
            for feature in range(features):
                states[step, feature] = state[feature]
            checkpoint = self.create_checkpoint()
            for action in range(actions):
                c_do_action(action)
                state = c_get_state()
                for feature in range(features):
                    following_states[step, action, feature] = state[feature]
                self.load_checkpoint(checkpoint)
            self.clear_checkpoints()
            self.bot.act(1)

        return {'states': states, 'following_states': following_states}
