"""
Collects (state, immediate rewards) tuples, trying each action at every step
of the level playthrough.
"""
from numpy import empty

from interface cimport c_do_action, c_get_score, c_get_state


cdef class Collector(BaseCollector):
    cpdef dict collect(self):
        cdef:
            int features = self.level['features'], \
                actions = self.level['actions'], \
                steps = self.level['steps'], \
                step, checkpoint, action, feature
            float[:, :] states, rewards
            float* state
            float score

        states = empty((steps, features), dtype='f4')
        rewards = empty((steps, actions), dtype='f4')

        for step in range(steps):
            state = c_get_state()
            for feature in range(features):
                states[step, feature] = state[feature]
            score = c_get_score()
            checkpoint = self.create_checkpoint()
            for action in range(actions):
                c_do_action(action)
                rewards[step, action] = c_get_score() - score
                self.load_checkpoint(checkpoint)
            self.clear_checkpoints()
            self.bot.act(1)

        return {'states': states, 'rewards': rewards}
