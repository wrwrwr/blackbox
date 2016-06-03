"""
Collects (state, immediate rewards) tuples, trying each action at every step
of the level playthrough.
"""
from cython import ccall, cclass, locals, returns
from numpy import empty

from collector_base import BaseCollector

from interface cimport c_do_action, c_get_score, c_get_state


@cclass
class Collector(BaseCollector):
    @ccall
    @returns('dict')
    @locals(steps='int', step='int', actions='int', action='int',
            features='int', feature='int', state='float*', checkpoint='int',
            states='float[:, :]', rewards='float[:, :]', score='float')
    def collect(self):
        steps = self.level['steps']
        actions = self.level['actions']
        features = self.level['features']
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
