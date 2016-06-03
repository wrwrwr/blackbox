"""
Collects (state, following states) tuples, trying each action at every step of
the level run.
"""
from cython import ccall, cclass, locals, returns
from numpy import empty

from collector_base import BaseCollector

from interface cimport c_do_action, c_get_state


@cclass
class Collector(BaseCollector):
    @ccall
    @returns('dict')
    @locals(steps='int', step='int', actions='int', action='int',
            features='int', feature='int', state='float*', checkpoint='int',
            states='float[:, :]', following_states='float[:, :, :]')
    def collect(self):
        steps = self.level['steps']
        actions = self.level['actions']
        features = self.level['features']
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
