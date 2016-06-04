"""
Calculates rewards per action, phase, and step congruency class. Can give a
first glance at possible temporal patterns.

Processes data from the ssa processor.
"""
from cython import ccall, cclass, returns
from numpy import diff, zeros

from processor_base cimport BaseProcessor


@cclass
class Processor(BaseProcessor):
    @ccall
    @returns('object')
    def process(self):
        bots = []
        levels = []
        action_counts = zeros(self.max_actions, dtype='i4')
        action_rewards = zeros(self.max_actions, dtype='f4')
        phase5_rewards = zeros(5, dtype='f4')
        phase8_rewards = zeros(8, dtype='f4')
        phase9_rewards = zeros(9, dtype='f4')
        mod5_rewards = zeros(5, dtype='f4')
        mod8_rewards = zeros(8, dtype='f4')
        mod9_rewards = zeros(9, dtype='f4')

        for record, meta in self.data:
            steps = meta['level']['steps']
            bots.append(meta['bot'])
            levels.append(meta['level']['key'])
            rewards = diff(record['scores'])
            for step, action in enumerate(record['actions']):
                reward = rewards[step]
                action_counts[action] += 1
                action_rewards[action] += reward
                phase5_rewards[int(5 * step / steps)] += reward
                phase8_rewards[int(8 * step / steps)] += reward
                phase9_rewards[int(9 * step / steps)] += reward
                mod5_rewards[step % 5] += reward
                mod8_rewards[step % 8] += reward
                mod9_rewards[step % 9] += reward

        return self.results((
                ("counts of actions taken", action_counts),
                ("average reward per action", action_rewards / action_counts),
                ("gains by 5 level phases", phase5_rewards / len(self.data)),
                ("gains by 8 level phases", phase8_rewards / len(self.data)),
                ("gains by 9 level phases", phase9_rewards / len(self.data)),
                ("gains by step mod 5", mod5_rewards / len(self.data)),
                ("gains by step mod 8", mod8_rewards / len(self.data)),
                ("gains by step mod 9", mod9_rewards / len(self.data))))
