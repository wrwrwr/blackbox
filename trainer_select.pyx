"""
A trainer that simply chooses the best seed from the ones given to it.

This may be used for a simple Monte-Carlo simulation (combined with the random
seed generation in the core), or to choose the best parameters set from a
collection of stored sets.

Does not need and config.
"""
from cython import ccall, cclass, returns

from trainer_base cimport BaseTrainer


@cclass
class Trainer(BaseTrainer):
    @ccall
    @returns('tuple')
    def train(self):
        best_score = float('-inf')
        best_bot = None
        best_history = []

        for bot, history in self.seeds:
            score = bot.evaluate(self.runs)
            if score > best_score:
                best_score = score
                best_bot = bot
                best_history = history

        return best_bot.params, best_history
