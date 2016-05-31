"""
A trainer that simply chooses the best seed from the ones provided to it.

This may be used for a simple Monte-Carlo simulation (combined with the random
seed generation in the core), or to choose the best params from a stored set.

Does not need and config.
"""
from bot_base cimport BaseBot


cdef class Trainer(BaseTrainer):
    cpdef tuple train(self):
        cdef:
            float best_score = float('-inf'), score
            BaseBot best_bot = None, bot
            list best_history = [], history
            int runs = self.runs

        for bot, history in self.seeds:
            score = bot.evaluate(runs)
            if score > best_score:
                best_score = score
                best_bot = bot
                best_history = history

        return best_bot.params, best_history
