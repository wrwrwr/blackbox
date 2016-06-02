"""
A random local search.

At each step a number of random parameter entries are interpolated between the
old value and a redrawn value. If the new parameters give a better score, they
are kept, otherwise the old set is preserved.

The config should consist of an integer and a float, the number of steps and
the variation scale respectively. Moreover, --dist_variations can be used to
control the count of parameter variations to do at once at each step.
"""
from bot_base cimport BaseBot


cdef class Trainer(BaseTrainer):
    def __cinit__(self, dict level, tuple config, dict dists, tuple emphases,
                  tuple seeds, int runs):
        self.steps = int(config[0])
        self.change = float(config[1])

    cpdef tuple train(self):
        cdef:
            dict dists = self.dists
            object variations_rvs = dists['variations'].rvs
            tuple emphases = self.emphases
            float change = self.change, \
                  best_score = float('-inf'), best_seed_score, score
            BaseBot best_bot = None, best_seed_bot, bot
            list best_history = [], history
            int steps = self.steps, \
                runs = self.runs, \
                step, variations

        for bot, history in self.seeds:
            best_seed_score = bot.evaluate(runs)
            best_seed_bot = bot

            for step in range(steps):
                variations = max(1, round(variations_rvs()))
                bot = best_seed_bot.clone(state=False)
                bot.vary_params(dists, emphases, change, variations)
                score = bot.evaluate(runs)
                if score > best_seed_score:
                    best_seed_score = score
                    best_seed_bot = bot

            if best_seed_score > best_score:
                best_score = best_seed_score
                best_bot = best_seed_bot
                best_history = history

        return best_bot.params, best_history
