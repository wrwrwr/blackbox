"""
The random local search with smaller and smaller changes.

The config should consist of three values: the number of steps (integer), the
starting variation scale (high) and the ending variation scale (low). Both of
the variation scales should be floats between 0 and 1.

Moreover, --dist_variations can be used to control the count of parameters
changed at once at each step.
"""
from bot_base cimport BaseBot


cdef class Trainer(BaseTrainer):
    def __cinit__(self, dict level, tuple config, dict dists, tuple emphases,
                  tuple seeds, int runs):
        self.steps = int(config[0])
        self.change_high = float(config[1])
        self.change_low = float(config[2])

    cpdef tuple train(self):
        cdef:
            dict dists = self.dists
            object variations_rvs = dists['variations'].rvs
            tuple emphases = self.emphases
            float change_high = self.change_high, \
                  change_diff = change_high - self.change_low, \
                  best_score = float('-inf'), best_seed_score, score, change
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
                change = change_high - change_diff * step / steps
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
