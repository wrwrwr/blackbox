"""
The probabilistic simplified simulated annealing with diminishing changes.

The config should consist of four values: the number of steps (integer), an
acceptance distribution easing factor (float power), the starting variation
scale (float ~1), and the ending variation scale (float ~0).

Two distributions, --dist_variations and --dist_accept, are used as in the
basic anneal trainer.
"""
from bot_base cimport BaseBot


cdef class Trainer(BaseTrainer):
    def __cinit__(self, dict level, tuple config, dict dists, tuple emphases,
                  tuple seeds, int runs):
        self.steps = int(config[0])
        self.acceptance_ease = float(config[1])
        self.change_high = float(config[2])
        self.change_low = float(config[3])

    cpdef tuple train(self):
        cdef:
            dict dists = self.dists
            object variations_rvs = dists['variations'].rvs, \
                   acceptance_rvs = dists['acceptance'].rvs
            tuple emphases = self.emphases
            float acceptance_ease = self.acceptance_ease, \
                  change_high = self.change_high, \
                  change_diff = change_high - self.change_low, \
                  best_score = float('-inf'), best_seed_score, score, \
                  change, improvement, acceptance_mult
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
                improvement = score - best_seed_score
                acceptance_mult = pow(1 - float(step) / steps, acceptance_ease)
                if acceptance_rvs() * acceptance_mult < improvement:
                    best_seed_score = score
                    best_seed_bot = bot

            if best_seed_score > best_score:
                best_score = best_seed_score
                best_bot = best_seed_bot
                best_history = history

        return best_bot.params, best_history
