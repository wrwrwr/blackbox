"""
The randomized local search with smaller and smaller changes.

The config should consist of three values: the number of steps (integer), the
starting variation scale (high) and the ending variation scale (low). Both of
the variation scales should be floats between 0 and 1.

Moreover, --dist_variations can be used to control the count of parameters
changed at once at each step.
"""
from cython import ccall, cclass, returns

from trainer_base cimport BaseTrainer


@cclass
class Trainer(BaseTrainer):
    arguments = (
        ('steps', int),
        ('change_high', float),
        ('change_low', float)
    )

    @ccall
    @returns('tuple')
    def train(self):
        variations_dist = self.dists['variations']
        change_diff = (self.change_high - self.change_low) / self.steps
        best_score = float('-inf')
        best_bot = None
        best_history = []

        for bot, history in self.seeds:
            best_seed_score = bot.evaluate(self.runs)
            best_seed_bot = bot

            for step in range(self.steps):
                bot = best_seed_bot.clone(state=False)
                change = self.change_high - change_diff * step
                variations = max(1, round(variations_dist.rvs()))
                bot.vary_params(self.dists, self.emphases, change, variations)
                score = bot.evaluate(self.runs)
                if score > best_seed_score:
                    best_seed_score = score
                    best_seed_bot = bot

            if best_seed_score > best_score:
                best_score = best_seed_score
                best_bot = best_seed_bot
                best_history = history

        return best_bot.params, best_history
