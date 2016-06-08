"""
A randomized local search.

At each step a number of random parameter entries are interpolated between the
old value and a redrawn value. If the new parameters give a better score, they
are kept, otherwise the old set is preserved.

The config should consist of an integer and a float, the number of steps and
the variation scale respectively. Moreover, --dist_variations can be used to
control the count of parameter variations to try at once at each step.
"""
from cython import ccall, cclass, returns

from trainer_base cimport BaseTrainer


@cclass
class Trainer(BaseTrainer):
    arguments = (
        ('steps', int),
        ('change', float)
    )

    @ccall
    @returns('tuple')
    def train(self):
        variations_dist = self.dists['variations']
        best_score = float('-inf')
        best_bot = None
        best_history = []

        for bot, history in self.seeds:
            best_seed_score = bot.evaluate(self.runs)
            best_seed_bot = bot

            for step in range(self.steps):
                bot = best_seed_bot.clone(state=False)
                change = self.change
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
