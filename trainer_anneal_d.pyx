"""
The probabilistic, simplified simulated annealing with diminishing variations.

The config should consist of four values: the number of steps (integer), an
acceptance distribution easing factor (float power), the starting variation
scale (float ~1), and the ending variation scale (float ~0).

Two distributions, --dist_variations and --dist_acceptance, are used as in the
basic anneal trainer.
"""
from cython import ccall, cclass, returns

from trainer_local_d cimport Trainer as TrainerLocalD


@cclass
class Trainer(TrainerLocalD):
    arguments = (
        ('steps', int),
        ('acceptance_ease', float),
        ('change_high', float),
        ('change_low', float)
    )

    @ccall
    @returns('tuple')
    def train(self):
        variations_dist = self.dists['variations']
        acceptance_dist = self.dists['acceptance']
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
                improvement = score - best_seed_score
                mult = pow(1 - float(step) / self.steps, self.acceptance_ease)
                if acceptance_dist.rvs() * mult < improvement:
                    best_seed_score = score
                    best_seed_bot = bot

            if best_seed_score > best_score:
                best_score = best_seed_score
                best_bot = best_seed_bot
                best_history = history

        return best_bot.params, best_history
