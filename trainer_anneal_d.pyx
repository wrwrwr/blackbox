"""
The probabilistic, simplified simulated annealing with diminishing variations.

The config should consist of four values: the number of steps (integer), an
acceptance distribution easing factor (float power), the starting variation
scale (float ~1), and the ending variation scale (float ~0).

Two distributions, --dist_variations and --dist_acceptance, are used as in the
basic anneal trainer.
"""
from cython import ccall, cclass, locals, returns

from bot_base cimport BaseBot
from trainer_base cimport BaseTrainer


@cclass
class Trainer(BaseTrainer):
    @locals(level='dict', config='tuple', dists='dict', emphases='tuple',
            seeds='tuple', runs='int')
    def __cinit__(self, level, config, dists, emphases, seeds, runs):
        self.steps = int(config[0])
        self.acceptance_ease = float(config[1])
        self.change_high = float(config[2])
        self.change_low = float(config[3])

    @ccall
    @returns('tuple')
    @locals(dists='dict', variation_rvs='object', acceptance_rvs='object',
            emphases='tuple', runs='int', steps='int', step='int',
            acceptance_ease='float',
            change_high='float', change_diff='float', change='float',
            best_score='float', best_seed_score='float', score='float',
            best_bot=BaseBot, base_seed_bot=BaseBot, bot=BaseBot,
            best_history='list', history='list',
            variations='int', improvement='float', acceptance_mult='float')
    def train(self):
        dists = self.dists
        variations_rvs = dists['variations'].rvs
        acceptance_rvs = dists['acceptance'].rvs
        emphases = self.emphases
        runs = self.runs
        steps = self.steps
        acceptance_ease = self.acceptance_ease
        change_high = self.change_high
        change_diff = change_high - self.change_low
        best_score = float('-inf')
        best_bot = None
        best_history = []

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
