"""
The random local search with smaller and smaller changes.

The config should consist of three values: the number of steps (integer), the
starting variation scale (high) and the ending variation scale (low). Both of
the variation scales should be floats between 0 and 1.

Moreover, --dist_variations can be used to control the count of parameters
changed at once at each step.
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
        self.change_high = float(config[1])
        self.change_low = float(config[2])

    @ccall
    @returns('tuple')
    @locals(dists='dict', variation_rvs='object', emphases='tuple',
            runs='int', steps='int', step='int',
            change_high='float', change_diff='float', change='float',
            best_score='float', best_seed_score='float', score='float',
            best_bot=BaseBot, base_seed_bot=BaseBot, bot=BaseBot,
            best_history='list', history='list', variations='int')
    def train(self):
        dists = self.dists
        variations_rvs = dists['variations'].rvs
        emphases = self.emphases
        runs = self.runs
        steps = self.steps
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
                if score > best_seed_score:
                    best_seed_score = score
                    best_seed_bot = bot

            if best_seed_score > best_score:
                best_score = best_seed_score
                best_bot = best_seed_bot
                best_history = history

        return best_bot.params, best_history
