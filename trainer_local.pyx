"""
A random local search.

At each step a number of random parameter entries are interpolated between the
old value and a redrawn value. If the new parameters give a better score, they
are kept, otherwise the old set is preserved.

The config should consist of an integer and a float, the number of steps and
the variation scale respectively. Moreover, --dist_variations can be used to
control the count of parameter variations to do at once at each step.
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
        self.change = float(config[1])

    @ccall
    @returns('tuple')
    @locals(dists='dict', variation_rvs='object', emphases='tuple',
            runs='int', steps='int', step='int', change='float',
            best_score='float', best_seed_score='float', score='float',
            best_bot=BaseBot, base_seed_bot=BaseBot, bot=BaseBot,
            best_history='list', history='list', variations='int')
    def train(self):
        dists = self.dists
        variations_rvs = dists['variations'].rvs
        emphases = self.emphases
        runs = self.runs
        steps = self.steps
        change = self.change
        best_score = float('-inf')
        best_bot = None
        best_history = []

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
