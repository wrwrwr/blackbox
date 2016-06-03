"""
A "probabilistic" simplified simulated annealing.

At each step a number of parameters are redrawn and the bot is evaluated with
the new parameters. Then a random threshold is drawn and scaled by a power of
the fraction of the level left. The resulting value is compared with the
returns change, if the growth or decline of the total score is greater than
the scaled threshold the new params are kept, otherwise the search continues
with the previous set.

The config should consist of three values: the number of steps (integer),
an easing factor for the effects of the acceptance distribution (float),
and a variation scale (float).

The thresholds drawn from the acceptance distribution are multpilied by the
fraction of the level left raised to the power of the easing factor, thus a
factor of 0 lets declines be accepted (with some probability) from the start
till the end of the training session, while a high factor only allows declines
to be accepted at the beginning of it.

If the variation scale is less than 1, new param entries are interpolated
between newly drawn values and their old values (rather than being redrawn
anew).

Two probability distributions are used: --dist_variations controls the number
of parameters changed at once at each step, and --dist_accept decides whether
to continue with a mutated bot or keep the previous one.
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
        self.change = float(config[2])

    @ccall
    @returns('tuple')
    @locals(dists='dict', variation_rvs='object', acceptance_rvs='object',
            emphases='tuple', runs='int', steps='int', step='int',
            acceptance_ease='float', change='float',
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
