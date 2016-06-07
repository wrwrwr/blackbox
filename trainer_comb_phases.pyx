"""
Tries to find the best phases assignment for a number of seeds.

Provide the seeds to combine using --stored_seeds. You may give the number of
phases as the config, otherwise each seed is used in a single phase.
"""
from itertools import product

from cython import ccall, cclass, returns
from numpy import concatenate

from trainer_comb cimport Trainer as TrainerComb


@cclass
class Trainer(TrainerComb):
    @ccall
    @returns('tuple')
    def train(self):
        target_class = type(self.seeds[0][0])
        target_keys = target_class.shapes(0, 0).keys()
        best_score = float('-inf')
        best_combined_params = {}
        best_history = []

        for seeds in product(enumerate(self.seeds), repeat=len(self.phases)):
            indices, seeds = zip(*seeds)
            bots, histories = zip(*seeds)

            combined_params = {}
            for key in target_keys:
                params = [b.params[key] for b in bots]
                combined_params[key] = concatenate(params, axis=-1)
            combined_params['_phases'] = self.phases

            bot = target_class(self.level, combined_params)
            score = bot.evaluate(self.runs)
            print(indices, score)

            if score > best_score:
                best_score = score
                best_combined_params = combined_params
                best_history = [histories]

        return best_combined_params, best_history
