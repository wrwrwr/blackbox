"""
Tries to find the best phases assignment for a number of seeds.

Provide the seeds to combine using --stored_seeds. You may give the number of
phases as the config, otherwise each seed is used in a single phase.
"""
from itertools import product
from warnings import warn

from cython import ccall, cclass, returns
from numpy import allclose, stack

from trainer_comb cimport Trainer as TrainerComb


@cclass
class Trainer(TrainerComb):
    @ccall
    @returns('tuple')
    def train(self):
        bot_class = type(self.seeds[0][0])
        best_score = float('-inf')
        best_combined_params = {}
        best_history = []

        for seeds in product(enumerate(self.seeds), repeat=len(self.phases)):
            combined_params = {}
            histories = []
            for index, (bot, history) in seeds:
                for key, param in bot.params.items():
                    combined_params.setdefault(key, []).append(param)
                histories.append(history)

            for key, arrays in combined_params.items():
                if key[0] != '_':
                    combined_params[key] = stack(arrays, axis=-1)
                else:
                    for array in arrays:
                        if not allclose(array, arrays[0]):
                            warn("Combining with differing meta-parameters")
                    combined_params[key] = arrays[0]
            combined_params['_phases'] = self.phases

            score = bot_class(self.level, combined_params).evaluate(self.runs)
            indices = [s[0] for s in seeds]
            print(indices, score)

            if score > best_score:
                best_score = score
                best_combined_params = combined_params
                best_history = [histories]

        return best_combined_params, best_history
