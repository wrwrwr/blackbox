"""
Tries to find the best phase assignment for a number of seeds.

Provide the seeds to combine using --stored-seeds. You may give the number of
phases as the config, otherwise each seed is used in a single phase.
"""
from itertools import product

from numpy import linspace, stack

from bot_base cimport BaseBot


cdef class Trainer(TrainerComb):
    cpdef tuple train(self):
        # TODO: Passing in bots as seeds turns out not to be such a good
        # idea after all.
        Bot = type(self.seeds[0][0])
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
                    # TODO: Warn if not equal.
                    combined_params[key] = arrays[0]
            combined_params['_phases'] = self.phases

            score = Bot(self.level, combined_params).evaluate(self.runs)
            indices = [s[0] for s in seeds]
            print(indices, score)

            if score > best_score:
                best_score = score
                best_combined_params = combined_params
                best_history = [histories]

        return best_combined_params, best_history
