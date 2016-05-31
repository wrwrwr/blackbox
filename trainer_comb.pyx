"""
Combines stored seeds into a multi-parameter set finding the phase assignment.

Takes the number of phases as the config.
"""
from numpy import linspace, stack

from bot_base cimport BaseBot


cdef class Trainer(BaseTrainer):
    def __cinit__(self, dict level, tuple config, dict dists, tuple emphases,
                  tuple seeds, int runs):
        phase_count = int(config[0]) if config else len(seeds)
        self.phases = linspace(1 / phase_count, 1, num=phase_count, dtype='f4')

    cpdef tuple train(self):
        combined_params = {}
        histories = []

        for bot, history in self.seeds:
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

        return combined_params, [histories]
