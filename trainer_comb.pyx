"""
Combines stored seeds into a multi-parameter set.

Takes the number of phases as the config, splitting the level into the given
number of equal parts (if the phases count does not match the seeds count, the
parameter values will be repeated by the base bot).
"""
from warnings import warn

from cython import ccall, cclass, locals, returns
from numpy import allclose, linspace, stack

from bot_base cimport BaseBot
from trainer_base cimport BaseTrainer


@cclass
class Trainer(BaseTrainer):
    @locals(level='dict', config='tuple', dists='dict', emphases='tuple',
            seeds='tuple', runs='int', phase_count='int')
    def __cinit__(self, level, config, dists, emphases, seeds, runs):
        phase_count = int(config[0]) if config else len(seeds)
        self.phases = linspace(1 / phase_count, 1, num=phase_count, dtype='f4')

    @ccall
    @returns('tuple')
    @locals(combined_params='dict', histories='list',
            bot=BaseBot, history='list', key='str', param='object',
            arrays='list')
    def train(self):
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
                for array in arrays:
                    if not allclose(array, arrays[0]):
                        warn("Combining params with differing meta-parameters")
                combined_params[key] = arrays[0]
        combined_params['_phases'] = self.phases

        return combined_params, [histories]
