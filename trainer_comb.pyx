"""
Combines stored seeds into a multi-parameter set.

Takes the number of phases as the config, splitting the level into the given
number of equal parts (if the phases count does not match the seeds count, the
parameter values will be repeated by the base bot).
"""
from cython import ccall, cclass, returns
from numpy import concatenate, linspace

from trainer_base cimport BaseTrainer


@cclass
class Trainer(BaseTrainer):
    def __init__(self, level, config, *args, **kwargs):
        super().__init__(level, (), *args, **kwargs)
        phase_count = int(config[0]) if config else len(self.seeds)
        self.phases = linspace(1 / phase_count, 1, num=phase_count, dtype='f4')

    @ccall
    @returns('tuple')
    def train(self):
        bots, histories = zip(*self.seeds)
        target_keys = bots[0].shapes(0, 0, 0).keys()

        combined_params = {}
        for key in target_keys:
            params = [b.params[key] for b in bots]
            combined_params[key] = concatenate(params, axis=-1)
        combined_params['_phases'] = self.phases

        return combined_params, [histories]
