"""
Calculates various global statistics about state and rewards.

Requires data from the srs collector.
"""
from cython import ccall, cclass, returns
from numpy import zeros, diff, fabs, sqrt, where

from processor_base cimport BaseProcessor


@cclass
class Processor(BaseProcessor):
    formats = ('srs',)

    @ccall
    @returns('object')
    def process(self):
        rewards = zeros(self.max_actions, dtype='f4')
        totals = zeros(self.max_features, dtype='f4')
        variances = zeros(self.max_features, dtype='f4')
        changes = zeros(self.max_features, dtype='i4')
        large_changes = zeros(self.max_features, dtype='i4')
        steps = 0

        for record, meta in self.data:
            rewards += record['rewards'].sum(axis=0)
            totals += record['states'].sum(axis=0)
            variances += record['states'].var(axis=0) * meta['level']['steps']
            diffs = diff(record['states'], axis=0)
            changes += where(diffs != 0., 1, 0).sum(axis=0)
            large_changes += where(fabs(diffs) > 1., 1, 0).sum(axis=0)
            steps += meta['level']['steps']

        return self.results((
                ("average rewards", rewards / steps),
                ("state component means", totals / steps),
                ("state standard deviations", sqrt(variances / steps)),
                ("change frequencies", changes / steps),
                ("change > 1 frequencies", large_changes / steps)))
