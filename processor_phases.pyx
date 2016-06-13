"""
Tries to determine if a level is composed of multiple phases or sub-episodes.

For each feature we find the steps at which its initial value first reoccurs.
If the repetitions are common to multiple playthroughs, and spaced more or
less evenly throughout the level, the component may mark phase starts.

for meaningful results you should pass at least a couple of inputs.
"""
from cython import ccall, cclass, locals, returns
from numpy import allclose, ones, diff, linspace, nonzero, split, transpose

from processor_base cimport BaseProcessor


@cclass
class Processor(BaseProcessor):
    formats = ('srs', 'ssa', 'sss')

    # Limit the number of phases in case a huge number of steps qualifies.
    max_phases = 100

    # What does it mean to be spaced more or less evenly.
    rtol = .3

    @ccall
    @returns('object')
    @locals(possible='bint[:, ::1]', states='float[:, ::1]',
            step='int', fstep='int', feature='int')
    def process(self):
        if any([l != self.levels[0] for l in self.levels]):
            raise ValueError("Data must be collected on a single level")

        possible = ones((self.max_steps, self.max_features), dtype='intc')

        for record in self.records:
            states = record['states']
            for step in range(self.max_steps):
                for feature in range(self.max_features):
                    if states[step, feature] != states[0, feature]:
                        possible[step, feature] = False

        for feature in range(self.max_features):
            for step in range(self.max_steps):
                if possible[step, feature]:
                    fstep = step + 1
                    while fstep < self.max_steps and possible[fstep, feature]:
                        possible[fstep, feature] = False
                        fstep += 1

        starts = nonzero(transpose(possible))
        feature_splits = nonzero(diff(starts[0]))[0] + 1
        phases = split(starts[1], feature_splits)
        for index, starts in enumerate(phases):
            even = linspace(0, 1, len(starts), endpoint=False)
            if len(starts) == 1:
                info = "no reoccurences"
            elif len(starts) > self.max_phases:
                info = "too many reoccurences"
            elif not allclose(starts / self.max_steps, even, rtol=self.rtol):
                info = "not evenly distributed"
            else:
                info = ", ".join(map(str, starts))
            phases[index] = "{}: {}".format(index, info)
        phases = "\n".join(phases)

        return self.results((
                ("possible subepisodes (feature resets)", phases),))
