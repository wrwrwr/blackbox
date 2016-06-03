"""
Shows information about the first few states.

Needs data from the sss collector.
"""
from cython import ccall, cclass, returns
from numpy import empty, var

from processor_base cimport BaseProcessor


@cclass
class Processor(BaseProcessor):
    @ccall
    @returns('object')
    def process(self):
        records_count = len(self.data)
        features = self.max_features
        actions = self.max_actions
        states0 = empty((records_count, features), dtype='f4')
        states3 = empty((records_count, features), dtype='f4')
        fstates1 = empty((records_count, actions, features), dtype='f4')
        fstates4 = empty((records_count, actions, features), dtype='f4')
        changes1 = empty((records_count, actions, features), dtype='f4')
        changes4 = empty((records_count, actions, features), dtype='f4')
        variances1 = empty((records_count, features), dtype='f4')
        variances4 = empty((records_count, features), dtype='f4')

        for index, (record, meta) in enumerate(self.data):
            states0[index] = record['states'][0]
            states3[index] = record['states'][3]
            fstates1[index] = record['following_states'][0]
            fstates4[index] = record['following_states'][3]
            changes1[index] = fstates1[index] - states0[index]
            changes4[index] = fstates4[index] - states3[index]
            variances1[index] = var(fstates1[index], axis=0)
            variances4[index] = var(fstates4[index], axis=0)

        return self.results((
                ("initial states", states0),
                # ("states after the first action", fstates1),
                ("changes after the first action", changes1),
                ("the first action matters", variances1.any()),
                ("state after three actions", states3),
                # ("following (fourth) states", fstates4),
                ("changes after the fourth action", changes4),
                ("the fourth action matters", variances4.any())))
