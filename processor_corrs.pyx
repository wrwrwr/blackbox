"""
How do state components relate to each other?
"""
from numpy import corrcoef, nonzero, transpose, triu, zeros


cdef class Processor(BaseProcessor):
    cpdef object process(self):
        corrs = zeros((self.max_features, self.max_features), dtype='f4')
        steps = 0

        for record, meta in self.data:
            corrs += corrcoef(record['states'], rowvar=0)
            steps += meta['level']['steps']

        # It would probably be better to calculate the correlation over the
        # combined data from all iterations / files.
        corrs = triu(corrs, 1) / len(self.data)
        corrs_large = list(nonzero(corrs > .5))
        corrs_large.append(corrs[corrs_large])
        corrs_large = transpose(corrs_large)
        corrs_small = list(nonzero(corrs < -.5))
        corrs_small.append(corrs[corrs_small])
        corrs_small = transpose(corrs_small)

        return self.results((
                ('state component pairs with large correlation', corrs_large),
                ('state component pairs with small correlation', corrs_small)))
