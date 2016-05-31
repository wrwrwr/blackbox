from collections import OrderedDict as odict

from processor_corrs cimport Processor as ProcessorCorrs
from processor_init cimport Processor as ProcessorInit
from processor_pt cimport Processor as ProcessorPt
from processor_stats cimport Processor as ProcessorStats

available_processors = odict((
    ('corrs', ProcessorCorrs),
    ('init', ProcessorInit),
    ('pt', ProcessorPt),
    ('stats', ProcessorStats)
))
