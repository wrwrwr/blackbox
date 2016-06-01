from collections import OrderedDict as odict

from collector_srs cimport Collector as CollectorSrs
from collector_ssa cimport Collector as CollectorSsa
from collector_sss cimport Collector as CollectorSss

available_collectors = odict((
    ('srs', CollectorSrs),
    ('ssa', CollectorSsa),
    ('sss', CollectorSss)
))
