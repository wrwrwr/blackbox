from collections import OrderedDict as odict

from collector_sas cimport Collector as CollectorSas
from collector_srs cimport Collector as CollectorSrs
from collector_sss cimport Collector as CollectorSss

available_collectors = odict((
    ('sas', CollectorSas),
    ('srs', CollectorSrs),
    ('sss', CollectorSss)
))
