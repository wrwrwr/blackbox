from numpy import asarray
from pytest import mark

from core import available_bots, available_collectors, available_processors
from iop import load_level

level = load_level('train', 0)
bot = available_bots['random'](level=level)
data = {}
for key, collector in available_collectors.items():
    record = collector(level=level, bot=bot).collect()
    record = {k: asarray(v) for k, v in record.items()}
    meta = {'collector': key, 'level': level, 'bot': ('random', None)}
    data[key] = ((record, meta),)


class Benchmarks:
    @mark.parametrize('processor_key, processor_class',
                      available_processors.items())
    @mark.benchmark(group='processors')
    def benchmark_train(self, benchmark, processor_key, processor_class):
        processor = processor_class(data=data[processor_class.formats[0]])
        results = benchmark(processor.process)
        assert results
