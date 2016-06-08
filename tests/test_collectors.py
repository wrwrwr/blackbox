from pytest import mark

from core import available_bots, available_collectors
from iop import load_level

level = load_level('train', 0)
bot = available_bots['random'](level=level)


class Benchmarks:
    @mark.parametrize('collector_key, collector_class',
                      available_collectors.items())
    @mark.benchmark(group='collectors')
    def benchmark_train(self, benchmark, collector_key, collector_class):
        collector = collector_class(level=level, bot=bot)
        data = benchmark(collector.collect)
        assert data
