from numpy import array
from pytest import mark
from scipy.stats import beta, norm

from core import available_bots
from iop import load_level

from interface import finish

level = load_level('train', 0)

dists = {
    'real': norm(0, 1),
    'unit': beta(2, 2),
    'new': {},
    'vary': {}
}

emphases = (1.,) * level['features']

no_phases = array([1.], dtype='f4')
phases = array([.33, .67, 1.], dtype='f4')


class Benchmarks:
    @mark.parametrize('bot_key, bot_class', available_bots.items())
    @mark.benchmark(group='bots')
    def benchmark_evaluate(self, benchmark, bot_key, bot_class):
        bot = bot_class(level, dists=dists, emphases=emphases,
                        phases=phases if bot_key.endswith('_m') else no_phases)
        benchmark(bot.evaluate, 1)
        finish(0)
