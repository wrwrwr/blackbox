from numpy import array
from pytest import mark
from scipy.stats import beta, norm

from core import available_bots, available_trainers
from iop import load_level

level = load_level('train', 0)

configs = {
    'anneal': (5, 2, 1),
    'anneal_d': (5, 2, 1, .1),
    'comb': (3,),
#    'comb_phases': (3,),
    'local': (5, 1),
    'local_d': (5, 1, .1),
    'select': ()
}

dists = {
    'real': norm(0, 1),
    'unit': beta(2, 2),
    'new': {},
    'vary': {},
    'variations': beta(2, 2, loc=1, scale=9),
    'acceptance': beta(1, 1, loc=-100, scale=110)
}

emphases = (1.,) * level['features']

phases = array([.5, 1.], dtype='f4')

bot_class = available_bots['linear_m']
bot1 = bot_class(level, dists=dists, emphases=emphases, phases=phases)
bot2 = bot_class(level, dists=dists, emphases=emphases, phases=phases)
seeds = ((bot1, []), (bot2, []))


class Benchmarks:
    @mark.parametrize('trainer_key, trainer_class', available_trainers.items())
    @mark.benchmark(group='trainers')
    def benchmark_train(self, benchmark, trainer_key, trainer_class):
        config = configs[trainer_key]
        trainer = trainer_class(level=level, config=config, dists=dists,
                                emphases=emphases, seeds=seeds, runs=1)
        params = benchmark(trainer.train)[0]
        assert 'free' in params and 'state0l' in params
