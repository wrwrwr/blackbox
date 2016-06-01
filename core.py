"""
Proxies between command-line utilities and statically compiled implementations.
"""
from collections import OrderedDict as odict
from datetime import datetime
from heapq import heappop, heappush
from time import clock

import cython_init  # Executes pyximport install on module import # noqa.
from bots import available_bots
from collectors import available_collectors
from iop import (common_printoptions, first_free, load_data, load_level,
                 load_params, save_data, save_params)
from prngs import seed_prngs
from processors import available_processors
from trainers import available_trainers

from interface import finish


def do_play(bot, params, levels, runs, prngs_seed, verbosity, **kwargs):
    """
    Evaluates the bot with params on some levels.
    """
    common_printoptions()
    prngs_seed = seed_prngs(prngs_seed)

    start = clock()
    bot_class = available_bots[bot]
    params_key, params = load_params(bot, params, verbosity)[:2]
    scores = odict()
    for level in levels:
        level = load_level(level, verbosity)
        scores[level['key']] = bot_class(level, params).evaluate(runs)
        finish(verbose=verbosity > 3)
    end = clock()

    return {'date': datetime.utcnow(),
            'bot': bot,
            'params_key': params_key,
            'levels': levels,
            'runs': runs,
            'scores': scores,
            'time': end - start,
            'prngs_seed': prngs_seed}


def do_train(bot, trainer, config, dists, emphases, phases,
             random_seeds, random_seeds_pool,
             stored_seeds, param_map, param_freeze, param_scale,
             level, eval_levels,
             runs, prngs_seed, output, verbosity, **kwargs):
    """
    Chooses the right bot and trainer, loads the level, generates or loads
    some seeds and lets the trainer work.
    """
    if output is None:
        output = first_free('params/{}'.format(bot))
    common_printoptions()
    prngs_seed = seed_prngs(prngs_seed)

    start = clock()
    bot_class = available_bots[bot]
    trainer_class = available_trainers[trainer]

    # Load the level we'll be training on.
    level = load_level(level, verbosity)

    # Convert the (index, weight) emphases to a flat tuple of weights.
    weights = [1.] * level['features']
    for index, weight in emphases:
        weights[int(index)] = float(weight)
    emphases = tuple(weights)

    # Load and/or draw some starting parameters.
    seeds = []
    for index, seed in enumerate(stored_seeds):
        params_key, params, history = load_params(bot, seed, verbosity)
        bot_ = bot_class(level, params, param_map, param_freeze, param_scale,
                         dists, emphases, phases)
        heappush(seeds, (float('inf'), index, bot_, history))
    for index in range(random_seeds_pool):
        bot_ = bot_class(level, dists=dists, emphases=emphases, phases=phases)
        score = bot_.evaluate(runs)
        heappush(seeds, (score, index + len(stored_seeds), bot_, []))
        if len(seeds) > len(stored_seeds) + random_seeds:
            heappop(seeds)
    seeds = tuple((s[2], s[3]) for s in seeds)

    trainer_ = trainer_class(level, config, dists, emphases, seeds, runs)
    params, history = trainer_.train()

    # Evaluate the params on more than just the training level.
    scores = odict()
    for key in eval_levels:
        level_ = load_level(key, verbosity)
        scores[key] = bot_class(level_, params).evaluate(runs)
    end = clock()

    # Save the final parameters with their history.
    meta = {
        'date': datetime.utcnow(),
        'bot': bot,
        'trainer': trainer,
        'config': config,
        'dists': dists,
        'emphases': emphases,
        'phases': phases,
        'seeds': (stored_seeds, random_seeds, random_seeds_pool),
        'param_map': param_map,
        'param_freeze': param_freeze,
        'param_scale': param_scale,
        'level': level,
        'runs': runs,
        'output': output,
        'scores': scores,
        'time': end - start,
        'prngs_seed': prngs_seed
    }
    history.append(meta)
    save_params(bot, output, params, history, verbosity)
    return params, meta


def do_collect(collector, level, bot, prngs_seed, output, verbosity, **kwargs):
    """
    Collects data from the level using the given collector.
    """
    if output is None:
        output = first_free('data/{}'.format(collector))
    common_printoptions()
    prngs_seed = seed_prngs(prngs_seed)

    start = clock()
    level = load_level(level, verbosity)
    bot_class = available_bots[bot[0]]
    if bot[1] is not None:
        params_key, params = load_params(*bot, verbosity=verbosity)[:2]
    else:
        params_key, params = None, {}
    bot_ = bot_class(level, params)
    collector_ = available_collectors[collector](level, bot_)
    data = collector_.collect()
    end = clock()

    meta = {
        'date': datetime.utcnow(),
        'collector': collector,
        'level': level,
        'bot': (bot[0], params_key),
        'output': '{}_{}'.format(collector, output),
        'time': end - start,
        'prngs_seed': prngs_seed
    }
    save_data(collector, output, data, meta, verbosity)
    return data, meta


def do_process(processor, input_, prngs_seed, verbosity, **kwargs):
    """
    Analyses data collected by a collector.
    """
    common_printoptions()
    prngs_seed = seed_prngs(prngs_seed)

    start = clock()
    data = tuple(load_data(key, verbosity) for key in input_)
    processor_ = available_processors[processor](data)
    results = processor_.process()
    end = clock()

    meta = {
        'date': datetime.utcnow(),
        'processor': processor,
        'input': input_,
        'time': end - start,
        'prngs_seed': prngs_seed
    }
    return results, meta
