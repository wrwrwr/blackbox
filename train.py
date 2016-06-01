#!/usr/bin/env python3

"""
The main bot-training command-line utility. To start a training session at
least 3 things need to be chosen:

* the bot to train, this should be a key such as linear or diffs_1;
* the trainer that will carry out the training, say local or anneal;
* configuration for the trainer, a varying amount of additional
  arguments that are passed to the trainer, the first one is usually
  the number of steps.

After that many different options can be given, some notable are:

--iterations
    controls the count of full training sessions; best parameters from each one
    are saved to params/<bot>_<index>.npz;

--random_seeds and --stored_seeds
    serve to choose the starting point, either the best params out of a random
    pool or a set loaded from a file;

--dist_new_param and --dist_vary_param
    can be used to specify a distribution for a parameter array generation or
    mutation, they take the name of the array and a SciPy stats distribution
    key and arguments;

--dist_variations
    determines the number of parameter variations to make at each trainer step,
    the more spread this one is the wider the search;

--dist_acceptance
    trainers such as anneal have a way of escaping local maxima by sometimes
    accepting a score decline, this distribution describes this process;

--emphasis
    can be used to hint trainers that some components of state are more or less
    important than other;

--param_map, --param_freeze, and --param_scale
    are there to translate parts of a parameter set from one bot to another.

A simple invocation could look as follows:

    ./train.py linear local 1000

This would make a thousand steps of the random local search trainer, starting
from a random parameters for the linear bot. If this was your first run, the
results would be saved as params/linear_0.npz`.

To start from a previous seed and emphasize the last state component ten-fold:

    ./train.py linear local 1000 --stored_seed 0 --emphasis 36 10

And to now reuse your linear coefficients for the diffs_1 bot:

    ./train.py diffs_1 local 1000 -ss 1 -dnp diffs0l norm 0 .01
"""
from numpy import array

from core import available_bots, available_trainers, do_train
from iop import (instantiate_dist, params_desc, parse_args, scores_desc,
                 training_desc)

description = "Train a bot to optimize its parameters."
arguments = (
    (('bot',), {
        'choices': available_bots.keys(),
        'help': "bot to train"
    }),
    (('trainer',), {
        'choices': available_trainers.keys(),
        'help': "trainer to use"
    }),
    (('config',), {
        'nargs': '*',
        'default': [],
        'help': "options to pass to the trainer (some may be required)"
    }),
    (('-dr', '--dist_real'), {
        'nargs': '+',
        'default': ['norm', '0', '11'],
        'help': "distribution of free, real parameters; (a scipy.stats name " +
                "with loc and scale args); for example: norm 0 9"
    }),
    (('-du', '--dist_unit'), {
        'nargs': '+',
        'default': ['beta', '0', '1', '1.2', '1.2'],
        'help': "distribution of free, real parameters limited to the " +
                "[0, 1] interval; for example: beta 0 1 1.2 1.2"
    }),
    (('-dv', '--dist_variations'), {
        'nargs': '+',
        'default': ['beta', '1', '27', '2', '6'],
        'help': "distribution of the number of changes to make per trainer " +
                "step (should be only supported above 1)"
    }),
    (('-da', '--dist_acceptance'), {
        'nargs': '+',
        'default': ['beta', '-10', '.001', '19', '1'],
        'help': "distribution for accepting param sets value changes; if it " +
                "is supported below zero, declines may be accepted; if it " +
                "supported above zero, growths may be rejected"
    }),
    (('-dpn', '--dist_param_new'), {
        'nargs': '+',
        'action': 'append',
        'default': [],
        'help': "choose a distribution for a specific param generation, " +
                "for example: diffs0l norm 0 .001"
    }),
    (('-dpv', '--dist_param_vary'), {
        'nargs': '+',
        'action': 'append',
        'default': [],
        'help': "choose a distribution for a specific param mutation; " +
                "(mutation redraws and adds scaled difference)"
    }),
    (('-e', '--emphasis'), {
        'dest': 'emphases',
        'nargs': 2,
        'action': 'append',
        'type': float,
        'default': [[-1, 111]],
        'help': "hint for the trainer to put emphasis on some state " +
                "components; pairs of numbers: index, relevance"
    }),
    (('-ph', '--phases'), {
        'nargs': '+',
        'type': float,
        'help': "fix or override phase splits for a multi-param-sets bot; " +
                "for example .25 .5 .75 divides levels into quarters"
    }),
    (('-rs', '--random_seeds'), {
        'type': int,
        'nargs': '+',
        'default': [0, 0],
        'help': "pool of random seeds to choose from, and optionally the " +
                "number of seeds to pass to the trainer"
    }),
    (('-ss', '--stored_seeds'), {
        'nargs': '+',
        'default': [],
        'help': "starting parameters to pass to the trainer; a list " +
                "of param keys, expanded to params/<bot>_<seed>.npz"
    }),
    (('-pm', '--param_map'), {
        'nargs': 2,
        'action': 'append',
        'default': [],
        'help': "reuse param from a stored seed under a different key; " +
                "\"state0l state1l\" uses state coeffs for last state"
    }),
    (('-pf', '--param_freeze'), {
        'nargs': '+',
        'default': [],
        'help': "do not let trainers vary some parameter arrays"
    }),
    (('-ps', '--param_scale'), {
        'nargs': 2,
        'action': 'append',
        'default': [],
        'help': "scale some stored params, to adapt it for another bot"
    }),
    (('-l', '--level'), {
        'default': 'train',
        'help': "level to train on (levels/<level>_level.data)"
    }),
    (('-el', '--eval_levels'), {
        'default': ['train', 'test'],
        'help': "levels to evaluate after the training session"
    }),
    (('-i', '--iterations'), {
        'type': int,
        'default': 1,
        'help': "count of full trainer executions (with the same config)"
    }),
    (('-r', '--runs'), {
        'type': int,
        'default': 1,
        'help': "number of repetitions for each params evaluation " +
                "(for non-deterministic bots)"
    }),
    (('-s', '--prngs_seed'), {
        'type': int,
        'default': None,
        'help': "fixed seed for all pseudo-random number generators"
    }),
    (('-o', '--output'), {
        'default': None,
        'help': "file suffix included in params/<bot>_<output>.npz"
    }),
    (('-v', '--verbosity'), {
        'type': int,
        'default': 1,
        'help': "0 = condensed, 1 = expanded, 4+ = debugging info"
    }),
    (('-p', '--precision'), {
        'type': int,
        'default': None,
        'help': "how many decimal digits of floats to print"
    })
)

if __name__ == '__main__':
    args = parse_args(description, arguments)
    args.config = tuple(args.config)
    args.dists = {
        'real': instantiate_dist(*args.dist_real),
        'unit': instantiate_dist(*args.dist_unit),
        'new': {d[0]: instantiate_dist(*d[1:]) for d in args.dist_param_new},
        'vary': {d[0]: instantiate_dist(*d[1:]) for d in args.dist_param_vary},
        'variations': instantiate_dist(*args.dist_variations),
        'acceptance': instantiate_dist(*args.dist_acceptance)
    }
    if args.phases is not None:
        if any(not 0 < p < 1 for p in args.phases):
            raise ValueError("--phases takes a list of phase ends, each " +
                             "should be a level time fraction in (0, 1)")
        args.phases = array(args.phases + [1.], dtype='f4')
    if not 1 <= len(args.random_seeds) <= 2:
        # WA: https://bugs.python.org/issue11354.
        raise ValueError("--random_seeds can only take one or two arguments")
    if not args.stored_seeds and args.random_seeds[0] == 0:
        args.random_seeds = [1, 1]
    args.random_seeds_pool = args.random_seeds.pop(0)
    args.random_seeds = args.random_seeds[0] if args.random_seeds else 1
    if args.random_seeds_pool < args.random_seeds:
        raise ValueError("Cannot choose {random_seeds} seeds from a pool of " +
                         "just {random_seeds_pool}".format(**args))
    param_map = {}
    for key, new_key in args.param_map:
        param_map.setdefault(key, []).append(new_key)
    args.param_map = param_map
    args.param_freeze = tuple(args.param_freeze)
    args.param_scale = {k: float(s) for k, s in args.param_scale}

    for iteration in range(args.iterations):
        params, info = do_train(**vars(args))

        if args.verbosity == 0:
            scores = scores_desc(info['scores'], args.verbosity,
                                 args.precision)
            out = "{} {} {}".format(info['bot'], info['output'], scores)
        else:
            info_desc = training_desc(info, args.verbosity, args.precision)
            out = "{}\n{}".format("\n".join(info_desc), params_desc(params, 8))
        print(out, flush=True)
