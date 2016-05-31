#!/usr/bin/env python3

"""
A command-line bot runner. Takes a bot name and a parameters key:

    ./play.py linear 0

Bots can be run with parameter sets from other bots as long as they contain
the needed params (matched by key):

    ./play.py linear diffs_1_0

Multiple parameter sets can be evaluated at once with a bit of shell magic:

    ./play linear `ls params/linear_*.npz` --ignore-exceptions
"""
from init import initialize; initialize()
from core import available_bots, do_play
from iop import date_desc, level_desc, parse_args, scores_desc, time_desc

description = "Evaluate a bot with params on a level."
arguments = (
    (('bot',), {
        'choices': available_bots,
        'help': "bot to run"
    }),
    (('params_sets',), {
        'nargs': '+',
        'metavar': 'params',
        'help': "parameters for the bot, one or more key or bot_key, " +
                "or \"last\"; for instance: 7 12 last linear_1_5"
    }),
    (('-l', '--levels'), {
        'nargs': '+',
        'choices': ('train', 'test'),
        'default': ['train', 'test'],
        'help': "levels to test on (levels/<level>_level.data)"
    }),
    (('-r', '--runs'), {
        'type': int,
        'default': 1,
        'help': "number of repetitions (over which to average scores)"
    }),
    (('-s', '--prngs_seed'), {
        'type': int,
        'default': None,
        'help': "fixed seed for all pseudo-random number generators"
    }),
    (( '-v', '--verbosity'), {
        'type': int,
        'default': 1,
        'help': "0 = condensed, 1 = expanded, 4+ = debugging info"
    }),
    (('-p', '--precision'), {
        'type': int,
        'default': None,
        'help': "how many decimal digits of floats to print"
    }),
    (('-ie', '--ignore_exceptions'), {
        'action': 'store_true',
        'default': False,
        'help': "continue with following param sets if one fails"
    })
)
args = parse_args(description, arguments)

for params in args.params_sets:
    try:
        info = do_play(params=params, **vars(args))
    except:
        if not args.ignore_exceptions:
            raise
        else:
            continue

    info['scores'] = scores_desc(info['scores'], args.verbosity,
                                 args.precision)
    if args.verbosity == 0:
        print("{bot} {params_key} {scores}".format(**info))
    else:
        info['date'] = date_desc(info['date'])
        info['time'] = time_desc(info['time'], args.precision)
        print(("\nDate: {date}\n" +
               "Bot: {bot}, params: {params_key}, runs: {runs}\n" +
               "Scores: {scores}\n" +
               "Time: {time}, PRNGs: {prngs_seed}").format(**info))
if args.verbosity != 0:
    print()
