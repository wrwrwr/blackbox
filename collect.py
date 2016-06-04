#!/usr/bin/env python3

"""
A command-line utility for gathering data. Only a collector name is required:

    ./collect.py ssa

By default the random bot is used to wander across the training level. Another
bot or level can be selected using --bot (name and params key) and --level.

Collectors vary greatly in what they do. They usually have a matching processor
or trainer that can make use of the collected data.
"""
from core import available_collectors, do_collect
from iop import date_desc, level_desc, parse_args, time_desc

description = "Collect data from a level."
arguments = (
    (('collector',), {
        'choices': available_collectors.keys(),
        'help': "collector to use"
    }),
    (('-l', '--level'), {
        'default': 'train',
        'help': "level to collect from (levels/<level>_level.data)"
    }),
    (('-b', '--bot'), {
        'nargs': 2,
        'default': ['random', None],
        'help': "bot and params key that will be used to wander the level"
    }),
    (('-i', '--iterations'), {
        'type': int,
        'default': 1,
        'help': "level runs to make, amount of data to collect"
    }),
    (('-s', '--prngs_seed'), {
        'type': int,
        'default': None,
        'help': "fixed seed for all pseudo-random number generators"
    }),
    (('-o', '--output'), {
        'default': None,
        'help': "file suffix in data/<collector>_<output>.npz"
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

    for iteration in range(args.iterations):
        data, info = do_collect(**vars(args))

        if args.verbosity == 0:
            info['level'] = info['level']['key']
            out = "{collector} {level} {output}".format(**info)
        else:
            info['date'] = date_desc(info['date'])
            info['level'] = level_desc(info['level'])
            info['time'] = time_desc(info['time'], args.precision)
            out = ("\nDate: {date}\n" +
                   "Collector: {collector}\n" +
                   "Level: {level}\n" +
                   "Bot: {bot[1]}\n" +
                   "Time: {time}\n" +
                   "Output: {output}, PRNGs: {prngs_seed}\n").format(**info)
        print(out, flush=True)
