#!/usr/bin/env python3

"""
A command-line params viewer.
"""
from collections import OrderedDict as odict
from datetime import datetime
from textwrap import indent

from init import initialize; initialize()
from core import available_bots
from iop import load_params, params_desc, parse_args, seeds_desc, training_desc

description = "Display a params set and its history."
arguments = (
    (('bot',), {
        'choices': available_bots.keys(),
        'help': "the bot for which to lookup parameters"
    }),
    (('params',), {
        'help': "the parameters to display, loaded from " +
                "params/<bot>_<params>.npz"
    }),
    (('-v', '--verbosity',), {
        'type': int,
        'default': 1,
        'help': "how much info to print; 0 disables printing of the history"
    }),
    (('-p', '--precision'), {
        'type': int,
        'default': None,
        'help': "how many decimal digits of floats to print"
    })
)
args = parse_args(description, arguments)

_, params, history = load_params(args.bot, args.params, args.verbosity)

params_desc = indent(params_desc(params, args.precision), "    ")
if history:
    history_desc = ""
    trace = []
    for record in history:
        if isinstance(record, (list, tuple)):
            history_desc += "\nCombined from multiple.\n"
            continue

        # Some backwards compatibility notches.
        if record['bot'] == 'linear_1':
            record['bot'] = 'linear'
        elif record['bot'] == 'linear_2':
            record['bot'] = 'states_1'
        elif record['bot'] == 'quadratic_1':
            record['bot'] = 'quadratic'
        if (record['trainer'] == 'local_dec' or
                    record['trainer'] == 'anneal' and
                                record['date'] < datetime(2016, 5, 28, 10)):
            record['trainer'] = 'local_d'
        if record['trainer'] == 'anneal_dec':
            record['trainer'] = 'anneal'
        if 'config' not in record:
            record['config'] = "not saved"
        if 'real' not in record['dists']:
            record['dists']['real'] = record['dists'].pop('float')
        if 'unit' not in record['dists']:
            record['dists']['unit'] = record['dists'].pop('float01')
        if 'new' not in record['dists']:
            record['dists']['new'] = {}
        if 'vary' not in record['dists']:
            record['dists']['vary'] = {}
        if 'emphases' not in record:
            record['emphases'] = ()
        if 'phases' not in record:
            record['phases'] = (1.,)
        named_seeds = record['seeds'][0]
        for index, seed in enumerate(named_seeds):
            if seed.isdigit():
                named_seeds[index] = "{}_{}".format(record['bot'], seed)
            elif seed.startswith('linear_1'):
                named_seeds[index] = "linear{}".format(seed[8:])
        if not 'param_map' in record:
            record['param_map'] = record.get('params_map', {})
        for key, new_keys in record['param_map'].items():
            if isinstance(new_keys, str):
                record['param_map'][key] = [new_keys]
        if not 'param_freeze' in record:
            record['param_freeze'] = record.get('params_freeze', ())
        if not 'param_scale' in record:
            record['param_scale'] = {}
        if isinstance(record['level'], str):
            record['level'] = {'key': record['level'], 'features': -1,
                               'actions': -1, 'steps': -1}
        if 'scores' not in record:
            score = record.pop('best_score', float('-inf'))
            record['scores'] = odict(train=score)
        elif isinstance(record['scores'], tuple):
            record['scores'] = odict(train=record['scores'][0])
        if 'prngs_seed' not in record:
            record['prngs_seed'] = record.get('rand_seed', '---')

        record_desc = training_desc(record, args.verbosity, args.precision)
        history_desc += "\n{}\n".format("\n".join(record_desc))
        trace.append(seeds_desc(record['seeds'], 0))
else:
    history_desc = "\nThese parameters were saved without history.\n"
trace.append("{}_{}".format(args.bot, args.params))
if args.verbosity == 0:
    print(" ".join(trace[1:]))
else:
    print("\nParams:\n\n{}".format(params_desc) +
          "\nHistory:\n{}".format(indent(history_desc, "    ")) +
          "\nTrace:\n\n{}\n".format(indent(" - ".join(trace[1:]), "    ")))
