#!/usr/bin/env python3

"""
A command-line tool for data munging. Select a processor and feed it with data:

    ./process.py stats srs_0

Multiple inputs can be listed and processed together:

    ./process.py stats srs_0 srs_1 srs_2
"""
from core import available_processors, do_process
from iop import date_desc, parse_args, results_desc, time_desc

description = "Process some previously collected data."
arguments = (
    (('processor',), {
        'choices': available_processors.keys(),
        'help': "processor to use"
    }),
    (('input_',), {
        'nargs': '+',
        'metavar': 'input',
        'help': "data to process (data/<data>.npz), for example srs_0"
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
    })

)

if __name__ == '__main__':
    args = parse_args(description, arguments)

    results, info = do_process(**vars(args))

    results = results_desc(results, args.verbosity, args.precision)
    if args.verbosity == 0:
        print(results.strip())
    else:
        info['date'] = date_desc(info['date'])
        info['input'] = " ".join(info['input'])
        info['time'] = time_desc(info['time'], args.precision)
        info['results'] = results
        print(("\nDate: {date}\n" +
               "Processor: {processor}\n" +
               "Input: {input}, Time: {time}\n" +
               "PRNGs: {prngs_seed}\n{results}").format(**info))
