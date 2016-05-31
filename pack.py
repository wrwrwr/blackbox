#!/usr/bin/env python3

"""
A command-line bot-packing utitlity.
"""
from os.path import getsize
from tarfile import open as taropen

from init import initialize; initialize()
from core import available_bots
from iop import parse_args, resolve_params

description = "Pack a bot with some params for submission."
arguments = (
    (('bot',), {
        'choices': available_bots.keys(),
        'help': "bot to pack"
    }),
    (('params_sets',), {
        'nargs': '+',
        'metavar': 'params',
        'help': "parameters for the bot, one or more key or bot_key"
    }),
    (( '-v', '--verbosity'), {
        'type': int,
        'default': 1,
        'help': "0 = condensed, 1 = expanded, 4+ = debugging info"
    })
)
args = parse_args(description, arguments)

for params in args.params_sets:
    params_key, params_filename = resolve_params(args.bot, params)
    pack = 'packs/{}.tar.gz'.format(params_key)
    with taropen(pack, 'w:gz') as archive:
        archive.add('packs/template/bot.py', 'bot.py')
        archive.add('packs/template/bot_wrapper.pyx', 'bot_wrapper.pyx')
        archive.add('packs/template/bot_base.pxd', 'bot_base.pxd')
        archive.add('packs/template/bot_base.pyx', 'bot_base.pyx')
        archive.add('bot_{}.pxd'.format(args.bot), 'bot_core.pxd')
        archive.add('bot_{}.pyx'.format(args.bot), 'bot_core.pyx')
        archive.add(params_filename, 'params.npz')

    if args.verbosity == 0:
        print(params_key)
    else:
        print("Packed {} with params {} as {} ({:.1f} KB).".format(
                    args.bot, params_key, pack, getsize(pack) / 1000))