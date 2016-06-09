#!/usr/bin/env python3

"""
A minimal bot player.

Loads the level and params and lets the bot act.
"""
from interface import (get_max_time, get_num_of_actions, get_num_of_features,
                       finish, load_level)
from numpy import get_include, load
from pyximport import install

install(setup_args={'include_dirs': get_include()}, reload_support=True)
from bot_wrapper import do_act

if __name__ == '__main__':
    load_level('../levels/train_level.data', verbose=1)
    level = {
        'steps': get_max_time(),
        'actions': get_num_of_actions(),
        'features': get_num_of_features()
    }
    params = dict(load('params.npz'))
    do_act(level, params)
    finish(verbose=1)
