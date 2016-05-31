"""
Bot are Cython extensions, so we need a wrapper to run them from Python.
"""
from bot_core cimport Bot


cpdef void do_act(dict level, dict params):
    Bot(level, params).act(level['steps'])
