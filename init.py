from functools import wraps
from os import fstat

import numpy
import pyximport.pyximport


def cython_on_demand(unsafe):
    """
    Enables loading .pyx files from .py files (on-demand compilation).

    With `unsafe` deactivates all Cython safety checks and compatibility
    options (do not use without first testing that things work reliably).
    """
    if unsafe:
        _old_get_du_ext = pyximport.pyximport.get_distutils_extension

        wraps(_old_get_du_ext)
        def _new_get_du_ext(*args, **kwargs):
            extension, setup_args = _old_get_du_ext(*args, **kwargs)
            directives = getattr(extension, 'cython_directives', {})
            directives.update({
                'language_level': 3,
                'boundscheck': False,
                'wraparound': False,
                'initializedcheck': False,
                'cdivision': True,
                'always_allow_keywords': False
            })
            extension.cython_directives = directives
            return extension, setup_args

        pyximport.pyximport.get_distutils_extension = _new_get_du_ext

    pyximport.install(setup_args={'include_dirs': numpy.get_include()})


def numpy_printoptions():
    """
    Configures array printing for console and file output.

    Array coefficients can be pretty big, we do want to print them in whole
    however.
    If we are printing to a file it's better not to wrap lines to be able to
    scroll less and copy with ease. But when printing to a console, which
    likely is set up to wrap lines, it's better to let NumPy do the wrapping.
    """
    linewidth = 75 if fstat(0) == fstat(1) else 1e6
    numpy.set_printoptions(linewidth=linewidth, threshold=1e6)


def initialize(cython_unsafe=False):
    """
    Performs common global initialization steps.
    """
    cython_on_demand(unsafe=cython_unsafe)
    numpy_printoptions()
