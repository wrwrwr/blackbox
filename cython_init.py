from functools import wraps

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

        @wraps(_old_get_du_ext)
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

# Needs to be executed before other imports.
cython_on_demand(unsafe=True)
