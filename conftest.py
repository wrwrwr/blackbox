"""
Monkey-patch pycodestyle, to let it process cimports as imports.
"""


def patch_checker(checker_class):
    old_checker_init = checker_class.__init__

    def checker_init(self, *args, **kwargs):
        old_checker_init(self, *args, **kwargs)
        self.lines = [l.replace('cimport', 'import') for l in self.lines]

    checker_class.__init__ = checker_init

try:
    from pycodestyle import Checker
    patch_checker(Checker)
except ImportError:
    pass

try:
    from pep8 import Checker
    patch_checker(Checker)
except ImportError:
    pass
