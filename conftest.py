"""
Monkey-patch pycodestyle, to let it process cimports as imports, and allow
using odict (lowercase) as an alias for collections.OrderedDict.
"""


def replace(line):
    return line.replace('cimport', 'import').replace('odict', 'ODict')


def patch_checker(checker_class):
    old_checker_init = checker_class.__init__

    def checker_init(self, *args, **kwargs):
        old_checker_init(self, *args, **kwargs)
        self.lines = list(map(replace, self.lines))

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
