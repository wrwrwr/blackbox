from collections import OrderedDict as odict

from cython import ccall, cclass, cfunc, locals, returns


@cclass
class BaseProcessor:
    """
    Abstract processor class, defines the interface.
    """
    @locals(data='tuple')
    def __cinit__(self, data):
        """
        Initializes the processor for the given data sets.

        The data should be a tuple of (record, level info) pairs. The level
        infos are prescanned to find the maximum features, actions and steps
        in the set.
        """
        self.data = data
        # WA: Cython doesn't like just "data" in the following three.
        self.max_features = max(m['level']['features'] for _, m in self.data)
        self.max_actions = max(m['level']['actions'] for _, m in self.data)
        self.max_steps = max(m['level']['steps'] for _, m in self.data)

    @cfunc
    @returns('object')
    @locals(entries='tuple')
    def results(self, entries):
        """
        Common parts of processors output (meta information on records).
        """
        # WA: Cython's compiler crashes if the following are generators.
        bots = ["{} {}".format(*m['bot']) for _, m in self.data]
        levels = [m['level']['key'] for _, m in self.data]
        info = odict((
            ("bots", ", ".join(bots)),
            ("levels", ", ".join(levels))))
        info.update(entries)
        return info

    @ccall
    @returns('object')
    def process(self):
        """
        The main processor function, returns an ordered dict of data to print.
        """
        raise NotImplementedError()
