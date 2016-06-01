from collections import OrderedDict as odict


cdef class BaseProcessor:
    """
    Abstract processor class, defines the interface.
    """
    def __cinit__(self, tuple data):
        """
        Initializes the processor for the given data sets.

        The data should be a tuple of (record, level info) pairs. The level
        infos are prescanned to find the maximum features, actions and steps
        in the set.
        """
        self.data = data
        self.max_features = max(m['level']['features'] for _, m in data)
        self.max_actions = max(m['level']['actions'] for _, m in data)
        self.max_steps = max(m['level']['steps'] for _, m in data)

    cdef object results(self, tuple entries):
        """
        Common parts of processors output (meta information on records).
        """
        bots = ("{} {}".format(*meta['bot']) for _, meta in self.data)
        levels = (meta['level']['key'] for _, meta in self.data)
        info = odict((
            ("bots", ", ".join(bots)),
            ("levels", ", ".join(levels))))
        info.update(entries)
        return info

    cpdef object process(self):
        """
        The main processor function, returns an ordered dict of data to print.
        """
        raise NotImplementedError()
