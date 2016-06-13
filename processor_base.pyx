from cython import ccall, cclass, locals, returns


@cclass
class BaseProcessor:
    """
    Abstract processor class, defines the interface.
    """
    formats = ()

    @locals(data='tuple')
    def __init__(self, data):
        """
        Initializes the processor for the given data sets.

        The data should be a tuple of (record, meta info) pairs. The meta info
        should contain level that the data was collected from and the bot that
        was used for the purpose.

        The meta data is prescanned to find the maximum numbers of features,
        actions and steps in the set.
        """
        self.data = data
        self.records, self.meta = zip(*data)
        for meta in self.meta:
            if meta['collector'] not in self.formats:
                formats_desc = ", ".join(self.formats)
                raise ValueError(("Can only process data from the {} " +
                                  "collector(s)").format(formats_desc))
        self.bots = tuple(m['bot'] for m in self.meta)
        self.levels = tuple(m['level'] for m in self.meta)
        self.max_steps = max(l['steps'] for l in self.levels)
        self.max_actions = max(l['actions'] for l in self.levels)
        self.max_features = max(l['features'] for l in self.levels)

    @ccall
    @returns('tuple')
    @locals(entries='tuple')
    def results(self, entries):
        """
        Common parts of processors output (meta information on records).
        """
        # WA: Cython's complains if the lists are replaced with generators.
        return ((("bots", ", ".join(["{} {}".format(*b) for b in self.bots])),
                 ("levels", ", ".join([l['key'] for l in self.levels]))) +
                 entries)

    @ccall
    @returns('object')
    def process(self):
        """
        The main processor function, returns an ordered dict of data to print.
        """
        raise NotImplementedError()
