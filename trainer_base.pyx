cdef class BaseTrainer:
    """
    Abstract trainer class, defines the interface, exposes level data.
    """
    def __cinit__(self, dict level, tuple config, dict dists, tuple emphases,
                  tuple seeds, int runs):
        """
        Initializes the trainer for the given level and config.

        The config is trainer-specific, but the first parameter is usually the
        number of steps or the time of the optimization. The distributions are
        used for generating new parameters or as further trainer configuration.
        Emphases multiply parameters directly relating to state components.
        Seeds are a tuple of (bot, params history) and runs is a number of
        times the level should be played with each parameters to average-out
        bot's randomness.
        """
        self.level = level
        self.config = config
        self.dists = dists
        self.emphases = emphases
        self.seeds = seeds
        self.runs = runs

    cpdef tuple train(self):
        """
        The main trainer function, returns the new params and their history.
        """
        raise NotImplementedError()
