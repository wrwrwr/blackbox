from cython import ccall, cclass, locals, returns


@cclass
class BaseTrainer:
    """
    Abstract trainer class, defines the interface, exposes level data.
    """
    @locals(level='dict', config='tuple', dists='dict', emphases='tuple',
            seeds='tuple', runs='int')
    def __cinit__(self, level, config, dists, emphases, seeds, runs):
        """
        Initializes the trainer for the given level and configuration.

        The config is trainer-specific, but the first parameter is usually the
        number of steps or the time of the optimization.

        The distributions are used for generating new parameters or as a
        further training configuration.

        Emphases multiply parameters directly relating to state components
        (with "state" in the key).

        Seeds should be a tuple of (bot, params history) pairs.

        Runs is a number of times the level should be played with each
        parameters to average out bot's randomness.
        """
        self.level = level
        self.config = config
        self.dists = dists
        self.emphases = emphases
        self.seeds = seeds
        self.runs = runs

    @ccall
    @returns('tuple')
    def train(self):
        """
        The main trainer function, returns the new params and their history.
        """
        raise NotImplementedError()
