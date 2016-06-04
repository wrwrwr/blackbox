from cython import ccall, cclass, cfunc, locals, returns

from interface import (clear_all_checkpoints, create_checkpoint,
                       load_from_checkpoint)


@cclass
class BaseCollector:
    """
    Abstract collector class, defines the interface.
    """
    @locals(level='dict', bot='BaseBot')
    def __cinit__(self, level, bot):
        """
        Initializes the collector for the given level and bot.
        """
        self.level = level
        self.bot = bot
        self.checkpoints = []

    @cfunc
    @returns('int')
    @locals(checkpoint='int', bot='BaseBot')
    def create_checkpoint(self):
        """
        Marks game and bot state for future restoration.
        """
        checkpoint = create_checkpoint()
        bot = self.bot.clone()
        self.checkpoints.append((checkpoint, bot))
        return len(self.checkpoints) - 1

    @cfunc
    @returns('void')
    @locals(index='int', checkpoint='int', bot='BaseBot')
    def load_checkpoint(self, index):
        """
        Brings the game and bot state to a past point.
        """
        checkpoint, bot = self.checkpoints[index]
        load_from_checkpoint(checkpoint)
        self.bot = bot

    @cfunc
    @returns('void')
    def clear_checkpoints(self):
        """
        Removes all checkpoints (potentially freeing up some memory).
        """
        clear_all_checkpoints()
        self.checkpoints = []

    @ccall
    @returns('dict')
    def collect(self):
        """
        The main collector function, returns a dict of data to be stored.
        """
        raise NotImplementedError()
