from interface import (clear_all_checkpoints, create_checkpoint,
                       load_from_checkpoint)

from bot_base cimport BaseBot


cdef class BaseCollector:
    """
    Abstract collector class, defines the interface.
    """
    def __cinit__(self, dict level, BaseBot bot):
        """
        Initializes the collector for the given level and bot.
        """
        self.level = level
        self.bot = bot
        self.checkpoints = []

    cdef int create_checkpoint(self):
        """
        Marks game and bot state for future restoration.
        """
        checkpoint = create_checkpoint()
        bot = self.bot.clone()
        self.checkpoints.append((checkpoint, bot))
        return len(self.checkpoints) - 1

    cdef void load_checkpoint(self, int index):
        """
        Brings the game and bot state to a past point.
        """
        checkpoint, bot = self.checkpoints[index]
        load_from_checkpoint(checkpoint)
        self.bot = bot

    cdef void clear_checkpoints(self):
        """
        Removes all checkpoints (potentially freeing up a lot of space).
        """
        clear_all_checkpoints()
        self.checkpoints = []

    cpdef dict collect(self):
        """
        The main collector function, returns a dict of data to be stored.
        """
        raise NotImplementedError()
