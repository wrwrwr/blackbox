from copy import deepcopy
from libc.math cimport sqrt

from cython import wraparound
from numpy import array, empty, newaxis, prod, repeat
from numpy.random import randint

from interface cimport c_get_score
from interface import reset_level


cdef class BaseBot:
    """
    Abstract bot class, defines the interface.

    Exposes level and params info and provides generic implementations of
    cloning, parameter generation and mutation. Can handle emphases, phases
    (or multiple parameter sets in general), parameter mapping, freezing and
    selecting specific distributions for parameters.

    The base constructor precalculates parameter array sizes and their total
    count of entries. The only info that a subclass should provide is a
    param_shapes map from param keys to array shape tuples. The attribute
    should be set before the super call in __init__().

    Three attributes are used for the generic parameter generation and
    mutation: param_sizes maps param keys to array sizes, param_entries is
    the total count of parameters (array entries), and param_multipliers
    can be used to scale one parameter array with respect to another (by
    default nothing is scaled, but some subclasses update the multipliers).
    Frozen parameters are removed from these dictionaries after generation.

    The base class manages a self.choices array, each entry in this array is
    the number of the parameters set that should be used at the said time.
    Parameter arrays automatically get an additional inner-most axis collecting
    the values from all parameter sets. If parameters without the additional
    axis are loaded, the array is repeated with the same value for each set.
    """
    def __init__(self, dict level, dict params={}, dict param_map={},
                 tuple param_freeze=(), dict param_scale={},
                 dict dists=None, tuple emphases=None,
                 float[:] phases=array([1.], dtype='f4')):
        """
        Prepares the bot for the given level, fixing some parameters.

        You may provide a dictionary of (bot-specific) parameters or give the
        info needed to generate a new random set: a dict of distributions for
        free-values randomization and a tuple of state feature emphases.

        The param_map dict serves to translate arrays from the input seed to
        other arrays in bot's parameters. Source arrays keys should be mapped
        to lists of target keys.

        Listing params in param_freeze prevents them from being varied.

        Loaded params may be prescaled to fit them for another bot.

        If both params, and dists and emphases, are given, params may contain
        just a subset of needed parameter arrays, the rest will be randomized
        (as if parameters were newly generated).

        Phases may be a list of phase ends as level time fractions.
        If it contains anything more than a single entry (1.), the choices
        array is initialized. For instance with phases (.25, .5, .75, 1.) it
        be set to [0, 0, ..., 0, 1, 1, ..., 1, 2, 2, ..., 2, 3, 3, ..., 3].
        """
        cdef:
            dict param_shapes = self.param_shapes
            int steps = level['steps'], \
                phase, step
            float phase_end, scale
            str key, target_key
            tuple shape
            object param

        self.level = level

        if len(phases) == 1 and '_phases' in params:
            # TODO: What do we do if params have conflicting phases?
            #       Repeat or clip?
            phases = params['_phases']
        if len(phases) == 1:
            self.param_choices = 1
            self.choices = None
        else:
            self.param_choices = len(phases)
            self.choices = empty(steps, dtype='i4')
            step = 0
            for phase, phase_end in enumerate(phases):
                while step < phase_end * steps:
                    self.choices[step] = phase
                    step += 1
            for key, shape in param_shapes.items():
                param_shapes[key] = tuple(list(shape) + [self.param_choices])

        self.param_sizes = {k: prod(s) for k, s in param_shapes.items()}
        self.param_entries = sum(self.param_sizes.values())
        self.param_multipliers = {k: 1 for k in param_shapes.keys()}

        if dists is None or emphases is None:
            self.params = {}
        else:
            self.params = self.new_params(dists, emphases)
        for key, param in params.items():
            target_keys = param_map.get(key, []) + [key]
            for target_key in target_keys:
                if target_key in param_shapes:
                    if self.param_choices != 1:
                        if param.ndim == len(param_shapes[target_key]) - 1:
                            param = repeat(param[..., newaxis],
                                           self.param_choices, axis=-1)
                        elif False:
                            pass  # TODO Warn or extend, allow downgrading?
                    self.params[target_key] = deepcopy(param)
        if self.param_choices != 1:
            self.params['_phases'] = array(phases)

        for key, scale in param_scale.items():
            self.params[key] *= scale

        for key in param_freeze:
            self.param_entries -= self.param_sizes[key]
            del self.param_shapes[key]
            del self.param_sizes[key]
            del self.param_multipliers[key]

        self.last_action = -1

    cdef BaseBot clone(self, bint state=True):
        """
        Returns a copy of this bot, one that can vary parameters independently.

        If state is true, the current bot's state should be cloned, otherwise,
        bot's state should be initialized as if the bot was created anew.
        """
        cdef BaseBot bot = self.__new__(type(self), self.level)
        bot.level = self.level
        bot.param_shapes = self.param_shapes
        bot.param_sizes = self.param_sizes
        bot.param_entries = self.param_entries
        bot.param_multipliers = self.param_multipliers
        bot.param_choices = self.param_choices
        bot.choices = self.choices
        bot.params = deepcopy(self.params)
        if state:
            bot.last_action = self.last_action
        return bot

    @wraparound(True)
    cdef dict new_params(self, dict dists, tuple emphases):
        """
        Generates a new set of real parameters according to self.param_shapes,
        taking emphases into account for typical parameters.

        Parameter arrays with keys containing "state" have emphases applied to
        their entries as if they were linear or quadratic state coefficients.
        """
        cdef:
            int features = self.level['features'], \
                feature, feature0, feature1
            float emp
            dict multipliers = self.param_multipliers, \
                 params = {}
            object dist, coeffs
            str key
            tuple shape

        for key, shape in self.param_shapes.items():
            dist = dists['new'].get(key, dists['real'])
            coeffs = dist.rvs(size=shape).astype('f4') * multipliers[key]
            if 'state' in key or 'diffs' in key:
                if key[-1] == 'l':
                    for feature in range(features):
                        emp = emphases[feature]
                        if self.param_choices == 1:
                            coeffs[..., feature] *= emp
                        else:
                            coeffs[..., feature, :] *= emp
                elif key[-1] == 'q':
                    for feature0 in range(features):
                        for feature1 in range(features):
                            emp = sqrt(emphases[feature0] * emphases[feature1])
                            if self.param_choices == 1:
                                coeffs[..., feature0, feature1] *= emp
                            else:
                                coeffs[..., feature0, feature1, :] *= emp
            params[key] = coeffs
        return params

    cdef void vary_params(self, dict dists, tuple emphases, float change,
                           int variations):
        """
        Makes a couple of variations in a single call.
        """
        cdef int variation

        for variation in range(variations):
            self.vary_param(dists, emphases, change)

    @wraparound(True)
    cdef void vary_param(self, dict dists, tuple emphases, float change):
        """
        Randomly varies a single entry from parameter arrays.

        The change should be in (0, 1], where 1 means that a parameter should
        be redrawn, while .001 makes a very small variation.
        """
        cdef:
            int features = self.level['features'], \
                entry, size, feature0, feature1
            float coeff, multiplier = 1
            object array, dist
            str key

        entry = randint(self.param_entries)
        for key, size in self.param_sizes.items():
            if entry < size:
                array = self.params[key].flat
                dist = dists['vary'].get(key, dists['real'])
                multiplier = self.param_multipliers[key]
                break
            else:
                entry -= size
        if 'state' in key or 'diffs' in key:
            if key[-1] == 'l':
                multiplier *= emphases[entry // self.param_choices % features]
            elif key[-1] == 'q':
                feature0, feature1 = divmod(
                        entry // self.param_choices % features ** 2, features)
                multiplier *= sqrt(emphases[feature0] * emphases[feature1])
        coeff = dist.rvs() * multiplier
        array[entry] += change * (coeff - array[entry])

    cpdef float evaluate(self, int runs):
        """
        Estimates the value of the current parameters by running the bot
        through a complete level.

        If runs is given, repeats the level a couple of times and averages the
        score -- to account for non-determinism in bots behavior.
        """
        cdef:
            float score
            int run

        # Average score on a couple of runs.
        score = 0
        for run in range(runs):
            reset_level()
            self.act(self.level['steps'])
            score += c_get_score()
        return score / runs

    cdef void act(self, int steps):
        """
        The main bot function, performs actions not minding any consequences.

        Takes the number of actions to do. Some bots have limitations on the
        number of actions that can be played.

        The bot is responsible for storing the last action it made (-1 if it
        didn't do any) as self.last_action (for collectors use).
        """
        raise NotImplementedError()