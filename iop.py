"""
Input, output and printing.
"""
from argparse import ArgumentParser, HelpFormatter
from contextlib import contextmanager
from os import fstat
from os.path import exists
from textwrap import indent

from numpy import (array, asscalar, get_printoptions, load, savez_compressed,
                   set_printoptions)
from scipy import stats

from interface import (get_max_time, get_num_of_actions, get_num_of_features,
                       load_level as bb_load_level)


@contextmanager
def printoptions(*args, **kwargs):
    """
    A context manager for changing NumPy print settings locally.
    """
    old_printoptions = get_printoptions()
    set_printoptions(*args, **kwargs)
    yield
    set_printoptions(**old_printoptions)


def common_printoptions():
    """
    Configures array printing for console and file output.

    Array coefficients can be pretty big, we do want to print them in whole
    however.
    If we are printing to a file it's better not to wrap lines to be able to
    scroll less and copy with ease. But when printing to the console, which
    likely is set up to wrap lines, it's better to let NumPy do the wrapping.
    """
    linewidth = 75 if fstat(0) == fstat(1) else 1e6
    set_printoptions(linewidth=linewidth, threshold=1e6)


class NoMetavarsHelpFormatter(HelpFormatter):
    """
    Skips the option destinations (normally repeated for both short and
    long option).
    """
    def _format_action_invocation(self, action):
        if action.option_strings and action.nargs != 0:
            return ", ".join(action.option_strings)
        return super()._format_action_invocation(action)


class NoMetavarsArgumentParser(ArgumentParser):
    """
    Provides a default formatter_class.
    """
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('formatter_class', NoMetavarsHelpFormatter)
        return super().__init__(*args, **kwargs)


def parse_args(description, arguments):
    """
    Parsers command-line arguments according to the specification given.

    The first argument, description, is a text included in the --help message.
    The second argument, arguments, should be a tuple of (argument flags,
    options) passed to argparse's add_argument() (respectively as positional
    and keyword arguments).
    """
    parser = NoMetavarsArgumentParser(description=description)
    for arg, opts in arguments:
        parser.add_argument(*arg, **opts)
    args = parser.parse_args()
    try:
        if args.iterations > 1 and args.output is not None:
            raise ValueError("Cannot use --output with --iterations")
    except AttributeError:
        pass
    try:
        if args.precision is None:
            args.precision = 0 if args.verbosity == 0 else 3
    except AttributeError:
        pass
    return args


def first_free(prefix):
    """
    Finds the first inexistent file of the form <prefix>_<i>.npz.
    """
    index = 0
    while exists('{}_{}.npz'.format(prefix, index)):
        index += 1
    return index


def instantiate_dist(name, *opts):
    """
    Creates distribution for the given command-line arguments.
    """
    try:
        dist = getattr(stats, name)
    except AttributeError:
        raise ValueError("No such distribution {}".format(name))
    opts = list(opts)
    try:
        loc = float(opts.pop(0))
    except IndexError:
        loc = 0
    try:
        scale = float(opts.pop(0))
    except IndexError:
        scale = 1
    return dist(*map(float, opts), loc=loc, scale=scale)


def load_level(level, verbosity):
    """
    Loads the named level and returns its name and description.
    """
    path = level
    if not path.startswith('levels/'):
        path = 'levels/{}'.format(path)
    if not path.endswith('_level.data'):
        path = '{}_level.data'.format(path)
    if verbosity > 3:
        print("Loading level from {}".format(path))
    bb_load_level(path, verbose=verbosity > 3)
    return {'key': level,
            'features': get_num_of_features(),
            'actions': get_num_of_actions(),
            'steps': get_max_time()}


def save_params(bot, key, params, history, verbosity):
    """
    Saves bot parameters and its history.

    Parameter files are zipped .npy files (saved as .npz), one for each params
    key and an additional reserved one for history.
    """
    params['__history'] = history
    path = 'params/{}_{}.npz'.format(bot, key)
    if verbosity > 3:
        print("Saving params to: {}".format(path))
    savez_compressed(path, **params)


def resolve_params(bot, key):
    """
    Finds params file identified by the given key (in the context of the bot).

    If the key is composed of only digits, adds the bot as a prefix. Otherwise
    uses the whole key, to allow using params from another bot. Moreover, if
    "last" is given as the key, finds the params with the highest numeric key.
    """
    if key == 'last':
        key = first_free('params/{}'.format(bot)) - 1
    if isinstance(key, int) or key.isdigit():
        key = '{}_{}'.format(bot, key)
    path = key
    if not path.startswith('params/'):
        path = 'params/{}'.format(path)
    if not path.endswith('.npz'):
        path = '{}.npz'.format(path)
    return key, path


def load_params(bot, key, verbosity):
    """
    Loads a set of bot parameters identified by the given key.
    """
    key, path = resolve_params(bot, key)
    if verbosity > 3:
        print("Loading params from: {}".format(path))
    with load(path) as data:
        params = dict(data)

    # Some backwards compatiblity bits.
    history = list(params.pop('__history', params.pop('history', [])))
    if 'coeffs' in params:
        if key.startswith('linear'):
            coeffs = params.pop('coeffs')
            params['constant'] = coeffs[:, -1]
            params['state0l'] = coeffs[:, :-1]
        elif key.startswith('states_1'):
            coeffs = params.pop('coeffs')
            params['constant'] = coeffs[0, :, -1] + coeffs[1, :, -1]
            params['state0l'] = coeffs[0, :, :-1]
            params['state1l'] = coeffs[1, :, :-1]

    if '_phases' in params:
        params['_phases'] = array(params['_phases'])

    return key, params, history


def save_data(collector, key, data, meta, verbosity):
    """
    Saves collected data to data/<collector>_<key>.npz.
    """
    data['__meta'] = meta
    path = 'data/{}_{}.npz'.format(collector, key)
    if verbosity > 3:
        print("Saving data to: {}".format(path))
    savez_compressed(path, **data)


def load_data(key, verbosity):
    """
    Loads collected data from data/<key>.npz.
    """
    path = key
    if not path.startswith('data/'):
        path = 'data/{}'.format(path)
    if not path.endswith('.npz'):
        path = '{}.npz'.format(path)
    if verbosity > 3:
        print("Loading data from: {}".format(path))
    with load(path) as data:
        records = dict(data)
        meta = asscalar(records.pop('__meta'))
    return records, meta


def date_desc(date):
    """
    Date as a string.
    """
    return "{:%Y-%m-%d %H:%M} (UTC)".format(date)


def dists_desc(dists):
    """
    Readable description of free parameter distributions.
    """
    def dist_desc(key):
        name = dists[key].dist.name
        args = list(dists[key].args)
        kwds = dict(dists[key].kwds)
        args.append('l={}'.format(kwds.pop('loc')))
        args.append('s={}'.format(kwds.pop('scale')))
        args.extend('{}={}'.format(k, v) for (k, v) in kwds.items())
        return "{}({})".format(name, ', '.join(map(str, args)))

    def sort_key(key):
        try:
            return str(['real', 'unit', 'variations', 'acceptance'].index(key))
        except ValueError:
            if key.startswith('new '):
                return key[4:] + 'new'
            elif key.startswith('vary '):
                return key[5:] + 'vary'
            return key

    dists = dists.copy()
    for key, dist in dists.pop('new').items():
        dists['new ' + key] = dist
    for key, dist in dists.pop('vary').items():
        dists['vary ' + key] = dist
    return "\n".join("{} {}".format(k, dist_desc(k))
                     for k in sorted(dists.keys(), key=sort_key))


def emphases_desc(emphases):
    """
    A list of (index, weight) pairs for weights not equal to one.
    """
    return ", ".join("{} {}".format(i + 1, w)
                     for i, w in enumerate(emphases) if w != 1)


def phases_desc(phases, precision):
    """
    Turns the phase splits into a more intelligible form.
    """
    precision = max(0, precision - 2)
    desc = " - ".join("{:.{}f}%".format(p * 100, precision) for p in phases)
    return "{:.{}f}% - {}".format(0, precision, desc)


def seeds_desc(seeds, verbosity):
    """
    Readable, combined description of seeding options, condensed or expanded.
    """
    named_seeds, random_seeds, random_seeds_pool = seeds
    desc = ""
    if named_seeds:
        if verbosity > 0:
            desc += "stored "
        desc += ", ".join(named_seeds)
    if random_seeds:
        if named_seeds:
            desc += ", "
            if verbosity > 0:
                desc += "and "
        if verbosity > 0:
            desc += "random "
        desc += str(random_seeds)
        if random_seeds_pool:
            desc += " out of" if verbosity > 0 else " of"
            desc += " " + str(random_seeds_pool)
    return desc


def param_map_desc(param_map):
    """
    Stringifies param_map dict in an "arrowy" style.
    """
    return "\n".join("{}: {}".format(k, " ".join(nks))
                     for k, nks in param_map.items())


def param_scale_desc(param_scale):
    """
    Human-readable version of prescaling directives.
    """
    return ", ".join("{} {}".format(k, s) for k, s in param_scale.items())


def level_desc(level):
    """
    Short, readable level and its parameters string.
    """
    return "{key} ({features}, {actions}, {steps})".format(**level)


def time_desc(duration, precision):
    """
    Humanized duration.
    """
    minutes, seconds = divmod(duration, 60)
    hours, minutes = divmod(minutes, 60)
    desc = "{:.{}f} s".format(seconds, precision)
    if minutes > 0 or hours > 0:
        desc = "{} m, {}".format(int(minutes), desc)
        if hours > 0:
            desc = "{} h, {}".format(int(hours), desc)
    return desc


def scores_desc(scores, verbosity, precision):
    """
    Formats a list of scores (from different levels) for display.
    """
    if verbosity == 0:
        return " ".join("{:.{}f}".format(s, precision)
                        for s in scores.values())
    else:
        return ", ".join("{} {:.{}f}".format(l, s, precision)
                         for l, s in scores.items())


def training_desc(info, verbosity, precision):
    """
    Textifies a single training history record. Returns a list of lines.
    """
    desc = [
        "Date: {}".format(date_desc(info['date'])),
        "Bot: {bot}".format(**info) +
                ", Trainer: {trainer} {config}".format(**info),
        "Dists: {}".format(indent(dists_desc(info['dists']), " " * 7).strip()),
        "Seeds: {}".format(seeds_desc(info['seeds'], verbosity)),
        "Level: {}".format(level_desc(info['level'])) +
                ", Runs: {}".format(info['runs']) +
                ", Time: {}".format(time_desc(info['time'], precision)),
        "Output: {output}, PRNGs: {prngs_seed}".format(**info) +
                ", Scores: {}".format(scores_desc(info['scores'], verbosity,
                                                  precision))
    ]
    if info['phases']:
        desc.insert(3, "Phases: {}".format(
                                    phases_desc(info['phases'], precision)))
    if info['emphases']:
        desc.insert(3, "Emphases: {}".format(emphases_desc(info['emphases'])))
    if info['param_scale']:
        desc.insert(5, "Scaled params: {}".format(
                                        param_scale_desc(info['param_scale'])))
    if info['param_freeze']:
        desc.insert(5, "Frozen params: {}".format(
                                        ", ".join(info['param_freeze'])))
    if info['param_map']:
        desc.insert(5, "Params map: {}".format(
                indent(param_map_desc(info['param_map']), " " * 12).strip()))
    return desc


def params_desc(params, precision):
    """
    Human-readable description of bot parameters.
    """
    with printoptions(precision=precision):
        desc = ""
        for key, value in sorted(params.items()):
            if key != 'history' and key != '__history':
                value = indent(repr(value), " " * (len(key) + 4)).strip()
                desc += "'{}': {}\n".format(key, value)
    return desc


def results_desc(results, verbosity, precision):
    """
    Nicely formatted processor results.
    """
    with printoptions(precision=precision, suppress=True):
        desc = ""
        for key, value in results.items():
            value = str(value)
            if verbosity > 0:
                value = indent(value, "    ")
                desc += "\n{}:\n\n{}\n".format(key, value)
            else:
                desc += "{}\n".format(value)
    return desc
