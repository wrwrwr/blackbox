blackbox
========

A specialized optimization toolkit for the [BlackBox Challenge][] competition.

The problem, in short, is to teach an agent to play a game with unknown rules.
The game consists of a number of steps, at each step the agent chooses one of
the available actions (a small integer) seeing just some undescribed state (a
bunch of floats), and gets a score update in return.

Such a formulation covers a wide subset of [reinforced learning problems][rlp],
allowing the toolkit to be potentially used for various optimization tasks.

[BlackBox Challenge]: http://blackboxchallenge.com/
[rlp]: https://webdocs.cs.ualberta.ca/~sutton/book/ebook/node27.html

#### Cons

Just a short excerpt from a few pages long list of notes and todos:

* In its current form, the toolkit presents just a simplistic, evolutionary
  approach, all of the coded trainers treat the blackbox opaquely, not even
  looking at intermediate level scores.
* The optimization process still requires a human to choose the results for
  further extension and refinement; and humans are not that easy to come by ;-)
* Parts of the toolkit are over-optimized; for instance, trainers that could
  benefit from statically compiled code are quite imaginable, but none of the
  implemented ones really needs it.
* The bot interface should be extended to allow more sophisticated trainers.

#### Pros

Although the framework is a simple Monte Carlo simulator at its heart, it has
some nice attributes:

* Simple interface between bots and trainers, each trainer can train every bot.
* C-speed bots, trainers, collectors, and processors; writing a new bot only
  requires describing its parameters and coding a single function.
* Extensive training infrastructure: seed trainers with previous results, move
  parts of parameter sets between bots, use different parameter sets for parts
  of levels, or emphasize components of the state.
* Most of the training configuration is done using probability distributions,
  rather than fixed constants, giving a lot of flexibility.

## Installation

Requires [Python][] >= 3.4, [NumPy][] >= 1.11, and [Cython][] >= 0.24.
Currently the only way to use the toolkit is to clone the source from GitHub
and get the [game interface and levels][] from the competition's site.

Copy the ``levels`` folder, ``interface.pxd`` (from the ``fast_bot`` folder),
and the proper ``interface-X.so`` for your environment from the package into
the folder that this file is in.

Alternatively, code an interface for your own problem according to the
[specification][]! Both Python and C API has to be provided, but there are no
requirements on the levels content or format, other than that the interface can
use them.

The file and directory structure should look as follows:

```
    blackbox
    ├── data                (records gathered by collectors)
    ├── levels              (should hold train/test_level.data)
    ├── packs               (self-contained bots for sharing)
    ├── params              (stored parameters, NumPy's format)
    ├ bot_*.pyx             (bot implementations)
    ├ bots.pyx              (add new bots here)
    ├ trainer_*.pyx         (trainer implementations)
    ├ trainers.pyx          (add new trainers here)
    ...
    ├ interface.[pxd,so]    (the game interface should be here)
```

[Python]: https://www.python.org/
[NumPy]: http://www.numpy.org/
[Cython]: http://cython.org/
[game interface and levels]: http://blackboxchallenge.com/static/blackbox.zip
[specification]: http://blackboxchallenge.com/specs/

## Commands

The toolkit consists of six command-line utilities. To see a full list of
options for any of them invoke it with ``--help``. Some expanded descriptions
and more examples can be found within their module docstrings.

### [train][]

Takes a bot name, trainer name and a configuration for it (plus a ton of
optional parameters). For example:

```bash
    ./train.py linear local 100 1
```

starts a random local search for a set of coefficients for a linear regression
bot.

[train]: train.py

### [play][]

Evaluates a bot on selected levels and prints out the scores. For instance:

```bash
    ./play.py linear 0
```

would play the bot with the coefficients found in the previous section on all
available levels.

[play]: play.py

### [view][]

Lets you see a set of coefficients and their history. To review the
coefficients generated above you would type:

```bash
    ./view.py linear 0
```

[view]: view.py

### [collect][]

Plays a bot on a level and stores some data, such as consequent states, actions
taken, or intermediate rewards:

```bash
    ./collect.py srs --bot linear 0
```

[collect]: collect.py

### [process][]

Analyzes data sets gathered by a collector to display statistics or preprocess
them for a trainer.

```bash
    ./process.py stats srs 0
```

[process]: process.py

### [pack][]

Creates a standalone bot package (that still requires the game interface
and levels).

[pack]: pack.py

## Bots

The agents choosing the actions. 

All current bot implementations assume 4 actions, some can only play a
particular number of steps on each ``act()`` call.

### [linear][]

A simple regression bot, optimized.

[linear]: bot_linear.pyx

### [belief][]

Bots that try to simulate an entry, or a few, of the hidden state, and evaluate
actions using state reinforced with beliefs.

The present bots simply update the beliefs linearly based on their previous
values and the visible state, and use linear regression for action choosing.

[belief]: bot_belief_1.pyx

### [states][]

Linear regression using the current and the last state.

[states]: bot_states_1.pyx

### [diffs][]

Approximates action value as a linear function of the state components and
their derivatives (with respect to level time). Or actually, uses finite
backward differences to approximate the derivatives.

[diffs]: bot_diffs_1.pyx

### [quadratic][]

Adds a coefficient for each pair of state components.

[quadratic]: bot_quadratic.pyx

### [_m suffix][]

Bots that can use different sets of coefficients at different parts of the
level. The "parts" may be phases (first half, second half), congruences (even
step, odd step) or some other temporal patterns.

[_m suffix]: bot_linear_m.pyx

## Trainers

The problem solvers, or optimization algorithms.

### [local][]

Randomly changes a number of parameter entries at each step. Can either redraw
values anew or do slight adjustments, according to a variation scale given.

[local]: trainer_local.pyx

### [anneal][]

Almost a random local search, but sometimes accepts score drops. Useful when
you feel you are stuck in a local maximum

[anneal]: trainer_anneal.pyx

### [_d suffix][]

Instead of taking a single variation scale, these variants are configured with
a high and low value and gradually diminish parameter variations during the
course of a training session.

[_d suffix]: trainer_local_d.pyx

### [select][]

Chooses the best from the seeds provided.

[select]: trainer_select.pyx

### [comb][]

Combines several parameter sets into a single multi-parameter set (for an "_m"
bot).

[comb]: trainer_comb.pyx

### [comb_phases][]

Tries all phase assignments printing their evaluations.

[comb_phases]: trainer_comb_phases.pyx

## Collectors

Collectors run a bot through a level, possibly moving back in time by using
checkpoints, observe various things such as states, rewards or optimal paths
and save them for later processing.

### [srs][]

Checks possible immediate rewards after each state.

[srs]: collector_srs.pyx

### [ssa][]

Stores states encountered, scores seen, and actions done during a playthrough.

[ssa]: collector_ssa.pyx

### [sss][]

Finds all states that could have come after each state.

[sss]: collector_sss.pyx

## Processors

Processors are meant to analyze the data gathered by collectors. Such a split
of responsibilities lets you quickly realize new investigations.

### [init][]

How do the first few states look? How do they depend on the first few actions?

[init]: processor_init.pyx

### [stats][]

How does the state look on average? What are typical rewards for actions?

[stats]: processor_stats.pyx

### [corrs][]

Are the state components independent from each other?

[corrs]: processor_corrs.pyx

### [phases][]

Is the level composed of smaller sublevels?

[phases]: processor_phases.pyx

### [pt][]

What was going on during that playthrough?

[pt]: processor_pt.pyx
