========
blackbox
========

Specialized optimization toolkit for the `BlackBox Challenge`_ competition.

In short, the problem is to prepare an agent to play a game with unknown rules.
The game consists of a number of steps, at each step the agent chooses one of
available actions (a small integer) seeing some undescribed state (a bunch of
floats) and getting a score update in return.

TODO: Is it?
Such a formulation is a fairly general reinforced learning problem statement,
allowing the framework to be potentially used for various optimization tasks.

.. _BlackBox Challenge: http://blackboxchallenge.com/

Intallation
-----------

Requires Python_ >= 3.4, NumPy_ >= 1.11, and Cython_ >= 0.24.

Copy the ``levels`` folder, ``interface.pxd`` (look in the ``fast_bot`` folder)
and the proper interface-X.so for your environment from the competition package
into the folder this file is in.
Or code an interface for your own problem according to the specification_ (both
Python and C API has to be provided).

The file and directories looks like this:

::

    blackbox
    ├── data                (records gathered by collectors)
    ├── levels              (with train/test_level.data)
    ├── packs               (self-contained bots for sharing)
    ├── params              (stored parameters, NumPy's format)
    ├ bot_*.pyx             (bot implementations)
    ├ bots.pyx              (add new bots here)
    ├ trainer_*.pyx         (trainer implementations)
    ├ trainers.pyx          (add new trainers here)
    ...
    ├ inteface.[pxd,so]     (the game inteface should be here)

.. _Python: https://www.python.org/
.. _NumPy: http://www.numpy.org/
.. _Cython: http://cython.org/
.. _specification: http://blackboxchallenge.com/specs/

Advantages
~~~~~~~~~~

The framework is a simple Monte Carlo simulator at its heart, but it has some
nice attributes:

* simple interface between bots and trainers, each trainer can train every bot;
* C-speed bots, trainers, collectors, and processors; writing a new bot only
  requires describing its parmeters and coding a single function;
* extensive training infrastructure: seed trainers with previous results, move
  parts of parameter sets between bots, use different parameter sets for parts
  of levels, or emphasize components of the state;
* most of trainer configuration is done using probability distributions, rather
  than fixed constants, giving a lot of flexibility.

Commands
--------

The toolkit consists of five basic command-line utilities. To see a full
list of optinos for any of them invoke it with ``--help``.

train
~~~~~

The main optimization command. Takes a bot name, trainer name and a config
for it, plus a ton of optional parameters. For example:

.. code:: bash

    ./train.py linear local 100

Is the simplest possible random local search for a set of coefficients for the
linear regression bot.

play
~~~~

Plays a bot with some given paremeters. For example:

.. code:: bash

    ./play.py linear 0

Would play the bot with the coefficients found in the previous section on all
available levels.

view
~~~~

Shows the coefficients and their history.

collect
~~~~~~~

Plays a bot on a level and stores some data, such as consequent staates,
actions taken, intermediate rewards or optimal actions at each step.

process
~~~~~~~

Analyzes some data sets gathered by a collector to display some statistics,
or preprocess them for a trainer.

Bots
----

All current bot implementations assume 4 actions, some can only play a
particular number of actions on each ``act()`` call.

linear
~~~~~~

A simple regression bot, just optimized.

belief
~~~~~~

Bots that tries to manage a value or a few of the hidden state, and evaluate
actions using state reinforced with the beliefs.

The implemented bots simply update the beliefs linearly based on their previous
values and the visible state, and use linear regression for action choosing.

states
~~~~~~

Linear regression on the current and a few previous states.

diffs
~~~~~

Approximates final return as a linear function of the state components (seen
as functions of level time) and their derivatives. Or actually, uses finite
backward differences to approximate the derivatives.

quadratic
~~~~~~~~~

Adds a coefficient for each pair of state components.

m suffix
~~~~~~~~

Bots that can use different sets of coefficients at different points of the
level. The points may be phases (first half, second half), congruences (even
steps, odd steps) or some other arbitrary temporal pattern.

Trainers
--------

local
~~~~~

Randomly changes a number of parameter entries at each step. Can either redraw
values anew or do some slight adjustments, according to configuration.

anneal
~~~~~~

Almost a random local search, but sometimes accepts score drops.

dec suffix
~~~~~~~~~~

Diminishing variations.

select
~~~~~~

Chooses the best from the seeds provided.

comb
~~~~

Combines several parameter sets into a single multi-parameter set (for a "m"
bot -- phases, congruences and other temporal choices).

comb_phases
~~~~~~~~~~~

Tries all phase assignments printing their evaluations.

Collectors
----------

sas
~~~

srs
~~~

sss
~~~

Processors
----------

init
~~~~

stats
~~~~~

corrs
~~~~~

pt
~~
