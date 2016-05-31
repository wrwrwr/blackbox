from collections import OrderedDict as odict

from bot_belief_1 cimport Bot as BotBelief1
from bot_belief_4 cimport Bot as BotBelief4
from bot_belief_f cimport Bot as BotBeliefF
from bot_diffs_1 cimport Bot as BotDiffs1
from bot_diffs_1_m cimport Bot as BotDiffs1M
from bot_diffs_2 cimport Bot as BotDiffs2
from bot_diffs_3 cimport Bot as BotDiffs3
from bot_diffs_4 cimport Bot as BotDiffs4
from bot_diffs_4_m cimport Bot as BotDiffs4M
from bot_linear cimport Bot as BotLinear
from bot_linear_m cimport Bot as BotLinearM
from bot_phases_1 cimport Bot as BotPhases1
from bot_quadratic cimport Bot as BotQuadratic
from bot_random cimport Bot as BotRandom
from bot_random_1 cimport Bot as BotRandom1
from bot_random_5 cimport Bot as BotRandom5
from bot_random_n cimport Bot as BotRandomN
from bot_simi cimport Bot as BotSimi
from bot_states_1 cimport Bot as BotStates1

available_bots = odict((
    ('belief_1', BotBelief1),
    ('belief_4', BotBelief4),
    ('belief_f', BotBeliefF),
    ('diffs_1', BotDiffs1),
    ('diffs_1_m', BotDiffs1M),
    ('diffs_2', BotDiffs2),
    ('diffs_3', BotDiffs3),
    ('diffs_4', BotDiffs4),
    ('diffs_4_m', BotDiffs4M),
    ('linear', BotLinear),
    ('linear_m', BotLinearM),
    ('phases_1', BotPhases1),
    ('quadratic', BotQuadratic),
    ('random', BotRandom),
    ('random_1', BotRandom1),
    ('random_5', BotRandom5),
    ('random_n', BotRandomN),
    ('simi', BotSimi),
    ('states_1', BotStates1)
))
