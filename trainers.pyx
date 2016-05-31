from collections import OrderedDict as odict

from trainer_anneal cimport Trainer as TrainerAnneal
from trainer_anneal_d cimport Trainer as TrainerAnnealD
from trainer_comb cimport Trainer as TrainerComb
from trainer_comb_phases cimport Trainer as TrainerCombPhases
from trainer_local cimport Trainer as TrainerLocal
from trainer_local_d cimport Trainer as TrainerLocalD
from trainer_select cimport Trainer as TrainerSelect

available_trainers = odict((
    ('anneal', TrainerAnneal),
    ('anneal_d', TrainerAnnealD),
    ('comb', TrainerComb),
    ('comb_phases', TrainerCombPhases),
    ('local', TrainerLocal),
    ('local_d', TrainerLocalD),
    ('select', TrainerSelect)
))
