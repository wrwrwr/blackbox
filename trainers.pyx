from collections import OrderedDict as odict

from trainer_anneal cimport Trainer as TrainerAnneal
from trainer_anneal cimport Trainer as TrainerAnnealDec
from trainer_comb cimport Trainer as TrainerComb
from trainer_comb_phases cimport Trainer as TrainerCombPhases
from trainer_local cimport Trainer as TrainerLocal
from trainer_local_dec cimport Trainer as TrainerLocalDec
from trainer_select cimport Trainer as TrainerSelect

available_trainers = odict((
    ('anneal', TrainerAnneal),
    ('anneal_dec', TrainerAnnealDec),
    ('comb', TrainerComb),
    ('comb_phases', TrainerCombPhases),
    ('local', TrainerLocal),
    ('local_dec', TrainerLocalDec),
    ('select', TrainerSelect)
))
