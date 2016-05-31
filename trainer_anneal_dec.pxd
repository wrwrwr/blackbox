from trainer_base cimport BaseTrainer


cdef class Trainer(BaseTrainer):
    cdef:
        int steps
        float acceptance_ease, change_high, change_low
