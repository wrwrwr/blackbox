from trainer_base cimport BaseTrainer


cdef class Trainer(BaseTrainer):
    cdef:
        int steps
        float change

