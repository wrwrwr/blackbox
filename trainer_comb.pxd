from trainer_base cimport BaseTrainer


cdef class Trainer(BaseTrainer):
    cdef:
        float[:] phases
