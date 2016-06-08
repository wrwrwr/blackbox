from trainer_base cimport BaseTrainer


cdef class Trainer(BaseTrainer):
    cdef:
        public float[:] phases
