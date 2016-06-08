from trainer_base cimport BaseTrainer


cdef class Trainer(BaseTrainer):
    cdef:
        public int steps
        public float change_high, change_low

