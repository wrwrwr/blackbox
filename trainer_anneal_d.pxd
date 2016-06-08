from trainer_local_d cimport Trainer as TrainerLocalD


cdef class Trainer(TrainerLocalD):
    cdef:
        public float acceptance_ease
