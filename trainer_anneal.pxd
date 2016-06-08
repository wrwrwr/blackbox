from trainer_local cimport Trainer as TrainerLocal


cdef class Trainer(TrainerLocal):
    cdef:
        public float acceptance_ease
