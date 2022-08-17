import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport log10, log, exp, log2

@cython.boundscheck(False)
@cython.wraparound(False)
def get_F_reward(int [:,:] drugs, int [:] len_list, int [:,:] gen_drugs, int [:] gen_len_list):
    cdef int total_num, i, j, k, len_truth, len_predict, true_positive
    cdef double P, R
    total_num = len(drugs)
    cdef double [:] reward = np.zeros(total_num, dtype=np.double)

    for i in range(total_num):
        len_truth = len_list[i]
        len_predict = gen_len_list[i]
        true_positive = 0
        for j in range(len_truth):
            for k in range(len_predict):
                if drugs[i, j] == gen_drugs[i, k]:
                    true_positive += 1
        if len_predict == 0:
            P = 0
        else:
            P = true_positive / len_predict
        R = true_positive / len_truth
        if (P + R) == 0:
            reward[i] = 0
        else:
            reward[i] = 2 * P * R / (P + R)

    return reward