import numpy as np


def softmax(Z_data):
    exp_Z_data = np.exp(Z_data)
    # print("exp(Z_data) = ",exp_Z_data)

    sum_of_exp_Z_data = np.sum(exp_Z_data)
    # print("sum of exponentials = ", sum_of_exp_Z_data)
    prob_dist = [exp_Zi / sum_of_exp_Z_data for exp_Zi in exp_Z_data]

    return np.array(prob_dist, dtype=float)


Z_data = [1.25, 2.44, 0.78, 0.12]  # for cat, dog, tiger, none

p_cat = softmax(Z_data)[0]
print("probability of being cat = ", p_cat)
p_dog = softmax(Z_data)[1]
print("probability of being dog = ", p_dog)
p_tiger = softmax(Z_data)[2]
print("probability of being tiger = ", p_tiger)
p_none = softmax(Z_data)[3]
print("probability of being none = ", p_none)