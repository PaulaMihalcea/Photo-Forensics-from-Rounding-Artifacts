import cv2
import numpy as np


def correlation_2d(x, y):

    corr = np.corrcoef(x, y)[0, 1]

    return corr


# Expectation
def expectation(blocks, c):
    # Compute correlation r
    r = np.zeros(blocks.shape[0])
    for i in range(blocks.shape[0]):
        r[i] = correlation_2d(blocks[i], c)

    # Initialize probabilities
    prob_b_in_c1 = np.ones(blocks.shape[0]) * 0.5
    prob_b_in_c2 = np.ones(blocks.shape[0]) * 0.5
    prob_r_b_in_c1 = np.ones(blocks.shape[0]) * 0.5  # TODO
    prob_r_b_in_c2 = np.ones(blocks.shape[0]) * 0.5  # TODO

    num = prob_r_b_in_c1 * prob_b_in_c1
    den = prob_r_b_in_c1 * prob_b_in_c1 + prob_r_b_in_c2 * prob_b_in_c2

    prob_b_in_c1_r = num / den

    return prob_b_in_c1_r


# Maximization
def maximization(blocks, prob_b_in_c1_r):
    num = np.sum(prob_b_in_c1_r * blocks, axis=0)
    den = np.sum(prob_b_in_c1_r, axis=0)

    c = num / den

    return c


def expectation_maximization(blocks, threshold):
    # Random initialize template c
    c = np.random.uniform(0, 1, (8, 8))

    # Main EM loop
    diff = np.zeros((8, 8))  # Difference between successive estimates of c is an 8x8 matrix

    while np.all(diff < threshold):  # Iterate E-M steps until difference is lower than threshold
        c_prev = c

        # E step
        prob_b_in_c1_r = expectation(blocks, c)

        # M step
        c = maximization(blocks, prob_b_in_c1_r)

        diff = abs(c - c_prev)

    return prob_b_in_c1_r, c
