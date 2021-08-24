import cv2
import numpy as np


def correlation_2d(x, y):

    corr = np.corrcoef(x, y)[0, 1]

    return corr


# E step
def expectation(prob_b_in_c1, prob_b_in_c2, prob_r_b_in_c1, prob_r_b_in_c2):
    num = prob_r_b_in_c1 * prob_b_in_c1
    den = prob_r_b_in_c1 * prob_b_in_c1 + prob_r_b_in_c2 * prob_b_in_c2

    prob_b_in_c1_r = num / den

    return prob_b_in_c1_r


# M step
def maximization(blocks, prob_b_in_c1_r):
    num = np.sum(prob_b_in_c1_r * blocks, axis=0)
    den = np.sum(prob_b_in_c1_r, axis=0)

    c = num / den

    return c


def expectation_maximization(blocks, threshold):
    # Random initialize template c
    c = np.random.uniform(0, 1, (8, 8))

    # Initialize probabilities
    prob_b_in_c1 = 0.5
    prob_b_in_c2 = 0.5
    prob_r_b_in_c1 = 0.5  # TODO
    prob_r_b_in_c2 = 0.5  # TODO

    # Main EM loop
    diff = 0

    while diff < threshold:
        c_prev = c

        prob_b_in_c1_r = expectation(prob_b_in_c1, prob_b_in_c2, prob_r_b_in_c1, prob_r_b_in_c2)

        c = maximization(blocks, prob_b_in_c1_r)

        diff = abs(c - c_prev)

    return c
