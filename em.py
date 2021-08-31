import cv2
import numpy as np

# TODO: delete all prints


def correlation_2d(x, y):

    corr = np.corrcoef(x, y)[0, 1]

    return corr


# Expectation
def expectation(blocks, c, first_iteration=False, prob_r_b_in_c1=0.5):

    # Compute correlation r
    r = np.zeros(blocks.shape[0])
    for i in range(blocks.shape[0]):
        #r[i] = correlation_2d(blocks[i], c)
        r[i] = np.corrcoef(blocks[i].flatten(), c.flatten())[0, 1]

    #print('c[0][0]:', c[0][0], 'r[0]:', r[0])  # TODO print buono

    # Initialize probabilities
    prob_b_in_c1 = np.ones(blocks.shape[0]) * 0.5
    prob_b_in_c2 = np.ones(blocks.shape[0]) * 0.5

    if first_iteration:
        prob_r_b_in_c1 = np.ones(blocks.shape[0]) * prob_r_b_in_c1  # TODO (C) Try multiple values
        prob_r_b_in_c2 = 1 - prob_r_b_in_c1
    else:
        prob_r_b_in_c1 = r  # TODO (C) Is abs() appropriate?
        prob_r_b_in_c2 = 1 - prob_r_b_in_c1

    #print('r:', r)
    #print('E prob_r_b_in_c1:', prob_r_b_in_c1)

    num = prob_r_b_in_c1 * prob_b_in_c1
    den = prob_r_b_in_c1 * prob_b_in_c1 + prob_r_b_in_c2 * prob_b_in_c2

    prob_b_in_c1_r = num / den
    #print('E prob_b_in_c1_r:', prob_b_in_c1_r)
    #print('prob_b_in_c1_r', prob_b_in_c1_r)
    #print('prob_b_in_c1_r num', num)
    #print('r', r)
    #print('c', c)
    #print()

    return prob_b_in_c1_r


# Maximization
def maximization(blocks, prob_b_in_c1_r):
    #num = np.sum(prob_b_in_c1_r * blocks, axis=0)
    num = np.sum(np.array([prob_b_in_c1_r * blocks for (prob_b_in_c1_r, blocks) in zip(prob_b_in_c1_r, blocks)]), axis=0)
    den = np.sum(prob_b_in_c1_r, axis=0)

    c = num / den

    return c


def expectation_maximization(blocks, threshold):
    # Initialize logging variables
    diff_log = []

    # Random initialize template c
    c = np.random.uniform(0, 1, (8, 8))

    # Main EM loop
    diff = np.ones((8, 8))  # Difference between successive estimates of c is an 8x8 matrix

    # First iteration
    prob_b_in_c1_r = expectation(blocks, c, True)
    c = maximization(blocks, prob_b_in_c1_r)
    #print('first iteration done')
    #print()

    #for i in range(0, 3):
    #print('diff:', diff)
    #print(np.all(diff <= threshold))
    while not np.all(diff < threshold):  # Iterate E-M steps until difference is lower than threshold
        #print('average diff:', np.average(diff))
        #print(np.all(diff <= threshold))
        c_prev = np.copy(c)

        # E step
        prob_b_in_c1_r = expectation(blocks, c)

        # M step
        c = maximization(blocks, prob_b_in_c1_r)
        #print('c:', c)
        #print('c_prev:', c_prev)

        diff = abs(c - c_prev)  # Update difference between successive estimates of c
        diff_log.append(np.average(diff))  # Add the difference matrix average to the difference log
        #print('diff:', diff)

    return prob_b_in_c1_r, c, diff_log
