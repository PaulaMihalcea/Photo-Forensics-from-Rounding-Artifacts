import numpy as np

np.seterr(invalid='ignore')  # Suppress NaN-related warnings


# Expectation step
def expectation(blocks, c, prob_r_b_in_c1, first_iteration=False):

    # Compute correlation r
    r = np.zeros(blocks.shape[0])
    for i in range(blocks.shape[0]):
        r[i] = np.corrcoef(blocks[i].flatten(), c.flatten())[0, 1]

    # Initialize probabilities
    prob_b_in_c1 = np.ones(blocks.shape[0]) * 0.5
    prob_b_in_c2 = np.ones(blocks.shape[0]) * 0.5

    if first_iteration:  # The first iteration uses arbitrary probabilities
        prob_r_b_in_c1 = np.ones(blocks.shape[0]) * prob_r_b_in_c1
        prob_r_b_in_c2 = 1 - prob_r_b_in_c1
    else:  # Successive iterations are estimated by the correlation r
        prob_r_b_in_c1 = abs(r)
        prob_r_b_in_c2 = 1 - prob_r_b_in_c1

    # Calculate conditional probability of each block b of belonging to c1 given the correlation r
    num = prob_r_b_in_c1 * prob_b_in_c1
    den = prob_r_b_in_c1 * prob_b_in_c1 + prob_r_b_in_c2 * prob_b_in_c2

    prob_b_in_c1_r = num / den

    return prob_b_in_c1_r


# Maximization step
def maximization(blocks, prob_b_in_c1_r):
    # Calculate template c
    num = np.sum(np.array([prob_b_in_c1_r * blocks for (prob_b_in_c1_r, blocks) in zip(prob_b_in_c1_r, blocks)]), axis=0)
    den = np.sum(prob_b_in_c1_r, axis=0)

    c = num / den

    return c


# Expectation-maximization algorithm
def expectation_maximization(blocks, threshold, prob_r_b_in_c1):
    # Initialize logging array for the differences plot
    diff_history = []

    # Random initialize template c
    c = np.random.uniform(0, 1, (8, 8))

    # Initialize difference matrix
    diff = np.ones((8, 8))  # Difference between successive estimates of c is an 8x8 matrix

    # First iteration
    prob_b_in_c1_r = expectation(blocks, c, True)
    c = maximization(blocks, prob_b_in_c1_r)

    # Main EM loop
    while np.all(diff > threshold):  # Iterate E-M steps until difference is lower than threshold
        # Store last iteration's template
        c_prev = np.copy(c)

        # E step
        prob_b_in_c1_r = expectation(blocks, c, prob_r_b_in_c1)

        # M step
        c = maximization(blocks, prob_b_in_c1_r)

        # Calculate difference between successive estimates of c
        diff = abs(c - c_prev)
        diff_history.append(np.average(diff))  # Add the difference matrix' average to the difference log

    return prob_b_in_c1_r, c, diff_history
