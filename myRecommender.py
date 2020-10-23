import numpy as np


def my_recommender(rate_mat, lr, with_reg):
    """

    :param rate_mat:
    :param lr:
    :param with_reg:
        boolean flag, set true for using regularization and false otherwise
    :return:
    """

    # TODO pick hyperparams
    iteration = 900
    mu = 0.0003
    if with_reg:
        lambDa = 0.01
    else:
        lambDa = 0
    n_user, n_item = rate_mat.shape[0], rate_mat.shape[1]

    U = np.random.rand(n_user, lr) / lr
    V = np.random.rand(n_item, lr) / lr

    # TODO implement your code here
    e = 10
    threshold = 0.001
    for i in range(iteration):
        if e > threshold:
            tmpU = U + 2 * mu * np.matmul((rate_mat - np.matmul(U, V.conj().transpose()) * (rate_mat > 0)), V) - 2 * mu * lambDa * U
            tmpV = V + 2 * mu * np.matmul((rate_mat - np.matmul(U, V.conj().transpose()) * (rate_mat > 0)).conj().transpose(), U) - 2 * mu * lambDa * V
            U = tmpU
            V = tmpV
            e = sum(sum((np.matmul((rate_mat - np.matmul(U, V.conj().transpose()) * (rate_mat > 0)), V)) ** 2)) + lambDa * sum(sum(U ** 2)) + lambDa * sum(sum(V ** 2))
        else:
            break
    return U, V
