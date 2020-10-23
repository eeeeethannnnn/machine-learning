import numpy as np
import numpy.matlib


def cluster(T, K, num_iters, epsilon=1e-12):
	"""
	:param bow:
		bag-of-word matrix of (num_doc, V), where V is the vocabulary size
	:param K:
		number of topics
	:return:
		idx of size (num_doc), idx should be 1, 2, 3 or 4
	"""
	# 400 * 100
	row = T.shape[0]  # document
	col = T.shape[1]  # words
	matrix = T
	pointer = 0  # for iterations
	gamma = np.empty((row, K))  # γ
	ic = np.empty((row, K))  # γ _ic
	pi_c = np.array([[0.25, 0.25, 0.25, 0.25]])  # π_c
	
	# random 0 ~ 1
	mu = np.random.rand(col, K)  # μ
	mu_jc = mu / np.kron(np.ones((col, 1)), sum(mu))  # μ_jc
	
	# EM algorithm start
	while pointer < num_iters:
		
		# expectation
		for i in range(row):
			tmp = np.prod(mu_jc ** np.matlib.repmat(matrix[i, :], K, 1).conj().transpose(), axis=0)
			tmp = np.expand_dims(tmp, axis=0)
			gamma[i, ] = pi_c * tmp
			s = gamma[i, ].sum(axis=0)
			ic[i, ] = gamma[i, ] / s
			
		# max
		mu = np.matmul(ic.conj().transpose(), matrix).conj().transpose()
		mu_jc = mu / np.kron(np.ones((col, 1)), sum(mu))
		pi_c = np.sum(ic) / row
		pointer += 1
		
	# collect index of maximum number in each row
	idx = []
	for i in range(ic.shape[0]):
		rowMax = numpy.argmax(ic[i, ])
		idx.append(rowMax)
		
	# raise NotImplementedError
	return idx
