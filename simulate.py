import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numba import jit


def transform_Gamma(K, p, gamma, delta):
	Gamma = [None]
	for k in range(1, K + 1):
		Gammak = np.identity(p[k - 1]) * (gamma[k] - delta[k]) + np.ones((p[k - 1], p[k - 1])) * delta[k]
		Gamma.append(Gammak)
	return Gamma


def simulate_A(K, p, A_row_sum_max, seed = None):
	A = [None]
	for k in range(1, K + 1):
		Ak = np.vstack([np.identity(p[k - 1]), np.identity(p[k - 1]), np.zeros((p[k] - 2 * p[k - 1], p[k - 1]))])
		for i in range(p[k - 1], p[k - 1] * 2):
			Ak[i, np.random.choice(np.append(np.arange(i - p[k - 1]), np.arange(i - p[k - 1] + 1, p[k - 1])), np.random.randint(0, A_row_sum_max))] = 1
		for i in range(p[k - 1] * 2, p[k]):
			Ak[i, np.random.choice(np.arange(p[k - 1]), np.random.randint(1, A_row_sum_max + 1))] = 1
		A.append(Ak)
	return A


def simulate_X(K, p, N, C, Gamma, A, PX0, seed = None):
	if seed is not None:
		np.random.seed(seed)
	X = [np.stack([np.identity(p[k]) for _ in range(N)]) for k in range(K + 1)]
	for k in range(K + 1):
		PXk = (1 / (1 + np.exp(- C[k] - A[k] @ (X[k - 1] * Gamma[k]) @ A[k].T))).reshape(-1) if k > 0 else PX0
		X[k].reshape(-1)[np.random.rand(N * p[k] ** 2) < PXk ** 0.5] = 1
		X[k] *= np.transpose(X[k], (0, 2, 1))
	return X


if __name__ == "__main__":
	K = 2
	p = [4, 16, 64]
	gamma = [None, 4, 4]
	delta = [None, 1, 1]
	C = [None, -4, -4]
	PX0 = 0.5
	N = 10
	Gamma = transform_Gamma(K, p, gamma, delta)
	A = simulate_A(K, p, A_row_sum_max = 2, seed = 0)
	X = simulate_X(K, p, N, C, Gamma, A, PX0, seed = 0)
	for n in range(N):
		plt.figure(figsize = (6, 8))
		for k in range(K + 1):
			plt.subplot(K + 1, 2, 2 * k + 1)
			sns.heatmap(X[k][n], cmap = "rocket_r", vmin = 0, vmax = 1)
			if A[k] is None:
				continue
			plt.subplot(K + 1, 2, 2 * k + 2)
			sns.heatmap(A[k], cmap = "rocket_r", vmin = 0, vmax = 1)
		plt.tight_layout()
		plt.show()
