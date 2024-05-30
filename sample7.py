import numpy as np
import numpy.linalg as nlg
import matplotlib.pyplot as plt
import seaborn as sns
from numba import jit
from simulate import *
from functools import *
from copy import *
import pickle
import os
import shutil
import time
from datetime import datetime
from pytz import timezone
from polyagamma import random_polyagamma
from scipy.stats import norm, truncnorm
from A_init import *
from numba import jit


def timeit(func):
	def wrapper(*args, **kwargs):
		t0 = time.time()
		result = func(*args,  **kwargs)
		dt = time.time() - t0
		print("{} uses {} seconds.".format(str(func)[:str(func).find(" at ")], np.round(dt, 4)))
		return result
	return func


def random_seed_generator(seed0):
	np.random.seed(seed0)
	seed_arr, i_seed = np.random.randint(0, 1e8, 10000), -1
	while 1:
		i_seed += 1
		if i_seed == 9999:
			np.random.seed(seed_arr[-1])
			seed_arr, i_seed = np.random.randint(0, 1e8, 10000), 0
		yield seed_arr[i_seed]


class Parameter():
	def __init__(self):
		pass

	def __str__(self):
		return str(self.__dict__)


@jit(nopython = True)
def update_Z_kappa_k(pk1, pk, N, Xk1, Xk, Ak, Zk, kappak, mask):
	l = 0
	for n in range(N):
		for i in range(pk):
			for j in range(i):
				if mask[i, j] == 0:
					continue
				kappak[l] = Xk[n, i, j] - 1 / 2
				l_ = 1
				for i_ in range(pk1):
					for j_ in range(i_ + 1):
						if j_ == i_:
							Zk[l, l_] = Ak[i, i_] * Ak[j, j_] * Xk1[n, i_, j_]
						else:
							Zk[l, l_] = Ak[i, i_] * Ak[j, j_] * Xk1[n, i_, j_] + Ak[i, j_] * Ak[j, i_] * Xk1[n, i_, j_]
						l_ += 1
				l += 1


@jit(nopython = True)
def encode_am(mat, p):
	m = 2 ** (p * (p - 1) / 2 - 1)
	val = 0
	for i in range(p):
		for j in range(i):
			val += mat[i, j] * m
			m /= 2
	return val


@jit(nopython = True)
def decode_am(val, p):
	mat = np.identity(p)
	for i in range(p - 1, -1, -1):
		for j in range(i - 1, -1, -1):
			mat[i, j] = val % 2
			mat[j, i] = mat[i, j]
			val //= 2
	return mat


@jit(nopython = True)
def count_X0(p0, X0_arr):
	count_arr = np.zeros(int(2 ** (p0 * (p0 - 1) / 2)))
	for n in range(X0_arr.shape[0]):
		count_arr[int(encode_am(X0_arr[n], p0))] += 1
	return count_arr


@jit(nopython = True)
def log_posterior_Xn0(p1, C1, Xn0_Gamma1, A1, Xn1, mask):
	log_post = 0
	for i in range(p1):
		for j in range(i):
			if mask[i, j] < 0.5:
				continue
			inner_prod = C1
			for p, e in enumerate(A1[i]):
				if e:
					for q, f in enumerate(A1[j]):
						if f:
							inner_prod += Xn0_Gamma1[p, q]
			log_post += Xn1[i, j] * inner_prod - np.log(1 + np.exp(inner_prod))
	return log_post


@jit(nopython = True)
def log_posterior_Xnkij(pk2, Ck1, Ck2, Xnk0_Gammak1, Xnk1_Gammak2, Ak1, Ak2, Xnk1, Xnk2, i, j, mask):
	inner_prod = Ck1
	for p, e in enumerate(Ak1[i]):
		if e:
			for q, f in enumerate(Ak1[j]):
				if f:
					inner_prod += Xnk0_Gammak1[p, q]
	log_post = Xnk1[i, j] * inner_prod - np.log(1 + np.exp(inner_prod))
	for i_ in range(pk2):
		for j_ in range(i_):
			if mask[i_, j_] < 0.5:
				continue
			inner_prod = Ck2
			for p, e in enumerate(Ak2[i_]):
				if e:
					for q, f in enumerate(Ak2[j_]):
						if f:
							inner_prod += Xnk1_Gammak2[p, q]
			log_post += Xnk2[i_, j_] * inner_prod - np.log(1 + np.exp(inner_prod))
	return log_post


@jit(nopython = True)
def sample_X_Gibbs(p, nu, C, Gamma, A, X, alpha, seed, mask):
	np.random.seed(seed)	
	for k in range(p.size - 1):
		mask_k = np.ones((p[k + 1], p[k + 1])) if k < p.size - 2 else mask
		if k == 0:
			p0, p1 = p[0], p[1]
			C1 = C[0]
			Gamma1 = Gamma[:(p0 ** 2)].copy().reshape(p0, p0)
			A1 = A[:(p0 * p1)].copy().reshape(p1, p0)
			iX1, iX2, iX3 = 0, p0 ** 2, p0 ** 2 + p1 ** 2
			X0, X1 = X[:, iX1:iX2].copy().reshape(-1, p0, p0), X[:, iX2:iX3].copy().reshape(-1, p1, p1)
			n_count = int(2 ** (p0 * (p0 - 1) / 2))
			for n in range(X.shape[0]):
				log_Xn0_post = np.zeros(n_count)
				for val in range(n_count):
					Xn0 = decode_am(val, p0)
					log_Xn0_post[val] = np.log(nu[val]) + log_posterior_Xn0(p1, C1, Xn0 * Gamma1, A1, X1[n], mask_k)
				log_Xn0_post -= log_Xn0_post.max()
				Xn0_post = np.exp(log_Xn0_post)
				Xn0_post_cum = np.cumsum(Xn0_post / Xn0_post.sum())
				r = np.random.rand()
				for val in range(n_count):
					if Xn0_post_cum[val] > r:
						break
				Xn0 = decode_am(val, p0)
				X[n, iX1:iX2] = Xn0.reshape(1, -1).copy()
		else:
			pk0, pk1, pk2 = p[k - 1], p[k], p[k + 1]
			Ck1, Ck2 = C[k - 1], C[k]
			iG1, iG2, iG3 = np.sum(p[:(k - 1)] ** 2), np.sum(p[:k] ** 2), np.sum(p[:(k + 1)] ** 2)
			Gammak1, Gammak2 = Gamma[iG1:iG2].copy().reshape(pk0, pk0), Gamma[iG2:iG3].copy().reshape(pk1, pk1)
			iA1, iA2, iA3 = np.sum(p[:(k - 1)] * p[1:k]), np.sum(p[:k] * p[1:(k + 1)]), np.sum(p[:(k + 1)] * p[1:(k + 2)])
			Ak1, Ak2 = A[iA1:iA2].copy().reshape(pk1, pk0), A[iA2:iA3].copy().reshape(pk2, pk1)
			iX1, iX2, iX3, iX4 = np.sum(p[:(k - 1)] ** 2), np.sum(p[:k] ** 2), np.sum(p[:(k + 1)] ** 2), np.sum(p[:(k + 2)] ** 2)
			Xk0, Xk1, Xk2 = X[:, iX1:iX2].copy().reshape(-1, pk0, pk0), X[:, iX2:iX3].copy().reshape(-1, pk1, pk1), X[:, iX3:iX4].copy().reshape(-1, pk2, pk2)
			Xk0_Gammak1, Xk1_Gammak2 = Xk0 * Gammak1.reshape(1, pk0, pk0), Xk1 * Gammak2.reshape(1, pk1, pk1)
			for i in range(p[k]):
				for j in range(i):
					for n in range(X.shape[0]):
						log_Xnkij_post = np.zeros(2)
						for Xnkij in range(2):
							Xk1[n, i, j], Xk1[n, j, i], Xk1_Gammak2[n, i, j], Xk1_Gammak2[n, j, i] = Xnkij, Xnkij, Xnkij * Gammak2[i, j], Xnkij * Gammak2[i, j]
							log_Xnkij_post[Xnkij] = log_posterior_Xnkij(pk2, Ck1, Ck2, Xk0_Gammak1[n], Xk1_Gammak2[n], Ak1, Ak2, Xk1[n], Xk2[n], i, j, mask_k)
						log_Xnkij_post -= np.max(log_Xnkij_post)
						Xnkij_lik = np.exp(log_Xnkij_post * alpha)
						Xnkij_lik /= np.sum(Xnkij_lik)
						if np.random.rand() <= Xnkij_lik[0]:
							Xk1[n, i, j], Xk1[n, j, i], Xk1_Gammak2[n, i, j], Xk1_Gammak2[n, j, i] = 0, 0, 0, 0
						else:
							Xk1[n, i, j], Xk1[n, j, i], Xk1_Gammak2[n, i, j], Xk1_Gammak2[n, j, i] = 1, 1, Gammak2[i, j], Gammak2[i, j]
			X[:, iX2:iX3] = Xk1.reshape(-1, pk1 ** 2)


@jit(nopython = True)
def count(n, max_ones):
	count, count_sum = 1, 0
	for i in range(max_ones):
		count *= (n - i) / (i + 1)
		count_sum += count
	return int(count_sum)


@jit(nopython = True)
def encode(arr, n, max_ones):
	ones_arr = np.arange(1, max_ones + 1)
	count_arr = np.ones(ones_arr.size)
	count = 1
	for i, ones in enumerate(ones_arr):
		count *= (n - i) / (i + 1)
		count_arr[i] = count
	n_ones = int(arr.sum())
	val = count_arr[:(n_ones - 1)].sum()
	comb = 1
	for i in range(n_ones - 1):
		comb *= (n - 1 - i) / (i + 1)
	n_ones_left = n_ones
	for i, a in enumerate(arr[:-1]):
		if a:
			comb *= (n_ones_left - 1) / (n - i - 1)
			n_ones_left -= 1
		else:
			val += comb
			comb *= (n - i - n_ones_left) / (n - i - 1)
	return val


@jit(nopython = True)
def decode(val, n, max_ones):
	ones_arr = np.arange(1, max_ones + 1)
	count_arr = np.ones(ones_arr.size)
	count = 1
	for i, ones in enumerate(ones_arr):
		count *= (n - i) / (i + 1)
		count_arr[i] = count
	for n_ones in range(ones_arr.size):
		if val < count_arr[n_ones]:
			break
		val -= count_arr[n_ones]
	n_ones += 1
	comb = 1
	for i in range(n_ones - 1):
		comb *= (n - 1 - i) / (i + 1)
	n_ones_left = n_ones
	arr = np.zeros(n)
	for i in range(n - 1):
		if val < comb:
			arr[i] = 1
			comb *= (n_ones_left - 1) / (n - i - 1)
			n_ones_left -= 1
		else:
			val -= comb
			comb *= (n - i - n_ones_left) / (n - i - 1)
	arr[-1] = n_ones_left
	return arr


@jit(nopython = True)
def log_posterior_Aki(pk, Ck, Xk1_Gammak, Ak, Xk, i, mask):
	log_post = 0
	for j in range(pk):
		if mask[i, j] < 0.5:
			continue
		if j < i:
			l = i * (i - 1) // 2 + j
		elif i == j:
			continue
		else:
			l = j * (j - 1) // 2 + i
		inner_prod = np.ones(Xk1_Gammak.shape[0]) * Ck
		for p in np.where(Ak[i])[0]:
			for q in np.where(Ak[j])[0]:
				inner_prod += Xk1_Gammak[:, p, q]
		log_post += np.sum(Xk[:, i, j] * inner_prod - np.log(1 + np.exp(inner_prod)))
	return log_post


@jit(nopython = True)
def sample_Aki(pk1, pk, Ck, Gammak, Ak, Xk1, Xk, i, max_ones, force, alpha, seed, mask):
	Xk1_Gammak = Xk1 * Gammak.reshape(1, *Gammak.shape)
	np.random.seed(seed)
	n_count = count(pk1, max_ones)
	log_Ak_post = np.zeros(n_count)
	for val in range(n_count):
		arr = decode(val, pk1, max_ones)
		if (arr < force).any():
			log_Ak_post[val] = -np.inf
		else:
			Ak_ = Ak.copy()
			Ak_[i, :] = arr.copy()
			log_Ak_post[val] = log_posterior_Aki(pk, Ck, Xk1_Gammak, Ak_, Xk, i, mask) * alpha
	log_Ak_post -= np.max(log_Ak_post)
	Ak_post = np.exp(log_Ak_post)
	Ak_post_cum = np.cumsum(Ak_post / np.sum(Ak_post))
	r = np.random.rand()
	for val in range(n_count):
		if Ak_post_cum[val] > r:
			break
	arr = decode(val, pk1, max_ones)
	Ak[i, :] = arr.copy()


@jit(nopython = True)
def sample_A(p, C, Gamma, A, X, max_ones, force_ones, alpha, seed, mask):
	np.random.seed(seed)
	seed_arr, i_seed = np.random.randint(0, int(1e8), 10000), 0
	for k in range(1, p.size):
		Ck =  C[k - 1]
		iG1, iG2 = np.sum(p[:(k - 1)] ** 2), np.sum(p[:k] ** 2)
		Gammak = Gamma[iG1:iG2].reshape(p[k - 1], p[k - 1])
		iA1, iA2 = np.sum(p[1:k] * p[:(k - 1)]), np.sum(p[1:(k + 1)] * p[:k])
		Ak = A[iA1:iA2].reshape(p[k], p[k - 1])
		iX1, iX2, iX3 = np.sum(p[:(k - 1)] ** 2), np.sum(p[:k] ** 2), np.sum(p[:(k + 1)] ** 2)
		Xk1, Xk = X[:, iX1:iX2].copy().reshape(-1, p[k - 1], p[k - 1]), X[:, iX2:iX3].copy().reshape(-1, p[k], p[k])
		mask_k = np.ones((p[k], p[k])) if k < p.size - 1 else mask
		for i in range(p[k]):
			force = ((Ak.sum(axis = 0) - Ak[i]) < force_ones - 0.5).astype(np.float64)
			sample_Aki(p[k - 1], p[k], Ck, Gammak, Ak, Xk1, Xk, i, max_ones, force, alpha, seed_arr[i_seed], mask_k)
			i_seed += 1


@jit(nopython = True)
def log_likelihood_0(p0, nu, X0):
	count_arr = np.zeros(int(2 ** (p0 * (p0 - 1) / 2)))
	for n in range(X0.shape[0]):
		count_arr[int(encode_am(X0[n], p0))] += 1
	return np.sum(np.log(nu) * count_arr)


@jit(nopython = True)
def log_likelihood_k(pk1, pk, Ck, Gammak, Ak, Xk1, Xk, mask):
	log_lik = 0
	Xk1_Gammak = Xk1 * Gammak.reshape(1, pk1, pk1)
	for i in range(pk):
		for j in range(i):
			if mask[i, j] < 0.5:
				continue
			inner_prod = np.ones(Xk.shape[0]) * Ck
			for p in np.where(Ak[i])[0]:
				for q in np.where(Ak[j])[0]:
					inner_prod += Xk1_Gammak[:, p, q]
			log_lik += (Xk[:, i, j] * inner_prod - np.log(1 + np.exp(inner_prod))).sum()
	return log_lik


@jit(nopython = True)
def log_likelihood_0_each(p0, nu, X0):
	log_lik_0_arr = np.zeros(X0.shape[0])
	for n in range(X0.shape[0]):
		log_lik_0_arr[n] = np.log(nu[int(encode_am(X0[n], p0))])
	return log_lik_0_arr


@jit(nopython = True)
def log_likelihood_k_each(pk1, pk, Ck, Gammak, Ak, Xk1, Xk, mask):
	log_lik_k_arr = np.zeros(Xk.shape[0])
	Xk1_Gammak = Xk1 * Gammak.reshape(1, pk1, pk1)
	for i in range(pk):
		for j in range(i):
			if mask[i, j] < 0.5:
				continue
			inner_prod = np.ones(Xk.shape[0]) * Ck
			for p in np.where(Ak[i])[0]:
				for q in np.where(Ak[j])[0]:
					inner_prod += Xk1_Gammak[:, p, q]
			log_lik_k_arr += Xk[:, i, j] * inner_prod - np.log(1 + np.exp(inner_prod))
	return log_lik_k_arr


class Pyramid():
	def __init__(self, p, X_K, seed = None):
		self.K = len(p) - 1
		self.p = p
		self.X_K = X_K
		self.N = len(X_K)
		self.samples = []

		self.seed = np.random.randint(0, 1e8) if seed is None else seed
		self.seed_gen = random_seed_generator(self.seed)
		np.random.seed(next(self.seed_gen))

		self.A = [None] + [np.zeros((p[k], p[k - 1])) for k in range(1, self.K + 1)]
		for k in range(1, self.K + 1):
			count = np.random.multinomial(p[k] - 2 * p[k - 1], np.ones(p[k - 1]) / p[k - 1]) + 2
			arr = np.random.permutation(sum([[j] * count[j] for j in range(p[k - 1])], start = []))
			for i in range(p[k]):
				self.A[k][i, arr[i]] = 1

		self.X = []
		for k in range(self.K):
			self.X.append(np.stack([np.identity(p[k]) for _ in range(self.N)]))
			X_k_ber = np.random.binomial(1, 0.5 ** 0.5, size = self.N * p[k] ** 2).reshape(self.N, p[k], p[k])
			X_k_ber *= np.transpose(X_k_ber, (0, 2, 1))
			self.X[k].reshape(-1)[X_k_ber.reshape(-1).astype(bool)] = 1
		self.X.append(X_K)

		self.mask = np.ones((p[-1], p[-1]))

		self.theta = [None]
		for k in range(1, self.K + 1):
			randn = np.abs(np.random.randn(1 + p[k - 1] * (p[k - 1] + 1) // 2))
			self.theta.append(randn * np.array([-1] + [1] * (p[k - 1] * (p[k - 1] + 1) // 2)))

		self.nu = np.ones(2 ** (p[0] * (p[0] - 1) // 2))
		self.nu /= np.sum(self.nu)

		self.theta_min = [None] + [np.zeros(1 + pk * (pk + 1) // 2) for pk in p[:-1]]
		self.theta_max = [None] + [np.zeros(1 + pk * (pk + 1) // 2) + np.inf for pk in p[:-1]]
		for k in range(1, self.K + 1):
			self.theta_min[k][0] = -np.inf
			self.theta_max[k][0] = 0

		self.nu_prior = np.ones(2 ** (p[0] * (p[0] - 1) // 2))
		self.theta_prior_mean = [None] + [np.zeros(1 + pk * (pk + 1) // 2) for pk in p[:-1]]
		self.theta_prior_precision = [None] + [np.ones(1 + pk * (pk + 1) // 2) for pk in p[:-1]]

		self.omega = None

	@timeit
	def log_posterior(self, subset = None, A = None, X = None, nu = None, theta = None, return_val = False):
		if A is None:
			A = self.A
		if X is None:
			X = self.X
		N = X[0].shape[0]
		if subset is None:
			subset = np.arange(N)
		if nu is None:
			nu = self.nu
		if theta is None:
			theta = self.theta
		C, Gamma = self.update_C_Gamma(theta = theta)
		log_post = - 0.5 * np.sum([np.sum((theta[k] - self.theta_prior_mean[k]) ** 2 * self.theta_prior_precision[k]) for k in range(1, self.K + 1)]) + log_likelihood_0(self.p[0], nu, X[0][subset]) * N / subset.size
		for k in range(1, self.K + 1):
			mask_k = np.ones((self.p[k], self.p[k])) if k < self.K else self.mask
			log_post += log_likelihood_k(self.p[k - 1], self.p[k], C[k], Gamma[k], A[k], X[k - 1][subset], X[k][subset], mask_k) * N / subset.size
		if return_val:
			return log_post
		else:
			self.log_post = log_post

	@timeit
	def log_likelihood_each(self, subset = None, A = None, X = None, nu = None, theta = None, return_val = False):
		if A is None:
			A = self.A
		if X is None:
			X = self.X
		N = X[0].shape[0]
		if subset is None:
			subset = np.arange(N)
		if nu is None:
			nu = self.nu
		if theta is None:
			theta = self.theta
		C, Gamma = self.update_C_Gamma(theta = theta)
		log_lik_arr = log_likelihood_0_each(self.p[0], nu, X[0][subset])
		for k in range(1, self.K + 1):
			mask_k = np.ones((self.p[k], self.p[k])) if k < self.K else self.mask
			log_lik_arr += log_likelihood_k_each(self.p[k - 1], self.p[k], C[k], Gamma[k], A[k], X[k - 1][subset], X[k][subset], mask_k)
		if return_val:
			return log_lik_arr
		else:
			self.log_lik_arr = log_lik_arr

	@timeit
	def sample_A(self, subset, max_ones = 2, force_ones = 2, alpha = 1):
		p_arr = np.array(self.p)
		C_arr = np.array(self.C[1:])
		Gamma_arr = np.hstack([Gammak.reshape(-1) for Gammak in self.Gamma[1:]])
		A_arr = np.hstack([Ak.reshape(-1) for Ak in self.A[1:]])
		X_arr = np.hstack([Xk.reshape(self.N, -1) for Xk in self.X])
		X_arr_subset = X_arr[subset, :]
		sample_A(p_arr, C_arr, Gamma_arr, A_arr, X_arr_subset, max_ones, force_ones, alpha, next(self.seed_gen), self.mask)
		self.A = [None] + [A_arr[np.sum(p_arr[1:k] * p_arr[:(k - 1)]):np.sum(p_arr[1:(k + 1)] * p_arr[:k])].reshape(p_arr[k], p_arr[k - 1]) for k in range(1, self.K + 1)]

	@timeit
	def sample_omega(self, subset, alpha = 1):
		self.omega = [None]
		for k in range(1, self.K + 1):
			self.omega.append(random_polyagamma(alpha, (self.Z[k] @ self.theta[k]).flatten(), random_state = next(self.seed_gen)))

	@timeit
	def sample_nu(self, subset, alpha = 1):
		np.random.seed(next(self.seed_gen))
		nu_post = self.nu_prior + count_X0(self.p[0], self.X[0][subset]) * self.N / subset.size * alpha
		nu = np.random.dirichlet(nu_post)
		self.nu = deepcopy(nu)

	@timeit
	def sample_theta(self, subset, alpha = 1):
		np.random.seed(next(self.seed_gen))
		theta = [None]
		for k in range(1, self.K + 1):
			theta_k = self.theta[k].copy()
			WZTheta_k = self.omega[k] * (self.Z[k] @ theta_k)
			for l in range(theta_k.size):
				u = np.random.rand()
				if l == 0:
					var = 1 / (self.theta_prior_precision[k][l] + np.sum(self.omega[k]))
					mean = var * (self.theta_prior_precision[k][l] * self.theta_prior_mean[k][l] + alpha * np.sum(self.kappa[k]) - np.sum(WZTheta_k - self.omega[k] * self.Z[k][:, 0] * theta_k[0]))
				else:
					var = 1 / (self.theta_prior_precision[k][l] + np.sum(self.Z[k][:, l] ** 2 * self.omega[k]))
					mean = var * (self.theta_prior_precision[k][l] * self.theta_prior_mean[k][l] + np.dot(self.Z[k][:, l], alpha * self.kappa[k] - (WZTheta_k - self.omega[k] * self.Z[k][:, l] * theta_k[l])))
				rescaled_a, rescaled_b = (self.theta_min[k][l] - mean) / var ** 0.5, (self.theta_max[k][l] - mean) / var ** 0.5
				if rescaled_b >= -6 and rescaled_a <= 6:
					u = np.random.rand()
					Fa, Fb = norm.cdf(rescaled_a), norm.cdf(rescaled_b)
					theta_k[l] = mean + var ** 0.5 * norm.ppf(Fa + u * (Fb - Fa))
				else:
					theta_k[l] = truncnorm(loc = mean, scale = var ** 0.5, a = rescaled_a, b = rescaled_b).rvs(random_state = next(self.seed_gen))
				WZTheta_k += self.omega[k] * self.Z[k][:, l] * (theta_k[l] - self.theta[k][l])
			theta.append(theta_k)
		self.theta = deepcopy(theta)

	@timeit
	def sample_X(self, subset, alpha = 1):
		p_arr = np.array(self.p)
		nu_arr = self.nu
		C_arr = np.array(self.C[1:])
		Gamma_arr = np.hstack([Gammak.reshape(-1) for Gammak in self.Gamma[1:]])
		A_arr = np.hstack([Ak.reshape(-1) for Ak in self.A[1:]])
		X_arr = np.hstack([Xk.reshape(self.N, -1) for Xk in self.X])
		X_arr_subset = X_arr[subset, :]
		sample_X_Gibbs(p_arr, nu_arr, C_arr, Gamma_arr, A_arr, X_arr_subset, alpha, next(self.seed_gen), self.mask)
		X_arr[subset, :] = X_arr_subset
		for k in range(self.K + 1):
			self.X[k] = X_arr[:, np.sum(p_arr[:k] ** 2):np.sum(p_arr[:(k + 1)] ** 2)].reshape(self.N, self.p[k], self.p[k])

	@timeit
	def save_param(self, subset, save_X_mean = True, save_X_subset = True, log_lik_each = False, skip_X_K = True):
		param = Parameter()
		param.A = deepcopy(self.A)
		if save_X_mean:
			param.X_mean = [np.mean(Xk[subset], axis = 0) for Xk in (self.X[:-1] if skip_X_K else self.X)]
		if save_X_subset:
			param.X_subset = deepcopy([Xk[subset] for Xk in (self.X[:-1] if skip_X_K else self.X)])
		if log_lik_each:
			param.log_lik_arr = deepcopy(self.log_lik_arr)
		param.nu = deepcopy(self.nu)
		param.theta = deepcopy(self.theta)
		param.omega = deepcopy(self.omega)
		param.log_post = self.log_post
		param.subset = subset.copy()
		self.samples.append(param)

	@timeit
	def update_C_Gamma(self, theta = None):
		if theta is None:
			theta = self.theta
			save = True
		else:
			save = False
		C = [None]
		Gamma = [None]
		for k in range(1, self.K + 1):
			C.append(theta[k][0])
			Gamma_k = np.zeros((self.p[k - 1], self.p[k - 1]))
			l = 1
			for i in range(self.p[k - 1]):
				for j in range(i + 1):
					Gamma_k[i][j] = theta[k][l]
					Gamma_k[j][i] = Gamma_k[i][j]
					l += 1
			Gamma.append(Gamma_k)
		if save:
			self.C = C
			self.Gamma = Gamma
		else:
			return C, Gamma

	@timeit
	def update_Z_kappa(self, subset):
		self.Z = [None]
		self.kappa = [None]
		for k in range(1, self.K + 1):
			mask_k = np.ones((self.p[k], self.p[k])) if k < self.K else self.mask
			length_k = int((np.sum(mask_k) - np.trace(mask_k)) // 2)
			Zk = np.ones((length_k * subset.size, 1 + self.p[k - 1] * (self.p[k - 1] + 1) // 2))
			kappak = np.zeros(length_k * subset.size)
			update_Z_kappa_k(self.p[k - 1], self.p[k], subset.size, self.X[k - 1][subset], self.X[k][subset], self.A[k], Zk, kappak, mask_k)
			self.Z.append(Zk)
			self.kappa.append(kappak)

	def sample(self, subset_proportion = None, no_X = False, no_nu = False, no_theta = False, no_A = False, A_max_ones = 2, A_force_ones = 2, alpha = 1, save_X_mean = True, save_X_subset = True, log_lik_each = False, skip_X_K = True):
		np.random.seed(next(self.seed_gen))
		if subset_proportion is None:
			subset = np.arange(self.N)
		else:
			subset = np.random.choice(self.N, int(subset_proportion * self.N), replace = False)
		self.update_C_Gamma()
		if not no_X:
			self.sample_X(subset, alpha)
		alpha *= (self.N / subset.size)
		self.update_Z_kappa(subset)
		self.sample_omega(subset, alpha)
		if not no_nu:
			self.sample_nu(subset, alpha)
		if not no_theta:
			self.sample_theta(subset, alpha)
		if not no_A:
			self.sample_A(subset, A_max_ones, A_force_ones, alpha)
		self.log_posterior(subset)
		if log_lik_each:
			self.log_likelihood_each(subset)
		self.save_param(subset, save_X_mean, save_X_subset, log_lik_each, skip_X_K)
		return self.samples[-1]

	def write(self, param, path):
		with open(path, "wb") as hf:
			pickle.dump(param, hf)


if __name__ == "__main__":
	K = 2
	p = [4, 16, 68]
	gamma = [None, 10.0, 10.0]
	delta = [None, 4.0, 4.0]
	C = [None, -7.0, -7.0]
	PX0 = 0.5
	N = 1000
	T = 5000
	seed_gen = random_seed_generator(1)
	Gamma = transform_Gamma(K, p, gamma, delta)
	A = simulate_A(K, p, A_row_sum_max = 2, seed = next(seed_gen))
	X = simulate_X(K, p, N, C, Gamma, A, PX0, seed = next(seed_gen))
	theta = [None]
	for k in range(1, K + 1):
		theta_k = [C[k]]
		for i in range(p[k - 1]):
			for j in range(i + 1):
				theta_k.append(delta[k] if i != j else gamma[k])
		theta.append(np.array(theta_k))

	A_init = regularize_A(initialize_A(np.mean(X[-1], axis = 0), p, mixed = True), p, max_ones = 2)

	theta_min, theta_max, theta_prior_mean, theta_prior_precision, theta_init = [None], [None], [None], [None], [None]
	for k in range(1, K + 1):
		min_k, max_k, mean_k, prec_k = [-np.inf], [-2.0], [-7.0], [0.25]
		for i in range(p[k - 1]):
			for j in range(i + 1):
				min_k.append(2.0 if i == j else 1.0)
				max_k.append(np.inf)
				mean_k.append(10.0 if i == j else 4.0)
				prec_k.append(0.25)
		theta_min.append(np.array(min_k))
		theta_max.append(np.array(max_k))
		theta_prior_mean.append(np.array(mean_k))
		theta_prior_precision.append(np.array(prec_k))
		theta_init.append(np.array([truncnorm(loc = m, scale = 1 / c ** 0.5, a = (a - m) * c ** 0.5, b = (b - m) * c ** 0.5).
			rvs(random_state = next(seed_gen)) for (m, c, a, b) in zip(mean_k, prec_k, min_k, max_k)]))

	directory = os.path.join("results", str(datetime.now(timezone('EST'))).split(".")[0].replace(":", "-").replace(" ", "-"))
	os.mkdir(directory)
	with open("sample7.py", "r") as hf:
		lines = hf.readlines()
	for i, line in enumerate(lines):
		if "main" in line: 
			lines = lines[i:]
			break
	with open(os.path.join(directory, "script.txt"), "w") as hf:
		for line in lines:
			hf.write(line)

	pyramid = Pyramid(p, X[-1], seed = next(seed_gen))
	true_dict = {"K": K, "p": p, "nu": np.ones(2 ** (p[0] * (p[0] - 1) // 2)) / 2 ** (p[0] * (p[0] - 1) // 2), "theta": theta, "N": N, "T": T, "A": A, "X": X}
	true_dict["log_post"] = pyramid.log_posterior(A = true_dict["A"], X = true_dict["X"], nu = true_dict["nu"], theta = true_dict["theta"], return_val = True)
	print("True log_posterior: {}".format(true_dict["log_posterior"]))
	pyramid.write(true_dict, os.path.join(directory, "true_dict.p"))

	pyramid.theta_min, pyramid.theta_max, pyramid.theta_prior_mean, pyramid.theta_prior_precision = theta_min, theta_max, theta_prior_mean, theta_prior_precision

	pyramid.A = deepcopy(A_init)
	pyramid.nu = deepcopy(nu)
	pyramid.theta = deepcopy([None, theta_init])
	pyramid.X = deepcopy(X)
	pyramid.A, pyramid.nu, pyramid.theta, pyramid.X = deepcopy(true_dict["A"]), deepcopy(true_dict["nu"]), deepcopy(true_dict["theta"]), deepcopy(true_dict["X"])
	init_dict = {"nu": pyramid.nu, "theta": pyramid.theta, "A": pyramid.A, "X": pyramid.X, "seed": pyramid.seed}
	pyramid.write(init_dict, os.path.join(directory, "init_dict.p"))

	for t in range(T):
		param = pyramid.sample(subset_proportion = 0.01, no_X = False, no_nu = False, no_theta = False, no_A = False, A_max_ones = 2, A_force_ones = 2, alpha = 1, save_X_mean = True, save_X_subset = True, log_lik_each = False)
		pyramid.write(param, os.path.join(directory, "iter_{}.p".format(t + 1)))
		print("Iteration {} log-posterior: {}".format(t + 1, param.log_post))

	print(f"All saved to folder {directory}.")
