import numpy as np
import numpy.linalg as nlg
from copy import deepcopy
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


def column_sort(mat):
	m, n = mat.shape
	order = np.argsort(["".join(list((1 - mat[:, j]).astype(str))) for j in range(n)])
	return mat[:, order], order


def sort_A(A):
	sorted_A = []
	order = np.arange(A[-1].shape[0])
	for Ak in A[1:][::-1]:
		sorted_Ak, order = column_sort(Ak[order, :])
		sorted_A.append(sorted_Ak)
	return [None] + sorted_A[::-1]


def sort_B_with_A(A, B):
	sorted_A, sorted_B = [], []
	order = np.arange(A[-1].shape[0])
	for Ak, Bk in zip(A[1:][::-1], B[1:][::-1]):
		sorted_Ak, new_order = column_sort(Ak[order, :])
		sorted_Bk = Bk[order, :][:, new_order]
		sorted_A.append(sorted_Ak)
		sorted_B.append(sorted_Bk)
		order = new_order
	return [None] + sorted_A[::-1], [None] + sorted_B[::-1]


def plot_A(A):
	K = len(A) - 1
	plt.figure(figsize = (3 * K, 5))
	for k in range(1, K + 1):
		plt.subplot(1, K, k)
		sns.heatmap(A[k], cmap = "rocket_r", vmin = 0, vmax = 1)


def successive_projection(R):
	n, d = R.shape
	Y = np.hstack([np.ones((n, 1)), R])
	vertices = np.zeros((d + 1, d))
	indices = np.zeros(d + 1)
	for k in range(d + 1):
		i = np.argmax(nlg.norm(Y, axis = 1))
		vertices[k, :] = R[i, :]
		indices[k] = i
		u = Y[i, :] / nlg.norm(Y[i, :])
		Y = Y @ (np.identity(d + 1) - np.outer(u, u))
	return vertices, indices.astype(int)


def sketched_vertex_hunting_membership(R):
    n_clusters = int(np.ceil(R.shape[0] // 2))
    km = KMeans(n_clusters).fit(R)
    vertices, _ = successive_projection(km.cluster_centers_)
    membership = np.concatenate([R, np.ones((R.shape[0], 1))], axis = 1) @ nlg.inv(np.concatenate([vertices, np.ones((vertices.shape[0], 1))], axis = 1))
    return membership, vertices


def find_connection_matrix(X_ave, n_communities, eps = 1e-2):
	X_ave = np.maximum(X_ave, np.ones(X_ave.shape) * 1e-4)
	eigval, eigvec = nlg.eig(X_ave)
	eig_index = np.argsort(eigval)[-n_communities:]
	score_mat = eigvec[:, eig_index[:-1]] / eigvec[:, eig_index[-1:]]
	memberships, vertices = sketched_vertex_hunting_membership(score_mat)
	lambda1, lambda2d = eigval[eig_index[-1]], eigval[eig_index[-2::-1]]
	b1 = (lambda1 + (vertices ** 2 * lambda2d.reshape(1, -1)).sum(axis = 1)) ** (-0.5)
	b1[b1 < eps] = eps
	memberships /= b1.reshape(1, -1)
	memberships /= memberships.sum(axis = 1).reshape(-1, 1)
	indices = np.zeros(n_communities, dtype = int)
	for i in range(n_communities):
		indices[i] = np.argmin(np.sum((score_mat - vertices[i].reshape(1, -1)) ** 2, axis = 1))
	return memberships, indices


def initialize_A(X_ave, p, threshold = 0.2):
	K = len(p) - 1
	A = []
	for k in range(K, 0, -1):
		memberships, indices = find_connection_matrix(X_ave, p[k - 1])
		_, order = column_sort((memberships > threshold).astype(int))
		A.append(memberships[:, order])
		X_ave = X_ave[indices[order], :][:, indices[order]]
	A.append(None)
	return A[::-1]


def mix_to_overlap(M, row_max_ones, col_min_ones, threshold = 0.2):
    M2 = np.zeros(M.shape)
    rows_done = np.zeros(M.shape[0])
    for _ in range(col_min_ones):
        cols_done = np.zeros(M.shape[1])
        for _ in range(M.shape[1]):
            index = np.argmax((M - 100 * rows_done.reshape(-1, 1) - 100 * cols_done.reshape(1, -1)).reshape(-1))
            row, col = index // M.shape[1], index % M.shape[1]
            M2[row, col] = 1
            rows_done[row] = 1
            cols_done[col] = 1
    for i in range(M.shape[0]):
        remaining = row_max_ones - (M2[i] == 1).any()
        Mi_rescaled = np.maximum(M[i], 0)
        Mi_rescaled = Mi_rescaled / Mi_rescaled.sum() * (1 - M2[i])
        one_indices = np.argsort(Mi_rescaled)[::-1][:min(remaining, int(np.sum(Mi_rescaled > threshold)))]
        M2[i, one_indices] = 1
    return M2


def regularize_A(A, p, row_max_ones, col_min_ones):
	K = len(p) - 1
	A2 = [None] + [mix_to_overlap(A[k], row_max_ones, col_min_ones) for k in range(1, K + 1)]
	return A2
