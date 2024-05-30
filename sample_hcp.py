from sample7 import *
from scipy.io import loadmat


if __name__ == "__main__":
	print("Loading matlab data...")
	dic = loadmat("HCP_cortical_TensorData_desikan.mat")
	X_K = dic["loaded_bd_network"]
	X_K += np.transpose(X_K, (1, 0, 2))
	X_K = np.transpose(X_K, (2, 0, 1)) + np.identity(68).reshape(1, 68, 68)
	print("Loaded matlab data.")

	K = 2
	p = [4, 16, 68]
	T, S = 5000, 100

	directory = os.path.join("results_hcp", str(datetime.now(timezone('EST'))).split(".")[0].replace(":", "-").replace(" ", "-"))
	os.mkdir(directory)
	with open("sample_hcp.py", "r") as hf:
		lines = hf.readlines()
	for i, line in enumerate(lines):
		if "main" in line: 
			lines = lines[i:]
			break
	with open(os.path.join(directory, "script.txt"), "w") as hf:
		for line in lines:
			hf.write(line)

	pyramid = Pyramid(p, X_K, seed = 0)

	A_init = regularize_A(initialize_A(np.mean(X_K, axis = 0), p, mixed = True), p, max_ones = 2)
	pyramid.A = deepcopy(A_init)

	nu_init = np.ones(int(2 ** (p[0] * (p[0] - 1) / 2)))
	nu_init /= np.sum(nu_init)
	pyramid.nu = deepcopy(nu_init)

	theta_min, theta_max, theta_prior_mean, theta_prior_precision = [None], [None], [None], [None]
	for k in range(1, K + 1):
		min_k, max_k, mean_k, prec_k = [-np.inf], [-1.0], [-3.0], [0.25]
		for i in range(p[k - 1]):
			for j in range(i + 1):
				min_k.append(1.0 if i == j else 0.5)
				max_k.append(np.inf)
				mean_k.append(2.0 if i == j else 1.0)
				prec_k.append(0.25)
		theta_min.append(np.array(min_k))
		theta_max.append(np.array(max_k))
		theta_prior_mean.append(np.array(mean_k))
		theta_prior_precision.append(np.array(prec_k))
	pyramid.theta_min, pyramid.theta_max, pyramid.theta_prior_mean, pyramid.theta_prior_precision = theta_min, theta_max, theta_prior_mean, theta_prior_precision
	pyramid.theta = deepcopy(theta_prior_mean)

	init_dict = {"nu": pyramid.nu, "theta": pyramid.theta, "A": pyramid.A, "X": pyramid.X}
	pyramid.write(init_dict, os.path.join(directory, "init_dict.p"))
	print("Pyramid parameters initialized.")

	X_ave = X_K.mean(axis = 0)
	mask = (X_ave > 1e-6).astype(np.float64)
	pyramid.mask = mask

	for t in range(T):
		param = pyramid.sample(subset_proportion = 0.01, no_X = False, no_nu = False, no_theta = False, no_A = False, A_max_ones = 2, A_force_ones = 2, alpha = 1, save_X_mean = True, save_X_subset = True)
		pyramid.write(param, os.path.join(directory, "iter_{}.p".format(t + 1)))
		print("Iteration {} log-posterior: {}".format(t + 1, param.log_post))
	for s in range(S):
		param = pyramid.sample(subset_proportion = 1, no_X = False, no_nu = False, no_theta = True, no_A = True, alpha = 1, save_X_mean = True, save_X_subset = True)
		pyramid.write(param, os.path.join(directory, "iter_{}.p".format(t + 1)))
		print("Iteration {} log-posterior: {}".format(T + s + 1, param.log_post))