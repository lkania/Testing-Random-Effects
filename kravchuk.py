from functools import partial

from jax import numpy as np, random, jit
from scipy.stats import bernoulli
import calibrate
import distribution
from scipy.stats import chi2
import empirical


def kernel_variance(x, y, tx, ty):
	return x * (x - 1) / (tx * (tx - 1)) + y * (y - 1) / (
			ty * (ty - 1)) - 2 * x * y / (
			tx * ty)


def kernel_mu_m1(x, y, tx, ty):
	return x / tx + y / ty - 2 * x * y / (tx * ty)


@partial(jit, static_argnames=['t'])
def estimate_mu_m1(counts, n, t):
	counts = counts.reshape(-1)
	positions = np.arange(t + 1)
	kernel_matrix = kernel_mu_m1(x=positions.reshape(-1, 1),
								 y=positions.reshape(1, -1),
								 tx=t,
								 ty=t)
	weights = counts.reshape(-1, 1) * counts.reshape(1, -1)
	kernel_matrix = kernel_matrix * weights
	kernel_matrix *= (1 - np.eye(kernel_matrix.shape[0]))
	return np.sum(kernel_matrix) / (n * (n - 1) * 2)


def estimate_mu_m1_ts(xs, n, ts):
	kernel_matrix = kernel_mu_m1(x=xs.reshape(-1, 1),
								 y=xs.reshape(1, -1),
								 tx=ts.reshape(-1, 1),
								 ty=ts.reshape(1, -1))
	kernel_matrix *= (1 - np.eye(kernel_matrix.shape[0]))
	return np.sum(kernel_matrix) / (n * (n - 1) * 2)


@partial(jit, static_argnames=['t'])
def estimate_variance(counts, n, t):
	counts = counts.reshape(-1)
	positions = np.arange(t + 1)
	kernel_matrix = kernel_variance(x=positions.reshape(-1, 1),
									y=positions.reshape(1, -1),
									tx=t,
									ty=t)
	weights = counts.reshape(-1, 1) * counts.reshape(1, -1)
	kernel_matrix = kernel_matrix * weights
	kernel_matrix *= (1 - np.eye(kernel_matrix.shape[0]))
	return np.sum(kernel_matrix) / (n * (n - 1) * 2)


def estimate_variance_ts(xs, n, ts):
	kernel_matrix = kernel_variance(x=xs.reshape(-1, 1),
									y=xs.reshape(1, -1),
									tx=ts.reshape(-1, 1),
									ty=ts.reshape(1, -1))
	kernel_matrix *= (1 - np.eye(kernel_matrix.shape[0]))
	return np.sum(kernel_matrix) / (n * (n - 1) * 2)


@partial(jit, static_argnames=['t'])
def unbias_cochran_chi2(counts, n, t, tol):
	counts = counts.reshape(-1)
	positions = np.arange(t + 1) / t
	mean = np.sum(positions * counts) / n
	bernoulli_variance = mean * (1 - mean)
	chi2 = estimate_variance(counts=counts, n=n, t=t)
	stat_ = t * chi2 / np.maximum(bernoulli_variance, tol)
	return np.abs(stat_)


@jit
def unbias_cochran_chi2_ts(x, n, t, tol):
	xs = x
	ts = t
	mean = np.mean(xs / ts)
	bernoulli_variance = mean * (1 - mean)
	chi2 = estimate_variance_ts(xs=xs, n=n, ts=ts)
	stat_ = chi2 / np.maximum(bernoulli_variance, tol)
	return np.abs(stat_)


@partial(jit, static_argnames=['t'])
def modified_cochran_chi2(counts, n, t, tol):
	counts = counts.reshape(-1)
	positions = np.arange(t + 1) / t
	mean = np.sum(positions * counts) / n
	bernoulli_variance = mean * (1 - mean)
	chi2 = np.sum(np.square(positions - mean) * counts) / (n - 1)
	stat_ = t * chi2 / np.maximum(bernoulli_variance, tol)
	return stat_


@jit
def modified_cochran_chi2_ts(x, n, t, tol):
	xs = x
	ts = t
	mean = np.sum(xs) / np.sum(ts)
	bernoulli_variance = mean * (1 - mean)
	chi2 = np.sum(ts * np.square(mean - xs / ts)) / (n - 1)
	stat_ = chi2 / np.maximum(bernoulli_variance, tol)
	return stat_


@partial(jit, static_argnames=['t'])
def adaptive_ustat(counts, n, t, tol):
	ustat_ = estimate_variance(counts=counts, n=n, t=t)
	std_estimator = estimate_mu_m1(counts=counts, n=n, t=t)
	stat_ = t * ustat_ / np.maximum(std_estimator, tol)
	return np.abs(stat_)


@jit
def adaptive_ustat_ts(x, n, t, tol):
	xs = x
	ts = t
	ustat_ = estimate_variance_ts(xs=xs, n=n, ts=ts)
	std_estimator = estimate_mu_m1_ts(xs=xs, n=n, ts=ts)
	stat_ = ustat_ / np.maximum(std_estimator, tol)
	return np.abs(stat_)


def mu(x, t):
	return (x / t) * (1 - x / t) / (t - 1)


# 2nd normalized Kravchuk polynomial
def Kravchuk2(x, p, t):
	return np.square(x / t - p) - mu(x, t)


def Kravchuk1(x, p, t):
	return (x / t - p)


@partial(jit, static_argnames=['t'])
def SumKravchuk2(counts, efingerprint, n, t, tol):
	counts = counts.reshape(-1)
	efingerprint = efingerprint.reshape(-1)

	v = np.maximum(efingerprint * (1 - efingerprint), 1 / (t + 1))

	terms = Kravchuk2(x=counts, p=efingerprint, t=n).reshape(-1) / v
	stat_ = np.mean(terms)
	return stat_


def debiased_pearson_chi2_gof_test(key, null_dist, B, t, n,
								   tol,
								   efingerprint=None,
								   alpha=None):
	key1 = key

	# generate expected fingerprint
	if efingerprint is None:
		key0, key1 = random.split(key, num=2)
		efingerprint = calibrate.expected_fingerprint(key=key0,
													  dist=null_dist,
													  B=B,
													  t=t)

	# define test statistic
	T = partial(SumKravchuk2,
				tol=tol,
				efingerprint=efingerprint)

	if alpha is None:
		return T

	# calibrate test statistic
	talpha = calibrate.quantile(key=key1,
								dist=null_dist,
								n=n,
								t=t,
								T=T,
								B=B,
								alpha=alpha)

	return T, talpha


def debiased_pearson_chi2_homogeneity_test(key, p0, B, t, n, alpha, tol):
	return debiased_pearson_chi2_gof_test(key=key,
										  null_dist=distribution.PointMass(p0),
										  B=B,
										  t=t,
										  n=n,
										  tol=tol,
										  alpha=alpha)


@partial(jit, static_argnames=['t'])
def __gof_global_minimax_test_statistic(counts, n, t,
										null_ps,
										null_ws,
										tol,
										efingerprint):
	T1 = SumKravchuk2(counts=counts,
					  efingerprint=efingerprint,
					  tol=tol,
					  n=n,
					  t=t)

	T2 = empirical._empirical_w1(counts=counts,
								 n=n,
								 t=t,
								 ps=null_ps,
								 ws=null_ws)

	return np.array((T1, T2)).reshape(-1)


def global_minimax_gof_test(key, null_dist, B, t, n,
							tol,
							efingerprint=None,
							alpha=None):
	# generate expected fingerprint
	if efingerprint is None:
		key0, key = random.split(key, num=2)
		efingerprint = calibrate.expected_fingerprint(key=key0,
													  dist=null_dist,
													  B=B,
													  t=t)

	# define test statistic
	T = partial(__gof_global_minimax_test_statistic,
				tol=tol,
				null_ps=null_dist.ps,
				null_ws=null_dist.weights,
				efingerprint=efingerprint)

	if alpha is None:
		return T

	# calibrate test statistic
	talpha = calibrate.quantile(key=key,
								dist=null_dist,
								n=n,
								t=t,
								T=T,
								B=B,
								alpha=alpha / 2)

	return T, talpha


def gof_global_minimax_homogeneity_test(key, p0, B, t, n, alpha, tol):
	if isinstance(t, int):
		return global_minimax_gof_test(key,
									   null_dist=distribution.PointMass(p0),
									   B=B,
									   t=t,
									   n=n,
									   efingerprint=None,
									   alpha=alpha,
									   tol=tol)
	else:
		assert False, 'Not implemented'


@partial(jit, static_argnames=['t'])
def pearson_chi2_test_statistic(x, efingerprint, n, t, tol):
	counts = x
	counts = counts.reshape(-1)
	props = counts / n
	efingerprint = efingerprint.reshape(-1)
	numerator = np.square(props - efingerprint)
	denominator = np.maximum(efingerprint * (1 - efingerprint), tol)
	terms = numerator / denominator
	return np.mean(terms)


def pearson_chi2_gof_test(key,
						  null_dist, B, t, n, tol,
						  alpha=None,
						  efingerprint=None):
	key1 = key

	# generate expected fingerprint
	if efingerprint is None:
		key0, key1 = random.split(key, num=2)
		efingerprint = calibrate.expected_fingerprint(key=key0,
													  dist=null_dist,
													  B=B,
													  t=t)

	# define test statistic
	T = partial(pearson_chi2_test_statistic,
				efingerprint=efingerprint,
				tol=tol)

	if alpha is None:
		return T

	# calibrate test statistic
	talpha = calibrate.quantile(key=key1,
								dist=null_dist,
								n=n,
								t=t,
								T=T,
								B=B,
								alpha=alpha)

	return T, talpha


@jit
def __pearson_chi2_homogeneity_test_statistic_ts(x, n, t, p0):
	t = t.reshape(-1)
	x = x.reshape(-1)
	return np.mean(t * np.square(x / t - p0))


@partial(jit, static_argnames=['t'])
def __pearson_chi2_homogeneity_test_statistic(x, n, t, p0):
	counts = x
	props = (counts / n).reshape(-1)

	return np.mean(t * np.square(np.arange(t + 1) / t - p0) * props)


def pearson_chi2_homogeneity_test(key, p0, B, t, n, alpha, tol):
	if isinstance(t, int):
		T = partial(__pearson_chi2_homogeneity_test_statistic, p0=p0)
	else:
		t = t.reshape(-1)
		assert len(t) == n
		T = partial(__pearson_chi2_homogeneity_test_statistic_ts, p0=p0)

	# calibrate test statistic
	talpha = calibrate.quantile(key=key,
								dist=distribution.PointMass(p0),
								n=n,
								t=t,
								T=T,
								B=B,
								alpha=alpha)

	return T, talpha


@partial(jit, static_argnames=['t'])
def LRT_test_statistic(counts, efingerprint, n, t, tol):
	counts = counts.reshape(-1)
	props = counts / n
	t_props = np.maximum(props, tol)
	efingerprint = efingerprint.reshape(-1)
	t_efingerprint = np.maximum(efingerprint, tol)
	terms = props * (np.log(t_props) - np.log(t_efingerprint))
	# return np.abs(np.mean(terms))
	return np.mean(terms)


def LRT_gof_test(key, null_dist, B, t, n, tol, alpha=None, efingerprint=None):
	key1 = key

	# generate expected fingerprint
	if efingerprint is None:
		key0, key1 = random.split(key, num=2)
		efingerprint = calibrate.expected_fingerprint(key=key0,
													  dist=null_dist,
													  B=B,
													  t=t)

	# define test statistic
	T = partial(LRT_test_statistic,
				efingerprint=efingerprint,
				tol=tol)

	if alpha is None:
		return T

	# calibrate test statistic
	talpha = calibrate.quantile(key=key1,
								dist=null_dist,
								n=n,
								t=t,
								T=T,
								B=B,
								alpha=alpha)

	return T, talpha


@jit
def __LRT_homogeneity_test_statistic_ts(x, n, t, tol, p0):
	t = t.reshape(-1)
	x = x.reshape(-1)
	term_1 = x * np.log(
		np.maximum(x / t, tol) / np.maximum(p0, tol))
	term_2 = (t - x) * np.log(
		np.maximum(1 - x / t, tol) / np.maximum(1 - p0, tol))
	return np.abs(np.mean(term_1 + term_2))


@partial(jit, static_argnames=['t'])
def __LRT_homogeneity_test_statistic(x, n, t, tol, p0):
	counts = x
	props = counts / n
	xs = np.arange(t + 1)
	term_1 = xs * np.log(
		np.maximum(xs / t, tol) / np.maximum(p0, tol)) * props
	term_2 = (t - xs) * np.log(
		np.maximum(1 - xs / t, tol) / np.maximum(1 - p0, tol)) * props
	return np.abs(np.sum(term_1 + term_2))


def LRT_homogeneity_test(key, p0, B, t, n, alpha, tol):
	if isinstance(t, int):
		T = partial(__LRT_homogeneity_test_statistic,
					p0=p0,
					tol=tol)
	else:
		t = t.reshape(-1)
		assert len(t) == n
		T = partial(__LRT_homogeneity_test_statistic_ts,
					p0=p0,
					tol=tol)

	# calibrate test statistic
	talpha = calibrate.quantile(key=key,
								dist=distribution.PointMass(p0),
								n=n,
								t=t,
								T=T,
								B=B,
								alpha=alpha)

	return T, talpha


@partial(jit, static_argnames=['t'])
def __local_minimax_homogeneity_test_statistic(x, n, t, p0):
	counts = x
	xs = np.arange(t + 1)
	props = (counts / n).reshape(-1)

	K1 = Kravchuk1(x=xs, p=p0, t=t).reshape(-1)
	K1 = np.abs(np.sum(K1 * props))

	K2 = Kravchuk2(x=xs, p=p0, t=t).reshape(-1)
	K2 = np.mean(K2 * props)

	return np.array((K1, K2)).reshape(-1)


@jit
def __local_minimax_homogeneity_test_statistic_ts(x, n, t, p0):
	t = t.reshape(-1)
	x = x.reshape(-1)

	K1 = Kravchuk1(x=x, p=p0, t=t).reshape(-1)
	K1 = np.abs(np.mean(K1))

	K2 = Kravchuk2(x=x, p=p0, t=t).reshape(-1)
	K2 = np.mean(K2)

	return np.array((K1, K2)).reshape(-1)


def local_minimax_homogeneity_test(key, p0, B, t, n, alpha):
	if isinstance(t, int):
		T = partial(__local_minimax_homogeneity_test_statistic, p0=p0)
	else:
		t = t.reshape(-1)
		assert len(t) == n
		T = partial(__local_minimax_homogeneity_test_statistic_ts, p0=p0)

	# calibrate test statistic
	talpha = calibrate.quantile(key=key,
								dist=distribution.PointMass(p0),
								n=n,
								t=t,
								T=T,
								B=B,
								alpha=alpha / 2)

	return T, talpha


@jit
def __l2_homogeneity_test_statistic_ts(x, n, t, p0):
	t = t.reshape(-1)
	x = x.reshape(-1)
	return np.mean(np.square(x / t - p0))


@partial(jit, static_argnames=['t'])
def __l2_homogeneity_test_statistic(x, n, t, p0):
	counts = x
	props = (counts / n).reshape(-1)
	return np.sum(np.square(np.arange(t + 1) / t - p0) * props)


def l2_homogeneity_test(key, p0, B, t, n, alpha):
	if isinstance(t, int):
		T = partial(__l2_homogeneity_test_statistic, p0=p0)
	else:
		t = t.reshape(-1)
		assert len(t) == n
		T = partial(__l2_homogeneity_test_statistic_ts, p0=p0)

	# calibrate test statistic
	talpha = calibrate.quantile(key=key,
								dist=distribution.PointMass(p0),
								n=n,
								t=t,
								T=T,
								B=B,
								alpha=alpha)

	return T, talpha


@jit
def __debiased_l2_homogeneity_test_statistic_ts(x, n, t, p0):
	t = t.reshape(-1)
	x = x.reshape(-1)
	stat_ = np.mean(Kravchuk2(x=x, p=p0, t=t))
	return stat_


@partial(jit, static_argnames=['t'])
def __debiased_l2_homogeneity_test_statistic(x, n, t, p0):
	counts = x
	props = (counts / n).reshape(-1)

	K2 = Kravchuk2(x=np.arange(t + 1), p=p0, t=t).reshape(-1)
	stat_ = np.sum(K2 * props)
	return stat_


def debiased_l2_homogeneity_test(key, p0, B, t, n, alpha):
	if isinstance(t, int):
		T = partial(__debiased_l2_homogeneity_test_statistic, p0=p0)
	else:
		t = t.reshape(-1)
		assert len(t) == n
		T = partial(__debiased_l2_homogeneity_test_statistic_ts, p0=p0)

	# calibrate test statistic
	talpha = calibrate.quantile(key=key,
								dist=distribution.PointMass(p0),
								n=n,
								t=t,
								T=T,
								B=B,
								alpha=alpha)

	return T, talpha


@partial(jit, static_argnames=['t'])
def __mean_homogeneity_test_statistic(x, n, t, p0):
	counts = x
	xs = np.arange(t + 1)
	props = (counts / n).reshape(-1)

	K1 = Kravchuk1(x=xs, p=p0, t=t).reshape(-1)
	K1 = np.abs(np.sum(K1 * props))

	return K1


@jit
def __mean_homogeneity_test_statistic_ts(x, n, t, p0):
	t = t.reshape(-1)
	x = x.reshape(-1)
	return np.abs(np.mean(Kravchuk1(x=x, p=p0, t=t)))


def mean_homogeneity_test(key, p0, B, t, n, alpha):
	if isinstance(t, int):
		T = partial(__mean_homogeneity_test_statistic, p0=p0)
	else:
		t = t.reshape(-1)
		assert len(t) == n
		T = partial(__mean_homogeneity_test_statistic_ts, p0=p0)

	# calibrate test statistic
	talpha = calibrate.quantile(key=key,
								dist=distribution.PointMass(p0),
								n=n,
								t=t,
								T=T,
								B=B,
								alpha=alpha)

	return T, talpha