from functools import partial

from jax import numpy as np, jit
import calibrate
import distribution
import wasserstein


@partial(jit, static_argnames=['t'])
def _empirical_w1(counts, n, t, ps, ws):
	return wasserstein.w1(

		u_values=np.arange(t + 1) / t,
		u_weights=counts.reshape(-1) / n,

		v_values=ps,
		v_weights=ws)


def plugin_gof_test(key, null_dist, B, t, n, alpha=None):
	# define test statistic
	T = partial(_empirical_w1,
				ps=null_dist.ps,
				ws=null_dist.weights)

	if alpha is None:
		return T

	# calibrate test statistic
	talpha = calibrate.quantile(key=key,
								dist=null_dist,
								n=n,
								t=t,
								T=T,
								B=B,
								alpha=alpha)

	return T, talpha


@partial(jit, static_argnames=['t'])
def __plugin_homogeneity_test_statistic(x, n, t, p0):
	counts = x
	counts = counts.reshape(-1)
	props = counts / n
	return np.sum(np.abs(np.arange(t + 1) / t - p0) * props)


def __plugin_homogeneity_test_statistic_ts(x, n, t, p0):
	t = t.reshape(-1)
	x = x.reshape(-1)
	return np.mean(np.abs((x / t) - p0))


def plugin_homogeneity_test(key, p0, B, t, n, alpha=None):
	# define test statistic
	if isinstance(t, int):
		T = partial(__plugin_homogeneity_test_statistic, p0=p0)
	else:
		t = t.reshape(-1)
		assert len(t) == n
		T = partial(__plugin_homogeneity_test_statistic_ts, p0=p0)

	# calibrate test statistic
	talpha = calibrate.quantile(key=key,
								dist=distribution.PointMass(p0),
								n=n,
								t=t,
								T=T,
								B=B,
								alpha=alpha)

	return T, talpha
