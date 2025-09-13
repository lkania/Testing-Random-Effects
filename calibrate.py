from functools import partial
import numpy as onp
from jax import random, numpy as np, pmap, jit, vmap

import bernstein
import distribution


@partial(jit, static_argnames=['t'])
def __simulate_t(key, ps, t):
	# ps is n-dimentional array of probabilities
	xs = np.int64(random.binomial(key=key, n=t, p=ps))
	vbincount = vmap(lambda x: np.bincount(x, length=t + 1),
					 in_axes=1,
					 out_axes=0)
	counts = vbincount(xs)
	return counts


@jit
def __simulate_ts(key, ps, ts):
	# ps is n-dimentional array of probabilities
	xs = np.int64(random.binomial(key=key,
								  n=np.expand_dims(ts, axis=1),
								  p=ps,
								  shape=(ts.shape[0],
										 ps.shape[1])))
	return xs


def __simulate(key, ps, t, ):
	if isinstance(t, int):
		return __simulate_t(key=key, ps=ps, t=t)
	else:
		return __simulate_ts(key=key, ps=ps, ts=t)


# @partial(jit, static_argnames=['t', 'debug'])
def _simulate(key, dist, n, t, B, debug=False):
	key0, key1 = random.split(key, num=2)
	ps = dist.sample(key=key0, shape=(n, B))

	if debug:
		assert not np.isnan(ps).any()
		assert np.all(ps >= 0)
		assert np.all(ps <= 1)

	return __simulate(key=key1, ps=ps, t=t)


def __evaluate_t(key, ps, n, t, T):
	counts = __simulate_t(key=key, ps=ps, t=t)
	vt = vmap(lambda count: T(x=count, n=n, t=t), in_axes=0)
	Ts = vt(counts)
	return Ts


def __evaluate_ts(key, ps, ts, n, T):
	xs = __simulate_ts(key=key, ps=ps, ts=ts)
	vt = vmap(lambda xs_: T(x=xs_, t=ts, n=n), in_axes=0)
	Ts = vt(xs.T)
	return Ts


def __evaluate(key, ps, t, n, T):
	if isinstance(t, int):
		Ts = __evaluate_t(key=key, ps=ps, n=n, t=t, T=T)
	else:
		Ts = __evaluate_ts(key=key, ps=ps, n=n, ts=t, T=T)
	return Ts


def _evaluate(key, dist, n, t, B, T, debug=False):
	key0, key1 = random.split(key, num=2)
	ps = dist.sample(key=key0, shape=(n, B))

	Ts = __evaluate(key=key1, ps=ps, t=t, n=n, T=T)

	if debug:
		assert not np.isnan(Ts).any()

	return Ts


def expected_fingerprint(key, dist, B, t):
	if isinstance(dist, distribution.PointMass):
		return bernstein.Bernstein(p=dist.p, j=np.arange(t + 1), t=t)

	counts = _simulate(key=key, dist=dist, n=1, t=t, B=B)
	return np.mean(counts, axis=0)


def quantile(key, dist, n, t, T, B, alpha, debug=False):
	Ts = _evaluate(key=key, dist=dist, n=n, t=t, B=B, T=T, debug=debug)
	talpha = np.quantile(Ts, q=1 - alpha, axis=0)
	return talpha


def power(key, dist, n, t, B, T, talpha):
	Ts = _evaluate(key=key, dist=dist, n=n, t=t, B=B, T=T)

	assert Ts.shape[0] == B

	# for 1D test statistics
	if Ts.ndim == 1:
		assert talpha.ndim == 0
		return np.mean(Ts > talpha)

	# for multi-dimensional test statistics
	assert Ts.shape[1] == talpha.shape[0]
	return np.mean(np.any(Ts > talpha.reshape(1, -1), axis=1))


# the function assume that distributions are ordered by increasing w1 distance
def powers(key, dists, n, t, B, T, talpha, eps):
	n_dist = len(dists)
	keys = random.split(key, num=n_dist)

	powers_ = onp.ones(n_dist, dtype=np.float64)
	for dist_idx, dist in enumerate(dists):
		powers_[dist_idx] = power(key=keys[dist_idx],
								  dist=dist,
								  n=n,
								  t=t,
								  B=B,
								  T=T,
								  talpha=talpha)

		# stop if you reach power 1
		if 1 - powers_[dist_idx] < eps:
			return np.array(powers_)

	return np.array(powers_)