from functools import partial

from jax import jit, numpy as np
from jaxopt import ProjectedGradient
from jaxopt.projection import projection_simplex

import bernstein
import calibrate
import distribution
import wasserstein


@jit
def loss(ws, data):
	basis, emoments = data
	ws = ws.reshape(-1)
	emoments = emoments.reshape(-1)

	moments = np.sum(basis * ws, axis=1).reshape(-1)
	loss_ = np.sum(np.abs(moments - emoments))

	return loss_


@partial(jit, static_argnames=['t'])
def estimate_moments(counts, n, t):
	# Unbiased estimation of moments
	props = counts / n
	arange = np.arange(t + 1)
	num = bernstein.comb(N=arange.reshape(1, -1),
						 k=arange.reshape(-1, 1))
	den = bernstein.comb(N=t, k=arange.reshape(-1, 1))
	combs = num / den

	# powers = np.power(arange.reshape(-1, 1), arange)
	emoments = np.sum(combs * props, axis=1).reshape(-1)
	return emoments


@partial(jit, static_argnames=['tol', 'maxiter', 't', 'm'])
def __mom(counts, n, t, m, tol, maxiter):
	# Unbiased estimation of moments

	emoments = estimate_moments(counts=counts, n=n, t=t)

	ps = np.linspace(start=0, stop=1, num=m)
	ps = ps.reshape(-1)
	basis = np.power(ps, np.arange(t + 1).reshape(-1, 1))

	ws_init = np.full(shape=(m,), fill_value=1 / m)

	pg = ProjectedGradient(
		fun=loss,
		verbose=False,
		acceleration=True,
		implicit_diff=False,
		tol=tol,
		maxiter=maxiter,
		jit=True,
		projection=projection_simplex)
	pg_sol = pg.run(
		init_params=ws_init,
		data=(basis, emoments))
	ws = pg_sol.params
	ws = ws.reshape(-1)
	ps = ps.reshape(-1)

	return ps, ws


def mom(xs, n, t, m, tol, maxiter):
	counts = np.bincount(xs, length=t + 1)
	ps, ws = __mom(counts=counts, n=n, t=t, m=m, tol=tol, maxiter=maxiter)
	return distribution.FiniteMixture(ps=ps, weights=ws)


@partial(jit, static_argnames=['t', 'm', 'tol', 'maxiter'])
def mom_w1_wrt_point_mass(counts, n, t, p0, m, tol, maxiter):
	ps, ws = __mom(counts=counts, n=n, t=t, m=m, tol=tol, maxiter=maxiter)
	return np.sum(np.abs(ps - p0) * ws)


def mom_homogeneity_test(key, p0, B, t, n, alpha, m, tol, maxiter):
	# define test statistic
	T = partial(mom_w1_wrt_point_mass, p0=p0, m=m, tol=tol, maxiter=maxiter)

	# calibrate test statistic
	talpha = calibrate.quantile(key=key,
								dist=distribution.PointMass(p0),
								n=n,
								t=t,
								T=T,
								B=B,
								alpha=alpha)

	return T, talpha


@partial(jit, static_argnames=['t', 'm', 'tol', 'maxiter'])
def _mom_w1(counts, n, t, ps, ws, m, tol, maxiter):
	ps_, ws_ = __mom(counts=counts, n=n, t=t, m=m, tol=tol, maxiter=maxiter)
	return wasserstein.w1(
		u_values=ps_,
		u_weights=ws_,

		v_values=ps,
		v_weights=ws)


def mom_test(key, null_dist, B, t, n, m, maxiter, tol, alpha=None):
	# define test statistic
	T = partial(_mom_w1,
				m=m,
				tol=tol,
				maxiter=maxiter,
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