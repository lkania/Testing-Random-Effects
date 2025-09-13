from functools import partial

from jax import jit, numpy as np
from jaxopt import ProjectedGradient
from jaxopt.projection import projection_simplex

import bernstein
import calibrate
import distribution
import wasserstein


@partial(jit, static_argnames=['tol'])
def __loss(ws, data, tol):
	basis, props = data
	props = props.reshape(-1)
	ws = ws.reshape(-1)

	efingerprint = np.sum(basis * ws.reshape(-1, 1), axis=0)
	efingerprint = np.where(efingerprint <= tol,
							tol,
							efingerprint)

	loss_ = (-1) * np.sum(props * np.log(efingerprint))

	return loss_


@partial(jit, static_argnames=['t', 'm', 'tol', 'maxiter'])
def __mle(counts, n, t, m, tol, maxiter):
	props = counts / n

	ps = np.linspace(start=0, stop=1, num=m)
	ps = ps.reshape(-1)
	basis = bernstein.evaluate(t=t, ps=ps)

	ws_init = np.full(shape=(m,), fill_value=1 / m)

	pg = ProjectedGradient(
		fun=partial(__loss, tol=tol),
		verbose=False,
		acceleration=True,
		implicit_diff=False,
		tol=tol,
		maxiter=maxiter,
		jit=True,
		projection=projection_simplex)
	pg_sol = pg.run(
		init_params=ws_init,
		data=(basis, props))
	ws = pg_sol.params
	ws = ws.reshape(-1)
	ps = ps.reshape(-1)

	return ps, ws


def mle(xs, n, t, m, tol, maxiter):
	counts = np.bincount(xs, length=t + 1)
	ps, ws = __mle(counts=counts, n=n, t=t, m=m, tol=tol, maxiter=maxiter)
	return distribution.FiniteMixture(ps=ps, weights=ws)


@partial(jit, static_argnames=['t', 'm', 'tol', 'maxiter'])
def mle_w1(counts, n, t, p0, m, tol, maxiter):
	ps, ws = __mle(counts=counts, n=n, t=t, m=m, tol=tol, maxiter=maxiter)
	return np.sum(np.abs(ps - p0) * ws)


def mle_homogeneity_test(key, p0, B, t, n, alpha, m, tol, maxiter):
	# define test statistic
	T = partial(mle_w1, p0=p0, m=m, tol=tol, maxiter=maxiter)

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
def __mle_w1(counts, n, t, ps, ws, m, tol, maxiter):
	ps_, ws_ = __mle(counts=counts, n=n, t=t, m=m, tol=tol, maxiter=maxiter)
	return wasserstein.w1(
		u_values=ps_,
		u_weights=ws_,

		v_values=ps,
		v_weights=ws)


def mle_test(key, null_dist, B, t, n, m, maxiter, tol, alpha=None):
	# define test statistic
	T = partial(__mle_w1,
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
