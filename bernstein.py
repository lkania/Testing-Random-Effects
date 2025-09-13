from functools import partial

from jax import numpy as np, jit
from jax._src.scipy.special import gammaln
from jax.lax import add, sub, exp
from jax.scipy.stats.beta import pdf as dbeta


# see: https://github.com/google/jax/discussions/7044
def comb(N, k):
	one = np.full(shape=k.shape, fill_value=1)
	N_plus_1 = add(N, one)
	k_plus_1 = add(k, one)
	approx_comb = exp(
		sub(gammaln(N_plus_1),
			add(gammaln(k_plus_1),
				gammaln(sub(N_plus_1, k)))))
	return np.rint(approx_comb)


def Bernstein(p, j, t):
	return comb(t, j) * np.power(p, j) * np.power(1 - p, t - j)


@partial(jit, static_argnames=['t'])
def evaluate(t, ps):
	r = np.arange(0, t + 1).reshape(1, -1)
	den = dbeta(x=ps.reshape(-1, 1), a=r + 1, b=t - r + 1)  # n x k
	den /= t + 1

	return den