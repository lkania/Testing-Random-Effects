import pandas as pd
from plotnine import ggplot, aes, geom_line, labs, theme_minimal, facet_wrap, \
	facet_grid, geom_hline, coord_cartesian, geom_histogram, geom_vline, \
	geom_text, scale_fill_manual, scale_color_manual, labels, \
	scale_color_discrete
import patchworklib as pw
import kravchuk
import empirical
import calibrate
import distribution
from functools import partial
from tqdm import tqdm
import jax

jax.config.update("jax_enable_x64", True)
from jax import numpy as np, random

from jaxopt import Bisection, LBFGSB
import mle
import numpy as onp
from scipy.stats import chi2


def get_pvalue(test_statistic_value,
			   threshold,
			   tol,
			   maxit,
			   upper_alpha,  # upper_alpha is a guess of the maximum P-value
			   verbose=False):
	optimality_fun = lambda alpha: np.max(
		test_statistic_value - threshold(alpha))

	lower_val = optimality_fun(0.0)
	if lower_val > 0:  # we reject for alpha=0
		return 0.0

	upper_val = optimality_fun(upper_alpha)
	if upper_val <= 0:
		upper_alpha = 1.0
		upper_val = optimality_fun(upper_alpha)
		if upper_val <= 0:
			return 1.0  # we do not reject for any alpha

	pvalue = Bisection(
		optimality_fun=optimality_fun,
		tol=tol,
		jit=False,
		maxiter=maxit,
		lower=0.0,
		upper=upper_alpha,
		verbose=verbose).run().params

	return pvalue


def get_pvalues(key, p0s, maxit, test_statistic,
				xs, ts, n, B, tol,
				upper_alpha):
	keys_p0 = random.split(key, num=len(p0s))
	Ts = onp.ones(shape=(len(p0s), B), dtype=np.float64)
	for p0_idx, p0 in enumerate(p0s):
		Ts[p0_idx, :] = calibrate._evaluate(key=keys_p0[p0_idx],
											dist=distribution.PointMass(p0),
											n=n,
											t=ts,
											B=B,
											T=test_statistic)

	value = test_statistic(x=xs, n=n, t=ts)
	quantiles = lambda alpha: np.max(np.quantile(Ts, q=1 - alpha, axis=1))

	pvalue = get_pvalue(test_statistic_value=value,
						threshold=quantiles,
						tol=tol,
						maxit=maxit,
						upper_alpha=upper_alpha)

	return pvalue