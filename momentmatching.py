import numpy as np
import distribution
import pandas as pd

from tqdm import tqdm
from plotnine import ggplot, aes, geom_line, labs, theme_minimal

import wasserstein

from joblib import Memory

memory = Memory("./moment_cache_dir", verbose=0)


def chebyshev_nodes(a, b, n):
	i = np.arange(n)
	x = 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2 * i + 1) / (2 * n) * np.pi)
	return x


def derivative_at_root(roots):
	roots = roots.reshape(-1)
	pairwise_differences = roots[:, None] - roots
	diff_ = pairwise_differences + np.eye(roots.shape[0])
	prod = np.prod(diff_, axis=1)
	return prod


# note that the computations are not numerically stable
def __moment_matching_distributions(lower, upper, moments, tol):
	roots = chebyshev_nodes(lower, upper, moments + 2)
	d = derivative_at_root(roots)

	# split the nodes into two sets
	# based on the sign of the d array
	p1 = roots[d > 0]
	w1 = 1 / d[d > 0]
	w1 /= np.sum(w1)
	d1 = distribution.FiniteMixture(ps=p1, weights=w1)

	p2 = roots[d < 0]
	w2 = 1 / np.abs(d[d < 0])
	w2 /= np.sum(w2)
	d2 = distribution.FiniteMixture(ps=p2, weights=w2)

	# assert that the number of nodes in the two sets is balanced
	assert np.abs(p1.shape[0] - p2.shape[0]) <= 1

	# assert that the distributions do share their first k moments
	moment_diff = np.max(np.abs(d1.moments(moments) - d2.moments(moments)))
	assert moment_diff <= tol, "lower={}, upper={}, moments={}, tol={}".format(
		lower, upper, moments, tol)

	return d1, d2


@memory.cache
def moment_matching_distributions(deltas, n_moments, tol):
	mm_distributions = pd.DataFrame(columns=['null_dist',
											 'moments',
											 'alt_dist',
											 'w1',
											 'delta'])

	print('Generating moment-matching distributions')
	moments = np.arange(start=1, stop=n_moments + 1, step=1)

	# after 500 moments on the [0,1] interval, the w1 distance
	# between the distributions becomes numerically zero
	assert max(moments) <= 500

	for moment in tqdm(moments):
		for delta in deltas:
			null_dist, alt_dist = __moment_matching_distributions(
				lower=0.5 - delta,
				upper=0.5 + delta,
				moments=moment,
				tol=tol)
			w1 = wasserstein.w1_with_dist(null_dist, alt_dist)
			assert w1 >= tol, "w1={}, delta={} moments={}".format(w1,
																  delta,
																  moment)
			s = pd.DataFrame(data={'null_dist': null_dist,
								   'w1': w1,
								   'alt_dist': alt_dist,
								   'delta': delta,
								   'moments': moment},
							 index=[0])
			mm_distributions = pd.concat([mm_distributions, s],
										 ignore_index=True)

	mm_distributions['w1'] = mm_distributions['w1'].astype(np.float64)
	mm_distributions['delta'] = mm_distributions['delta'].astype(np.float64)
	mm_distributions['moments'] = mm_distributions['moments'].astype(np.int32)
	mm_distributions['ratio'] = mm_distributions['w1'] / (
			1 / mm_distributions['moments'])

	return mm_distributions
