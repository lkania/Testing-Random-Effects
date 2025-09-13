from jax import jit, numpy as np


def w1_with_dist(dist1, dist2):
	return w1(u_values=dist1.ps,
			  u_weights=dist1.weights,
			  v_values=dist2.ps,
			  v_weights=dist2.weights)


# See wasserstein_distance from scipy.stats
@jit
def w1(u_values, u_weights, v_values, v_weights):
	u_sorter = np.argsort(u_values)
	v_sorter = np.argsort(v_values)

	all_values = np.concatenate((u_values, v_values))
	all_values = np.sort(all_values)

	# Compute the differences between pairs of successive values of u and v.
	deltas = np.diff(all_values)

	# Get the respective positions of the values of u and v among the values of
	# both distributions.
	u_cdf_indices = u_values[u_sorter].searchsorted(all_values[:-1], 'right')
	v_cdf_indices = v_values[v_sorter].searchsorted(all_values[:-1], 'right')

	# Calculate the CDFs of u and v using their weights, if specified.
	u_sorted_cumweights = np.concatenate((np.array([0]),
										  np.cumsum(u_weights[u_sorter])))
	u_cdf = u_sorted_cumweights[u_cdf_indices] / u_sorted_cumweights[-1]

	v_sorted_cumweights = np.concatenate((np.array([0]),
										  np.cumsum(v_weights[v_sorter])))
	v_cdf = v_sorted_cumweights[v_cdf_indices] / v_sorted_cumweights[-1]

	# Compute the value of the integral based on the CDFs.
	# If p = 1 or p = 2, we avoid using np.power, which introduces an overhead
	# of about 15%.
	# if p == 1:
	return np.sum(np.multiply(np.abs(u_cdf - v_cdf), deltas))
