from jax import numpy as np, random

import hasher


class FiniteMixture(object):

	def __init__(self, ps, weights=None):
		self.ps = np.array(ps).reshape(-1)
		if weights is None:
			n = self.ps.shape[0]
			self.weights = np.full(shape=(n,), fill_value=1 / n)
		else:
			self.weights = np.array(weights).reshape(-1)

	def moment(self, j):
		return np.sum(self.weights * np.power(self.ps, j))

	def moments(self, j):
		powers_ = np.power(self.ps.reshape(-1, 1),
						   np.arange(j + 1).reshape(1, -1))
		return np.sum(self.weights.reshape(-1, 1) * powers_, axis=0)

	def sample(self, key, shape):
		return random.choice(key=key,
							 a=self.ps,
							 p=self.weights,
							 replace=True,
							 shape=shape)

	def __hash__(self):
		return hasher.hash_(str(self.ps) + str(self.weights))


class PointMass(FiniteMixture):

	def __init__(self, p):
		# Call the constructor of the parent class
		super().__init__(ps=np.array([p]),
						 weights=np.array([1.0]))
		self.p = p

	def sample(self, key, shape):
		return np.full(shape=shape, fill_value=self.p)
