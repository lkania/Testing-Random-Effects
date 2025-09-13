from functools import partial
from jax import random, numpy as np, jit
import pandas as pd
from jaxopt import Bisection
from tqdm import tqdm


# cannot be jit due to if condition depending on input
def evaluate_ci(x,
				t,
				n,
				test_statistic,
				threshold,
				tol,
				maxit,
				p0s):
	if isinstance(t, int):
		counts = x
		props = counts / n
		start_p0 = np.sum(props * (np.arange(t + 1) / t))
	else:
		start_p0 = np.mean(x / t).item()

	optimality_fun = lambda p0: np.max(
		test_statistic(x=x, t=t, n=n, p0=p0) - threshold(p0=p0))

	# starting point should not be rejected
	opt = optimality_fun(start_p0)

	# For the method to work properly, we need to have a starting point
	# at which opt < 0
	if opt >= 0:
		# global search
		min_opt = opt
		zero_opt = []
		for p0 in p0s:
			opt = optimality_fun(p0)
			min_opt = np.minimum(min_opt, opt)
			if opt < 0:
				start_p0 = p0
				break
			if opt == 0:
				zero_opt.append(p0)
		if min_opt > 0:
			# reject all null hypotheses
			return 0, 0
		if min_opt == 0:
			# return interval containing non-rejected hypotheses
			zero_opt = np.array(zero_opt)
			return np.min(zero_opt), np.max(zero_opt)

	if optimality_fun(0) > 0:
		# p0 = 0 is rejected
		lower = Bisection(
			optimality_fun=optimality_fun,
			tol=tol,
			jit=True,
			maxiter=maxit,
			lower=0,
			upper=start_p0).run().params
	else:
		# p0 = 0 is not rejected
		lower = 0

	if optimality_fun(1) > 0:
		# p0 = 1 is rejected
		upper = Bisection(
			optimality_fun=optimality_fun,
			tol=tol,
			jit=True,
			maxiter=maxit,
			lower=start_p0,
			upper=1).run().params
	else:
		# p0 = 1 is not rejected
		upper = 1

	return lower, upper


def evaluate_cis(key, alpha, x, t, n, tests, names, tol, eps, maxit, B, p0s):
	assert len(tests) == len(names)

	cis = pd.DataFrame(columns=['test',
								'lower',
								'upper',
								'width'])
	cis['lower'] = cis['lower'].astype(np.float64)
	cis['upper'] = cis['upper'].astype(np.float64)
	cis['width'] = cis['width'].astype(np.float64)
	cis['test'] = cis['test'].astype('category')

	keys_tests = random.split(key, num=len(tests))

	for idx, test in enumerate(tqdm(tests)):
		tqdm.write('\n Test {} \n'.format(names[idx]))

		lower, upper = evaluate_ci(key=keys_tests[idx],
								   alpha=alpha,
								   x=x,
								   t=t,
								   n=n,
								   test=test,
								   tol=tol,
								   eps=eps,
								   maxit=maxit,
								   B=B,
								   p0s=p0s)

		cis = pd.concat([cis,
						 pd.DataFrame(data={'test': names[idx],
											'lower': lower,
											'upper': upper,
											'width': upper - lower},
									  index=[0])])

		tqdm.write(str(cis.sort_values(by='width')))

	return cis