import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import jax

jax.config.update("jax_enable_x64", True)
from jax import numpy as np, random

import distribution
import pandas as pd
import os.path
import kravchuk
from plotnine import ggplot, aes, geom_line, labs, theme_minimal, facet_wrap, \
	facet_grid, geom_hline
import mle
import mom
import empirical
from tqdm import tqdm
from functools import partial, reduce
import calibrate

#######################################################
# activate parallelism in CPU for JAX
# See:
# - https://github.com/google/jax/issues/3534
# - https://blackjax-devs.github.io/blackjax/examples/howto_sample_multiple_chains.html
#######################################################
import os

import multiprocessing

match multiprocessing.cpu_count():
	case 32:
		n_jobs = 25
	case 12:
		n_jobs = 10
	case 4:
		n_jobs = 2
	case 1:
		n_jobs = 1
	case _:
		n_jobs = multiprocessing.cpu_count() // 2

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
	n_jobs)


# %%


def run_lowerbounds(key,
					tests,
					names,
					store_path,
					ns,
					ts,
					B,
					alpha,
					eps,
					discretization,
					usecache,
					p0s,
					prefix,
					only_type_I_error=False,
					stop_if_type_I_not_controlled=False,
					first_moment_perturbation=True):
	assert len(names) == len(tests)

	if usecache and os.path.isfile(store_path):
		df = pd.read_csv(store_path, encoding="utf-8")
	else:

		df = pd.DataFrame(columns=['p0',
								   'power',
								   'w1',
								   'null_dist',
								   'alt_dist',
								   'typeI',
								   'n',
								   't',
								   'tag'])

		# all lower-bounds assume that p0 is less tha 0.5
		assert np.all(p0s <= 0.5)

		keys_ns = random.split(key, num=len(ns))
		for n_idx, n in enumerate(ns):

			ts_ = ts[n_idx]

			keys_ts = random.split(keys_ns[n_idx], num=len(ts_))

			for t_idx, t in enumerate(ts_):

				keys_tests = random.split(keys_ts[t_idx], num=len(tests))
				print('\n n={} t={} \n'.format(n, t))

				for test_idx, create_test in enumerate(tqdm(tests)):
					tqdm.write('Test {}'.format(names[test_idx]))

					keys_p0 = random.split(keys_tests[test_idx], num=len(p0s))

					for p0_idx, p0 in enumerate(p0s):
						null_dist = distribution.PointMass(p0)

						keys_lb = random.split(keys_p0[p0_idx], num=5)

						T, talpha = create_test(key=keys_lb[0],
												p0=p0,
												B=B,
												t=t,
												n=n,
												alpha=alpha)

						# assert that the test has correct type I error
						typeI_error = calibrate.power(key=keys_lb[1],
													  dist=null_dist,
													  n=n,
													  t=t,
													  B=B,
													  T=T,
													  talpha=talpha)

						if stop_if_type_I_not_controlled:
							assert typeI_error <= alpha + eps

						if only_type_I_error:

							df = pd.concat(
								[df,
								 pd.DataFrame(data={'p0': p0,
													'power': -1,
													'w1': -1,
													'null_dist': null_dist,
													'alt_dist': None,
													'typeI': typeI_error,
													'n': n,
													't': t,
													'test': names[test_idx],
													'tag': 'only_type_I'},
											  index=[0])])

						else:

							# if the type I error is not controlled,
							# we skip the lower-bound computation

							if first_moment_perturbation:
								# lower-bound small p_0
								range_ = np.linspace(start=p0, stop=1,
													 num=discretization)
								dists = [distribution.PointMass(p) for p in
										 range_]
								w1s = np.array([np.abs(p - p0) for p in range_])

								if typeI_error > alpha + eps:
									powers = -1
								else:
									powers = calibrate.powers(key=keys_lb[2],
															  dists=dists,
															  n=n,
															  t=t,
															  B=B,
															  T=T,
															  eps=eps,
															  talpha=talpha)

								df = pd.concat([df, pd.DataFrame({'p0': p0,
																  'power': powers,
																  'w1': w1s,
																  'null_dist': null_dist,
																  'alt_dist': dists,
																  'typeI': typeI_error,
																  'n': n,
																  't': t,
																  'test': names[
																	  test_idx],
																  'tag': 'Perturb first moment'})])

							# random effects lower-bound
							range_ = np.linspace(start=0, stop=1,
												 num=discretization)
							dists = [distribution.FiniteMixture(
								ps=np.array([p0, 1]),
								weights=np.array([(1 - p), p])) for p in range_]
							w1s = np.array([(1 - p0) * p for p in range_])

							if typeI_error > alpha + eps:
								powers = -1
							else:
								powers = calibrate.powers(key=keys_lb[3],
														  dists=dists,
														  n=n,
														  t=t,
														  B=B,
														  T=T,
														  eps=eps,
														  talpha=talpha)

							df = pd.concat([df,
											pd.DataFrame({'p0': p0,
														  'power': powers,
														  'w1': w1s,
														  'null_dist': null_dist,
														  'alt_dist': dists,
														  'typeI': typeI_error,
														  'n': n,
														  't': t,
														  'test': names[
															  test_idx],
														  'tag': 'Perturb probabilities'})])

							# medium and large p0 minimax lower-bound
							# we match the first moment of the null distribution
							range_ = np.linspace(start=0, stop=p0,
												 num=discretization)
							dists = [
								distribution.FiniteMixture(
									ps=np.array([p0 + p, p0 - p]))
								for p
								in range_]
							w1s = np.array([p for p in range_])

							if typeI_error > alpha + eps:
								powers = -1
							else:
								powers = calibrate.powers(key=keys_lb[4],
														  dists=dists,
														  n=n,
														  t=t,
														  B=B,
														  T=T,
														  eps=eps,
														  talpha=talpha)

							df = pd.concat([df,
											pd.DataFrame({'p0': p0,
														  'power': powers,
														  'w1': w1s,
														  'null_dist': null_dist,
														  'alt_dist': dists,
														  'typeI': typeI_error,
														  'n': n,
														  't': t,
														  'test': names[
															  test_idx],
														  'tag': 'Match first moment'})])

		df['p0'] = df['p0'].astype(np.float64)
		df['n'] = df['n'].astype(np.int32)
		df['t'] = df['t'].astype(np.int32)
		df['power'] = df['power'].astype(np.float64)
		df['typeI'] = df['typeI'].astype(np.float64)
		df['w1'] = df['w1'].astype(np.float64)
		df['test'] = df['test'].astype(str)
		df['tag'] = df['tag'].astype(str)
		df = df.drop(['null_dist'], axis=1)
		df = df.drop(['alt_dist'], axis=1)

		# Save results (only supports strings and numbers)
		df.to_csv(store_path, index=False)

	###################################################
	# Plot results per (t,n) tuple
	###################################################
	df_ = df.copy()
	df_['n'] = df_['n'].astype('category').cat.rename_categories(
		lambda x: 'n=' + str(x))
	df_['t'] = df_['t'].astype('category').cat.rename_categories(
		lambda x: 't=' + str(x))
	(
			ggplot(df_, aes(x='p0')) +
			geom_line(aes(y='typeI', color='test'), size=1) +
			geom_hline(yintercept=alpha, linetype='dashed', color='black',
					   size=1) +
			facet_grid('n~t', scales='fixed') +
			labs(
				x='Null hypothesis $\pi_0 = \delta_{p_0}$',
				y='Type I error',
				color='Test'
			) +
			theme_minimal()
	).save("./img/{}_homogeneity_validity.pdf".format(prefix),
		   width=10,
		   height=4)

	for n_idx, n in enumerate(ns):

		ts_ = ts[n_idx]

		for t_idx, t in enumerate(ts_):

			df_ = df[(df['n'] == n) & (df['t'] == t)]

			def min_w1(group):
				w1s = group['w1'].values
				w1s = np.sort(w1s)
				for w1 in w1s:
					d = group[group['w1'] >= w1]
					min_power = min(d['power'])
					max_typeI = max(d['typeI'])
					if (min_power >= 1 - alpha - eps) and (
							max_typeI <= alpha + eps):
						return pd.Series({'w1': w1})
				# Note: this means that you should try distributions with larger w1
				return pd.Series({'w1': -1})

			minimum_w1 = (df_).groupby(['p0', 't', 'n', 'test']).apply(
				min_w1).reset_index()
			minimum_w1['w1'] = minimum_w1['w1'].astype(np.float64)
			minimum_w1 = minimum_w1[minimum_w1['w1'] >= 0]
			if not minimum_w1.empty:
				(
						ggplot(minimum_w1[minimum_w1['n'] == n],
							   aes(x='p0')) +
						geom_line(aes(y='w1', color='test'), size=1) +
						# facet_wrap('~n', nrow=1) +
						labs(
							title='Minimum $W_1(\pi,\pi_0)$ such that power$\geq${} and type I error$\leq${}'.format(
								1 - alpha, alpha),
							x='Null hypothesis $\pi_0 = \delta_{p_0}$',
							y='$W_1$ (lower is better)',
							color='Test'
						) +
						theme_minimal()
				).save(
					"./img/{}_homogeneity_t={}_n={}_w1.pdf".format(prefix,
																   t,
																   n),
					width=8,
					height=3)

			minimum_w1 = (df_).groupby(['p0', 't', 'n', 'test', 'tag']).apply(
				min_w1).reset_index()
			minimum_w1['w1'] = minimum_w1['w1'].astype(np.float64)
			minimum_w1 = minimum_w1[minimum_w1['w1'] >= 0]
			if not minimum_w1.empty:
				(
						ggplot(minimum_w1, aes(x='p0')) +
						geom_line(aes(y='w1', color='test'), size=1) +
						facet_wrap('~tag', nrow=1) +
						labs(
							title='Minimum $W_1(\pi,\pi_0)$ such that power$\geq${} and type I error$\leq${}'.format(
								1 - alpha, alpha),
							x='Null hypothesis $\pi_0 = \delta_{p_0}$',
							y='$W_1$ (lower is better)',
							color='Test'
						) +
						theme_minimal()
				).save(
					"./img/{}_homogeneity_t={}_n={}_w1_per_lb.pdf".format(
						prefix, t, n),
					width=10,
					height=3)

			# power-plot per p0
			valid_tests = df_[df_['typeI'] <= alpha + eps]
			min_power_per_lm = \
				valid_tests.groupby(['p0', 't', 'n', 'test', 'w1', 'tag'])[
					'power'].min().reset_index()

			min_power_per_lm = min_power_per_lm[min_power_per_lm['w1'] <= 0.35]

			for p0 in [0.01, 0.1, 0.5]:
				(
						ggplot(min_power_per_lm[(min_power_per_lm['p0'] == p0)],
							   aes(x='w1')) +
						geom_line(aes(y='power', color='test'), size=1) +
						facet_wrap('~tag', nrow=1) +
						labs(
							# title='p0={}'.format(tag, round(p0, 3)),
							x='$W_1$ distance between null and alternative hypothesis',
							y='Power (higher is better)',
							color='Test'
						) +
						theme_minimal()
				).save(
					"./img/{}_homogeneity_t={}_n={}_power_per_lb_p0={}.pdf".format(
						prefix,
						t, n,
						round(p0, 3)),
					width=8, height=3)


if __name__ == "__main__":
	###################################################
	# Simulation parameters
	###################################################

	seed = 0

	ns = [5, 10, 50, 100]
	ts = []
	for n in ns:
		ts.append([2, 5, 10])
	B = 5000
	alpha = 0.05
	eps = 0.02
	maxit = 500
	tol = 1e-10
	discretization = 100
	usecache = False
	p0s = np.arange(start=0, stop=0.501, step=0.01)
	store_path = "./results/known_homogeneity.csv"
	# %%

	###################################################
	# Simulation for homogeneity testing
	###################################################
	tests = [
		partial(kravchuk.gof_global_minimax_homogeneity_test, tol=tol),
		kravchuk.local_minimax_homogeneity_test,
		empirical.plugin_homogeneity_test,
		kravchuk.mean_homogeneity_test,
		partial(kravchuk.debiased_pearson_chi2_homogeneity_test, tol=tol),
		partial(kravchuk.pearson_chi2_homogeneity_test, tol=tol),
		partial(kravchuk.LRT_homogeneity_test, tol=tol),
		partial(mle.mle_homogeneity_test,
				m=1000,
				tol=tol,
				maxiter=maxit),
		partial(mom.mom_homogeneity_test,
				m=1000,
				tol=tol,
				maxiter=maxit)
	]
	names = [
		'Global Minimax',
		'Local Minimax',
		'Plug-in',
		'Mean',
		'deb. Pearson\'s $\chi^2$',
		'mod. Pearson\'s $\chi^2$',
		'mod. LRT',
		'Distance to MLE',
		'Distance to MOM'
	]

	assert len(names) == len(tests)

	run_lowerbounds(key=random.key(seed=seed),
					tests=tests,
					names=names,
					store_path=store_path,
					ns=ns,
					ts=ts,
					B=B,
					alpha=alpha,
					eps=eps,
					discretization=discretization,
					usecache=usecache,
					p0s=p0s,
					prefix='known',
					only_type_I_error=False,
					stop_if_type_I_not_controlled=True)