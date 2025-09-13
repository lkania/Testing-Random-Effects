import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from plotnine import ggplot, aes, geom_line, labs, theme_minimal, facet_wrap, \
	facet_grid, geom_histogram
import patchworklib as pw

import jax

jax.config.update("jax_enable_x64", True)
from jax import numpy as np, random

import calibrate
import pandas as pd
import distribution
import wasserstein
import kravchuk
import mle
import mom
import empirical
from tqdm import tqdm
import momentmatching
from functools import partial

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

###################################################
# Simulation parameters
###################################################

seed = 0
ns = [2, 5, 10, 50, 100]
ts = []
for n in ns:
	ts.append([2, 5, 10, 50, 100])
B = 5000
alpha = 0.05
maxit = 500
tol = 1e-10
discretization = 100
eps = 0.02
usecache = False
stop_if_type_I_not_controlled = True

###################################################
# Simulation for homogeneity testing
###################################################

store_path = "./results/generic.csv"
key = random.key(seed=seed)

###############################################################
# Create moment-matching distributions
###############################################################

mm_distributions = momentmatching.moment_matching_distributions(
	deltas=[0.5], n_moments=500, tol=tol)

if usecache and os.path.isfile(store_path):
	df = pd.read_csv(store_path, encoding="utf-8")
else:
	tests = [
		partial(kravchuk.global_minimax_gof_test, tol=tol),
		empirical.plugin_gof_test,
		partial(kravchuk.debiased_pearson_chi2_gof_test, tol=tol),
		partial(kravchuk.pearson_chi2_gof_test, tol=tol),
		partial(kravchuk.LRT_gof_test, tol=tol),
		partial(mom.mom_test, m=1000, tol=tol, maxiter=maxit),
		partial(mle.mle_test, m=1000, tol=tol, maxiter=maxit)
	]
	names = [
		'Global minimax',
		'Plug-in',
		'debiased Pearson\'s $\chi^2$',
		'mod. Pearson\'s $\chi^2$',
		'mod. LRT',
		'Distance to MOM',
		'Distance to MLE'
	]

	assert len(names) == len(tests)

	###############################################################
	# Start simulation
	###############################################################

	df = pd.DataFrame(
		columns=['null_dist',
				 'power',
				 'w1',
				 'alt_dist',
				 'tag',
				 'test',
				 'n',
				 't'])

	keys_ns = random.split(key, num=len(ns))
	for n_idx, n in enumerate(ns):

		ts_ = ts[n_idx]

		keys_ts = random.split(keys_ns[n_idx], num=len(ts_))

		for t_idx, t in enumerate(ts_):

			keys_tests = random.split(keys_ts[t_idx], num=len(tests))

			print('\n n={}, t={} \n'.format(n, t))
			for test_idx, create_test in enumerate(tqdm(tests)):
				tqdm.write('Test {}'.format(names[test_idx]))

				keys_lb = random.split(keys_tests[test_idx], num=5)

				null_dist = distribution.FiniteMixture(ps=np.array([0, 1]))
				T, talpha = create_test(key=keys_lb[0],
										null_dist=null_dist,
										B=B,
										t=t,
										n=n,
										alpha=alpha)

				typeI_error = calibrate.power(key=keys_lb[1],
											  dist=null_dist,
											  n=n,
											  t=t,
											  B=B,
											  T=T,
											  talpha=talpha)
				# Note: Power is only computed if the type I error is controlled
				range_ = np.linspace(start=0, stop=0.5, num=discretization)
				dists = [distribution.FiniteMixture(
					ps=np.array([0, 1]),
					weights=np.array(
						[0.5 - p, 0.5 + p])) for p in range_]
				w1s = np.array(
					[wasserstein.w1_with_dist(null_dist, d) for d in dists])

				if typeI_error > alpha + eps:
					powers = -1
					assert not stop_if_type_I_not_controlled
				else:
					powers = calibrate.powers(key=keys_lb[2],
											  dists=dists,
											  n=n,
											  t=t,
											  B=B,
											  T=T,
											  talpha=talpha,
											  eps=eps)

				df = pd.concat([df, pd.DataFrame({'null_dist': null_dist,
												  'typeI': typeI_error,
												  'power': powers,
												  'w1': w1s,
												  'alt_dist': dists,
												  'test': names[test_idx],
												  'n': n,
												  't': t,
												  'tag': 'Perturb probabilities'})])

				# moment matching lower-bound
				n_moments = discretization
				stop_ = min(t,
							500)  # after matching 500 moments w1 is numerically zero
				moments = np.arange(1, stop_ + 1)[::-1]
				# the number of moments to match decreases
				# thus, the w1 increases
				keys_lb_moments = random.split(keys_lb[3], num=len(moments))

				power = -1
				w1 = 0
				for moment_idx, moment in enumerate(moments):
					cond1 = (mm_distributions['moments'] == moment)
					cond2 = (mm_distributions['delta'] == 0.5)
					dists = mm_distributions[cond1 & cond2]

					assert dists.shape[0] == 1

					null_dist = dists['null_dist'].values[0]
					alt_dist = dists['alt_dist'].values[0]

					# check that w1 is increasing at each iteration
					assert dists['w1'].values[0] >= w1
					w1 = dists['w1'].values[0]

					keys_ = random.split(keys_lb_moments[moment_idx],
										 num=3)

					T, talpha = create_test(key=keys_[0],
											null_dist=null_dist,
											B=B,
											t=t,
											n=n,
											alpha=alpha)

					typeI_error = calibrate.power(key=keys_[1],
												  dist=null_dist,
												  n=n,
												  t=t,
												  B=B,
												  T=T,
												  talpha=talpha)

					if typeI_error > alpha + eps:
						power = -1
						assert not stop_if_type_I_not_controlled
					else:
						if 1 - power > eps:
							# we compute the power, only if power is not
							# already close to 1
							power = calibrate.power(key=keys_[2],
													dist=alt_dist,
													n=n,
													t=t,
													B=B,
													T=T,
													talpha=talpha)

					df = pd.concat(
						[df, pd.DataFrame(data={'null_dist': null_dist,
												'typeI': typeI_error,
												'power': power,
												'w1': w1,
												'alt_dist': alt_dist,
												'test': names[test_idx],
												'n': n,
												't': t,
												'tag': 'Match moments'},
										  index=[0])])

	# add moment matching lower-bound

	df['power'] = df['power'].astype(np.float64)
	df['n'] = df['n'].astype(np.int32)
	df['t'] = df['t'].astype(np.int32)
	df['typeI'] = df['typeI'].astype(np.float64)
	df['w1'] = df['w1'].astype(np.float64)
	df['test'] = df['test'].astype(str)
	df['tag'] = df['tag'].astype(str)
	df = df.drop(['null_dist'], axis=1)
	df = df.drop(['alt_dist'], axis=1)
	df.to_csv(store_path, index=False)

###################################################
# Plot momenth matching distributions
###################################################

mm_distributions_ = mm_distributions[mm_distributions['moments'] == 10]
null_dist = mm_distributions_['null_dist'].values[0]
alt_dist = mm_distributions_['alt_dist'].values[0]

key0, key1 = random.split(key, num=2)

marginals = pd.DataFrame(columns=['dist', 'x', 'weight'])
null_plugin_dist = distribution.FiniteMixture(
	ps=np.arange(10 + 1) / 10,
	weights=calibrate._simulate(key=key0,
								dist=null_dist,
								n=100,
								t=10,
								B=1) / 100)
marginals = pd.concat([marginals,
					   pd.DataFrame(
						   data={'dist': 'Null',
								 'x': null_plugin_dist.ps,
								 'weight': null_plugin_dist.weights})])

alt_plugin_dist = distribution.FiniteMixture(
	ps=np.arange(10 + 1) / 10,
	weights=calibrate._simulate(key=key0,
								dist=alt_dist,
								n=100,
								t=10,
								B=1) / 100)
marginals = pd.concat([marginals,
					   pd.DataFrame(
						   data={'dist': 'Alternative',
								 'x': alt_plugin_dist.ps,
								 'weight': alt_plugin_dist.weights})])
plot0 = (
		ggplot(marginals,
			   aes(x='x',
				   weight='weight',
				   color='dist',
				   fill='dist')) +
		geom_histogram(position='identity',
					   alpha=0.5,
					   bins=30) +
		labs(
			title='A sample from the marginal distribution',
			x='Value',
			y='Normalized counts',
			color='Distribution',
			fill='Distribution'
		) +
		theme_minimal()
)
g0 = pw.load_ggplot(plot0, figsize=(3, 2))

mm_df = pd.DataFrame(columns=['x', 'dist'])
key0, key1 = random.split(key, num=2)
mm_df = pd.concat([mm_df,
				   pd.DataFrame(
					   data={'dist': 'Null',
							 'x': null_dist.sample(key=key0,
												   shape=(5000,)).reshape(-1)},
				   )])
mm_df = pd.concat([mm_df,
				   pd.DataFrame(
					   data={'dist': 'Alternative',
							 'x': alt_dist.sample(key=key1,
												  shape=(5000,)).reshape(-1)},
				   )])
plot1 = (
		ggplot(mm_df,
			   aes(x='x', fill='dist')) +
		geom_histogram(aes(y='..density..'),
					   position='identity',
					   alpha=1,
					   bins=30,
					   show_legend=False) +
		labs(
			title='Two distributions that match 10 moments',
			x='Values',
			y='Normalized Counts',
			fill='Distribution'
		) +
		theme_minimal()
)
g1 = pw.load_ggplot(plot1, figsize=(3, 2))

plot2 = (
		ggplot(mm_distributions, aes(x='moments')) +
		geom_line(aes(y='w1'), size=1) +
		labs(
			title='$W_1$ decreases as 1/k for $k$ matched moments',
			x='Number of matched moments',
			y='$W_1$  (lower is better)'
		) +
		theme_minimal()
)

g2 = pw.load_ggplot(plot2, figsize=(3, 2))

g = (g2 | g1 | g0)
g.savefig("./img/moment_matching.pdf", dpi=600, bbox_inches='tight')

(
		ggplot(mm_distributions, aes(x='moments')) +
		geom_line(aes(y='ratio', color='factor(delta)'), size=1) +
		labs(
			title='Ratio between w1 and (1/moments matched)',
			x='Moments matched',
			y='w1/(1/moments matched)',
			color='Delta'
		) +
		theme_minimal()
).save("./img/moment_matching_ratio.pdf", width=8, height=3)

###################################################
# Plot results
###################################################

if stop_if_type_I_not_controlled:
	assert df['typeI'].max() <= alpha + eps


def min_w1(group):
	w1s = group['w1'].values
	w1s = np.sort(w1s)
	for w1 in w1s:
		d = group[group['w1'] >= w1]
		min_power = min(d['power'])
		max_typeI = max(d['typeI'])
		if (min_power >= 1 - alpha - eps) and (max_typeI <= alpha + eps):
			return pd.Series({'w1': w1})
	return pd.Series({'w1': -1})


minimum_w1 = (df).groupby(['t', 'n', 'test']).apply(min_w1).reset_index()
minimum_w1['w1'] = minimum_w1['w1'].astype(np.float64)
minimum_w1 = minimum_w1[minimum_w1['w1'] >= 0]
for n in ns:
	(
			ggplot(minimum_w1[minimum_w1['n'] == n], aes(x='t')) +
			geom_line(aes(y='w1', color='test'), size=1) +
			labs(
				title='Minimum $W_1(\pi,\pi_0)$ such that power$\geq${} and type I error $\leq${}'.format(
					1 - alpha, alpha),
				x='t (number of trials)',
				y='$W_1$ (lower is better)',
				color='Test'
			) +
			theme_minimal()
	).save("./img/identity_w1_n={}.pdf".format(n),
		   width=8,
		   height=3)

minimum_w1 = (df).groupby(['t', 'n', 'test', 'tag']).apply(min_w1).reset_index()
minimum_w1['w1'] = minimum_w1['w1'].astype(np.float64)
minimum_w1 = minimum_w1[minimum_w1['w1'] >= 0]
(
		ggplot(minimum_w1, aes(x='t')) +
		geom_line(aes(y='w1', color='test'), size=1) +
		facet_grid('tag~n', scales='free_x') +
		labs(
			title='Minimum $W_1(\pi,\pi_0)$ such that power$\geq${} and type I error $\leq${}'.format(
				1 - alpha, alpha),
			x='t (number of trials)',
			y='$W_1$ (lower is better)',
			color='Test'
		) +
		theme_minimal()
).save("./img/identity_w1_per_lb.pdf", width=10, height=6)

# selective power-plot

valid_tests = df[df['typeI'] <= alpha + eps]
min_power = \
	valid_tests.groupby(['test', 'tag', 't', 'n', 'w1'])[
		'power'].min().reset_index()
min_power['t'] = min_power['t'].astype('category').cat.rename_categories(
	lambda x: 't={}'.format(x))
min_power['n'] = min_power['n'].astype('category').cat.rename_categories(
	lambda x: 'n={}'.format(x))

for tag in set(min_power['tag']):
	(
			ggplot(min_power[min_power['tag'] == tag], aes(x='w1')) +
			geom_line(aes(y='power', color='test'), size=1) +
			facet_grid('n~t', scales='fixed') +
			labs(
				title='{}'.format(tag),
				x='$W_1(\pi,\pi_0)$ between null and alternative mixing distributions',
				y='Power (higher is better)',
				color='Test'
			) +
			theme_minimal()
	).save("./img/indentity_testing_full_{}.pdf".format(tag),
		   width=10,
		   height=10)

# selective power-plot
valid_tests = df[df['typeI'] <= alpha + eps]
min_power = \
	valid_tests.groupby(['test', 'tag', 't', 'n', 'w1'])[
		'power'].min().reset_index()
min_power = min_power[min_power['n'] == 10]
min_power = min_power[min_power['t'].isin([2, 10, 100])]
min_power['t'] = min_power['t'].astype('category').cat.rename_categories(
	lambda x: 'n={} t={}'.format(min(min_power['n']), x))

for tag in set(min_power['tag']):
	(
			ggplot(min_power[min_power['tag'] == tag], aes(x='w1')) +
			geom_line(aes(y='power', color='test'), size=1) +
			facet_wrap('~t', nrow=1) +
			labs(
				title='{}'.format(tag),
				x='$W_1(\pi,\pi_0)$ between null and alternative mixing distributions',
				y='Power (higher is better)',
				color='Test'
			) +
			theme_minimal()
	).save("./img/indentity_testing_{}.pdf".format(tag),
		   width=10,
		   height=3)