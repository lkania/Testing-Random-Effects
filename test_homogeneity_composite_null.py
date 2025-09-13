import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import jax

jax.config.update("jax_enable_x64", True)
from jax import numpy as np, random

from test_homogeneity_simple_null import run_lowerbounds
import distribution
import pandas as pd
import kravchuk
from scipy.stats import chi2
from plotnine import ggplot, aes, geom_line, labs, theme_minimal, facet_grid, \
	geom_hline, \
	geom_histogram
from tqdm import tqdm
from functools import partial
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
tol = 1e-10
discretization = 100
usecache = False
p0s = np.arange(start=0, stop=0.501, step=0.01)

test_statistics = [
	partial(kravchuk.modified_cochran_chi2, tol=tol),
	partial(kravchuk.unbias_cochran_chi2, tol=tol),
	partial(kravchuk.adaptive_ustat, tol=tol)
]
names = [
	'mod. Cochran\'s $\chi^2$',
	'debiased Cochran\'s $\chi^2$ (I)',
	'debiased Cochran\'s $\chi^2$ (II)'
]

assert len(names) == len(test_statistics)

###################################################
# Simulation for homogeneity testing
###################################################


key = random.key(seed=seed)
key0, key1 = random.split(key, num=2)

store_path = "./results/unknown_homogeneity_validity.parquet"
if usecache and os.path.isfile(store_path):

	df = pd.read_parquet(store_path, engine='pyarrow')

else:

	df = pd.DataFrame(columns=['p0',
							   'quantile',
							   'n',
							   't',
							   'Ts',
							   'test'])

	# all lower-bounds assume that p0 is less tha 0.5
	assert np.all(np.array(p0s) <= 0.5)

	keys_ns = random.split(key0, num=len(ns))
	for n_idx, n in enumerate(ns):

		ts_ = ts[n_idx]

		keys_ts = random.split(keys_ns[n_idx], num=len(ts_))

		for t_idx, t in enumerate(ts_):

			keys_tests = random.split(keys_ts[t_idx], num=len(test_statistics))

			print('\n n={} t={} \n'.format(n, t))

			for test_idx, test_statistic in enumerate(tqdm(test_statistics)):

				tqdm.write('Test {}'.format(names[test_idx]))

				keys_p0 = random.split(keys_tests[test_idx], num=len(p0s))

				for p0_idx, p0 in enumerate(p0s):
					null_dist = distribution.PointMass(p0)

					# keys_lb = random.split(keys_p0[p0_idx], num=5)

					T = test_statistic

					# assert that the test has correct type I error
					Ts = calibrate._evaluate(key=keys_p0[p0_idx],
											 dist=null_dist,
											 n=n,
											 t=t,
											 B=B,
											 T=T)

					df = pd.concat(
						[df,
						 pd.DataFrame(
							 data={
								 'p0': p0,
								 'Ts': [Ts.tolist()],
								 'quantile': np.quantile(Ts, q=1 - alpha,
														 axis=0),
								 'mean': np.mean(Ts),
								 'n': n,
								 't': t,
								 'test': names[test_idx]},
							 index=[0])])

	df['p0'] = df['p0'].astype(np.float64)
	df['n'] = df['n'].astype(np.int32)
	df['t'] = df['t'].astype(np.int32)
	df['quantile'] = df['quantile'].astype(np.float64)
	df['mean'] = df['mean'].astype(np.float64)
	df['test'] = df['test'].astype('category')

	# parquet supports saving numeric lists
	df.to_parquet(path=store_path, engine='pyarrow')

# Plot results
df_ = df.copy()
df_['n'] = df_['n'].astype('category').cat.rename_categories(
	lambda x: 'n=' + str(x))
df_['t'] = df_['t'].astype('category').cat.rename_categories(
	lambda x: 't=' + str(x))

df_exploded = df_.explode('Ts')
df_exploded['Ts'] = df_exploded['Ts'].astype(np.float64)

for p0 in [0.01, 0.1, 0.5]:
	(
			ggplot(df_exploded[df_exploded['p0'] == p0],
				   aes(x='Ts', fill='test', color='test')) +
			geom_histogram(position='identity', alpha=0.5, bins=30) +
			facet_grid('n~t', scales='free_y') +
			labs(
				x='Null hypothesis $\pi_0 = \delta_{p_0}$',
				y='Counts',
				fill='Statistic',
				color='Statistic'
			) +
			theme_minimal()
	).save("./img/unkown_homogeneity_distribtuion_p0={}.pdf".format(p0),
		   width=10, height=4)

(
		ggplot(df_, aes(x='p0')) +
		geom_line(aes(y='mean', color='test'), size=1) +
		facet_grid('n~t', scales='fixed') +
		labs(
			x='Null hypothesis $\pi_0 = \delta_{p_0}$',
			y='Mean of statistic',
			color='Statistic'
		) +
		theme_minimal()
).save("./img/unkown_homogeneity_mean.pdf", width=10, height=4)

thresholds = df.groupby(['t', 'n', 'test']).apply(
	lambda group: pd.Series({
		'threshold': group['quantile'].max()
	})).reset_index()

# thresholds for plot
hline_df = thresholds.copy()
hline_df['test_'] = hline_df['test']
hline_df = hline_df.drop(['test'], axis=1)
hline_df['test_'] = hline_df['test_'].astype('category').cat.rename_categories(
	lambda x: 'Max. quantile of {}'.format(x))

for n_idx, n in enumerate(ns):
	hline_df = pd.concat(
		[hline_df,
		 pd.DataFrame(
			 data={'test_': 'Cochran\'s asymptotic threshold',
				   't': ts[n_idx],
				   'n': n,
				   'threshold': chi2.ppf(q=1 - alpha, df=n - 1) / (n - 1)},
		 )])

hline_df['n'] = hline_df['n'].astype('category').cat.rename_categories(
	lambda x: 'n=' + str(x))
hline_df['t'] = hline_df['t'].astype('category').cat.rename_categories(
	lambda x: 't=' + str(x))

(
		ggplot(df_, aes(x='p0')) +
		geom_line(aes(y='quantile', color='test'), size=1) +
		geom_hline(aes(yintercept='threshold', linetype='test_'),
				   color='black',
				   size=1,
				   data=hline_df) +
		facet_grid('n~t', scales='fixed') +
		labs(
			x='Null hypothesis $\pi_0 = \delta_{p_0}$',
			y='{} quantile'.format(1 - alpha),
			color='Statistic',
			linetype='Threshold'
		) +
		theme_minimal()
).save("./img/unkown_homogeneity_quantile.pdf", width=10, height=4)

# create tests
tests = []
for i, test_statistic in enumerate(test_statistics):
	name = names[i]


	def test(key, p0, B, t, n, alpha):
		threshold = thresholds[(thresholds['t'] == t)
							   & (thresholds['n'] == n)
							   & (thresholds['test'] == name)]
		threshold = threshold['threshold'].values[0]
		return test_statistic, threshold


	tests.append(test)


# Add asymptotic Cochran's chi2 test

def asymptotic_cochran_chi2_test(key, p0, B, t, n, alpha, tol):
	threshold = chi2.ppf(q=1 - alpha, df=n - 1) / (n - 1)
	return partial(kravchuk.modified_cochran_chi2, tol=tol), threshold


tests.append(partial(asymptotic_cochran_chi2_test, tol=tol))
names.append('Asymptotic Cochran\'s $\chi^2$')

run_lowerbounds(
	only_type_I_error=False,
	first_moment_perturbation=False,
	stop_if_type_I_not_controlled=True,
	key=key1,
	tests=tests,
	names=names,
	store_path="./results/unknown_homogeneity_validity.csv",
	ns=ns,
	ts=ts,
	B=B,
	alpha=alpha,
	eps=eps,
	discretization=discretization,
	usecache=usecache,
	p0s=p0s,
	prefix='unknown')