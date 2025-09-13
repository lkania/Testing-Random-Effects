from functools import partial

import jax
import pandas as pd
import patchworklib as pw
from plotnine import ggplot, aes, labs, theme_minimal, geom_histogram, \
	geom_line, geom_point, position_dodge

from tqdm import tqdm

import calibrate
import distribution
import empirical
import kravchuk
import mom

jax.config.update("jax_enable_x64", True)
from jax import numpy as np, random

import mle
import pvalues

from scipy.io import loadmat

#####################################################
# Parameters
#####################################################

seed = 0
key = random.key(seed=seed)
tol = 1e-10
maxit = 500
B = 5000
discretization = 10000

data = loadmat('./data/ElectionProcess.mat')
data = data['electionData']
t = 8

# remove counties with less than 8 counts

data = data[data[:, 0] >= t]
assert np.all(data[:, 0] == t)
assert np.max(data[:, 1]) <= t
assert np.min(data[:, 1]) >= 0

# number of times a county voted for the Republican party
xs = data[:, 1]
counts = np.bincount(xs, length=t + 1)
n = len(xs)
data = pd.DataFrame({'values': xs / t})

# Split into two parts

key0, key = random.split(key, num=2)
shuffled_indices = jax.random.permutation(key0, np.arange(n))
split_point = n // 2
indices_1 = shuffled_indices[:split_point]
indices_2 = shuffled_indices[split_point:]
xs1 = xs[indices_1]
xs2 = xs[indices_2]

results = pd.DataFrame(columns=['dist', 'x', 'weight'])

plugin_dist_1 = distribution.FiniteMixture(
	ps=np.arange(t + 1) / t,
	weights=np.bincount(xs1, length=t + 1) / len(xs1))
plugin_dist_2 = distribution.FiniteMixture(
	ps=np.arange(t + 1) / t,
	weights=np.bincount(xs2, length=t + 1) / len(xs2))

dataset = pd.DataFrame(columns=['dist', 'x', 'weight'])
dataset = pd.concat([dataset,
					 pd.DataFrame(
						 data={'dist': 'Train',
							   'x': plugin_dist_1.ps,
							   'weight': plugin_dist_1.weights})])
dataset = pd.concat([dataset,
					 pd.DataFrame(
						 data={'dist': 'Test',
							   'x': plugin_dist_2.ps,
							   'weight': plugin_dist_2.weights})])

xs = xs1
counts = np.bincount(xs, length=t + 1)
n = len(xs)

results = pd.DataFrame(columns=['dist', 'x', 'weight'])

plugin_dist = distribution.FiniteMixture(
	ps=np.arange(t + 1) / t,
	weights=counts / n)

results = pd.concat([results,
					 pd.DataFrame(
						 data={'dist': 'Empirical',
							   'x': plugin_dist.ps,
							   'weight': plugin_dist.weights})])

mle_dist = mle.mle(xs=xs,
				   n=n,
				   t=t,
				   m=discretization,
				   tol=tol,
				   maxiter=maxit)

results = pd.concat([results,
					 pd.DataFrame(
						 data={'dist': 'MLE',
							   'x': mle_dist.ps,
							   'weight': mle_dist.weights})])

mom_dist = mom.mom(xs=xs,
				   n=n,
				   t=t,
				   m=discretization,
				   tol=tol,
				   maxiter=maxit)

results = pd.concat([results,
					 pd.DataFrame(
						 data={'dist': 'MOM',
							   'x': mom_dist.ps,
							   'weight': mom_dist.weights})])

moments = pd.DataFrame(columns=['dist', 'value', 'moment'])
moments = pd.concat([moments,
					 pd.DataFrame(
						 data={'dist': 'MOM',
							   'value': mom_dist.moments(t)[1:],
							   'moment': np.arange(t + 1)[1:]})])
moments = pd.concat([moments,
					 pd.DataFrame(
						 data={'dist': 'MLE',
							   'value': mle_dist.moments(t)[1:],
							   'moment': np.arange(t + 1)[1:]})])
moments = pd.concat([moments,
					 pd.DataFrame(
						 data={'dist': 'Empirical',
							   'value': plugin_dist.moments(t)[1:],
							   'moment': np.arange(t + 1)[1:]})])
moments = pd.concat([moments,
					 pd.DataFrame(
						 data={'dist': 'Unbiased',
							   'value': mom.estimate_moments(counts=counts,
															 n=n,
															 t=t)[1:],
							   'moment': np.arange(t + 1)[1:]})])

p = (
		ggplot(moments,
			   aes(x='moment',
				   y='value',
				   color='dist',
				   linetype='dist')) +
		geom_point(size=3,
				   position=position_dodge(0.2)) +
		labs(
			x='Moment',
			y='Value',
			color='Method',
			linetype='Method'
		) +
		theme_minimal()
)
p.save("./img/counties_moments.pdf",
	   width=6, height=3)

p1 = (
		ggplot(dataset,
			   aes(x='x',
				   weight='weight',
				   color='dist',
				   fill='dist')) +
		geom_histogram(position='identity',
					   alpha=0.5,
					   bins=35) +
		labs(
			x='No. of times Republicans won / No. of elections',
			y='Approximate density',
			color='Empirical\nDistribution',
			fill='Empirical\nDistribution'
		) +
		theme_minimal()
)

p2 = (
		ggplot(results,
			   aes(x='x',
				   weight='weight',
				   color='dist',
				   fill='dist')) +
		geom_histogram(position='identity',
					   alpha=0.3,
					   bins=35) +
		labs(
			x='No. of times Republicans won / No. of elections',
			y='Approximate density',
			color='Estimated\nDistribution',
			fill='Estimated\nDistribution'
		) +
		theme_minimal()
)

g1 = pw.load_ggplot(p1, figsize=(3, 2))
g2 = pw.load_ggplot(p2, figsize=(3, 2))
g = (g1 | g2)
g.savefig("./img/counties_data.pdf",
		  dpi=600,
		  bbox_inches='tight')

xs = xs2
counts = np.bincount(xs, length=t + 1)
n = len(xs)

null_dists = [mle_dist, mom_dist, plugin_dist]
null_dists_names = ['MLE', 'MOM', 'Plug-in']

pvalues_ = pd.DataFrame(columns=['pvalue', 'test', 'dist'])

keys_dists = random.split(key, num=len(null_dists))

for dist_idx, null_dist in enumerate(null_dists):

	key0, key1 = random.split(keys_dists[dist_idx], num=2)

	efingerprint = calibrate.expected_fingerprint(key=key0,
												  dist=null_dist,
												  B=B,
												  t=t)

	tests = [
		partial(kravchuk.global_minimax_gof_test,
				efingerprint=efingerprint,
				tol=tol),
		partial(kravchuk.debiased_pearson_chi2_gof_test,
				tol=tol,
				efingerprint=efingerprint),
		partial(kravchuk.pearson_chi2_gof_test,
				tol=tol,
				efingerprint=efingerprint),
		partial(kravchuk.LRT_gof_test,
				tol=tol,
				efingerprint=efingerprint),
		empirical.plugin_gof_test,
		partial(mom.mom_test,
				m=discretization,
				tol=tol,
				maxiter=maxit),
		partial(mle.mle_test,
				m=discretization,
				tol=tol,
				maxiter=maxit)
	]
	tests_names = [
		'Global minimax',
		'debiased Pearson\'s $\chi^2$',
		'mod. Pearson\'s $\chi^2$',
		'mod. LRT',
		'Plug-in',
		'Distance to MOM',
		'Distance to MLE'
	]

	assert len(tests) == len(tests_names)

	keys_tests = random.split(key1, num=len(tests))

	for test_idx, create_test in enumerate(tqdm(tests)):
		tqdm.write('\n Test {}'.format(tests_names[test_idx]))

		test_statistic = create_test(key=None,
									 null_dist=null_dist,
									 B=B,
									 t=t,
									 n=n)

		value = test_statistic(counts=counts, n=n, t=t)

		Ts = calibrate._evaluate(key=keys_tests[test_idx],
								 dist=null_dist,
								 n=n,
								 t=t,
								 B=B,
								 T=test_statistic)

		dim = Ts.ndim

		talpha = lambda alpha: np.quantile(Ts, q=1 - alpha / dim, axis=0)

		pvalue = pvalues.get_pvalue(test_statistic_value=value,
									threshold=talpha,
									tol=tol,
									maxit=maxit,
									upper_alpha=0.1,
									verbose=False)

		tqdm.write('\n P-value {}'.format(pvalue))

		pvalues_ = pd.concat(
			[pvalues_,
			 pd.DataFrame(
				 data={
					 'pvalue': pvalue,
					 'test': tests_names[test_idx],
					 'dist': null_dists_names[dist_idx]},
				 index=[0])])

pvalues_['pvalue'] = pvalues_['pvalue'].astype(np.float64)
pvalues_['test'] = pvalues_['test'].astype('category')
pvalues_['dist'] = pvalues_['dist'].astype('category')

pivot_df = pvalues_.round(3).pivot(index='test',
								   columns='dist',
								   values='pvalue')
pivot_df.columns.name = None
pivot_df.reset_index(inplace=True)
pivot_df = pivot_df.sort_values(by='MLE', ascending=True)
pivot_df = pivot_df.rename(columns={'test': 'Test'})

print(pivot_df)

print(pivot_df.to_latex(index=False,
						float_format="%.3g"))

pivot_df.to_latex('./results/counties_pvalues.tex',
				  float_format="%.3g",
				  index=False)