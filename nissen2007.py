from functools import partial

import jax
import pandas as pd
import patchworklib as pw
from plotnine import ggplot, aes, labs, theme_minimal, geom_histogram, \
	geom_vline
from tqdm import tqdm

import calibrate
import distribution
import empirical
import kravchuk

jax.config.update("jax_enable_x64", True)
from jax import numpy as np, random

from scipy.stats import chi2
import pvalues
import ci

# Table 3 (Table 3. Myocardial Infarctions and
# Cardiovascular Deaths in Rosiglitazone Trials.) of
# https://www.nejm.org/doi/10.1056/NEJMoa072761
# DOI: 10.1056/NEJMoa072761

columns = [
	"ID",  # Study ID
	"R_N",  # No. of Patients (Rosiglitazone Group)
	"R_MI",  # Myocardial Infarction (Rosiglitazone Group)
	"R_D",  # Death from Cardiovascular Cause (Rosiglitazone Group)
	"C_N",  # No. of Patients (Control Group)
	"C_MI",  # Myocardial Infarction (Control Group)
	"C_D"  # Death from Cardiovascular Cause (Control Group)
]

data = [
	["49653/011", 357, 2, 1, 176, 0, 0],
	["49653/020", 391, 2, 0, 207, 1, 0],
	["49653/024", 774, 1, 0, 185, 1, 0],
	["49653/093", 213, 0, 0, 109, 1, 0],
	["49653/094", 232, 1, 1, 116, 0, 0],
	["100684", 43, 0, 0, 47, 1, 0],
	["49653/143", 121, 1, 0, 124, 0, 0],
	["49653/211", 110, 5, 3, 114, 2, 2],
	["49653/284", 382, 1, 0, 384, 0, 0],
	["712753/008", 284, 1, 0, 135, 0, 0],
	["AVM100264", 294, 0, 2, 302, 1, 1],
	["BRL 49653C/185", 563, 2, 0, 142, 0, 0],
	["BRL 49653/334", 278, 2, 0, 279, 1, 1],
	["BRL 49653/347", 418, 2, 0, 212, 0, 0],
	["49653/015", 395, 2, 2, 198, 1, 0],
	["49653/079", 203, 1, 1, 106, 1, 1],
	["49653/080", 104, 1, 0, 99, 2, 0],
	["49653/082", 212, 2, 1, 107, 0, 0],
	["49653/085", 138, 3, 1, 139, 1, 0],
	["49653/095", 196, 0, 1, 96, 0, 0],
	["49653/097", 122, 0, 0, 120, 1, 0],
	["49653/125", 175, 0, 0, 173, 1, 0],
	["49653/127", 56, 1, 0, 58, 0, 0],
	["49653/128", 39, 1, 0, 38, 0, 0],
	["49653/134", 561, 0, 1, 276, 2, 0],
	["49653/135", 116, 2, 2, 111, 3, 1],
	["49653/136", 148, 1, 2, 143, 0, 0],
	["49653/145", 231, 1, 1, 242, 0, 0],
	["49653/147", 89, 1, 0, 88, 0, 0],
	["49653/162", 168, 1, 1, 172, 0, 0],
	["49653/234", 116, 0, 0, 61, 0, 0],
	["49653/330", 1172, 1, 1, 377, 0, 0],
	["49653/331", 706, 0, 1, 325, 0, 0],
	["49653/137", 204, 1, 0, 185, 2, 1],
	["SB-712753/002", 288, 1, 1, 280, 0, 0],
	["SB-712753/003", 254, 1, 0, 272, 0, 0],
	["SB-712753/007", 314, 1, 0, 154, 0, 0],
	["SB-712753/009", 162, 0, 0, 160, 0, 0],
	["49653/132", 442, 1, 1, 112, 0, 0],
	["AVA100193", 394, 1, 1, 124, 0, 0],
	["DREAM", 2635, 15, 12, 2634, 9, 10],
	["ADOPT", 1456, 27, 2, 2895, 41, 5]
]

# check that all rows have the same length
for row in data:
	assert len(row) == 7

df = pd.DataFrame(data, columns=columns)

# asserts that we match the total counts given in Nissen et al. 2007
# 42 clinical trials
assert len(df) == 42

# There were 86 myocardial infarctions in the rosiglitazone group
assert df["R_MI"].sum() == 86

# There were 39 deaths from cardiovascular causes in the rosiglitazone group
assert df["R_D"].sum() == 39

# There were 72 myocardial infarctions in the control group.
assert df["C_MI"].sum() == 72

# There were 22 deaths from cardiovascular causes in the control group
assert df["C_D"].sum() == 22

assert df["R_N"].sum() == 15556
assert df["C_N"].sum() == 12277

print('There are n={} observations and t in [{},{}]'.format(
	len(df),
	min(df['R_N'].min(), df['C_N'].min()),
	max(df['R_N'].max(), df['C_N'].max())))

rdf = []
for _, row in df.iterrows():
	rdf.append(
		{'N': row['R_N'], 'MI': row['R_MI'], 'D': row['R_D'], 'control': 'No'})
	rdf.append(
		{'N': row['C_N'], 'MI': row['C_MI'], 'D': row['C_D'], 'control': 'Yes'})
rdf = pd.DataFrame(rdf)
rdf['control'] = rdf['control'].astype('category')

# We check the homogeneity of the studies across each column

rdf["MI_N"] = rdf["MI"] / rdf["N"]
rdf["D_N"] = rdf["D"] / rdf["N"]

mean = rdf.groupby('control')['MI_N'].mean().reset_index()
mean = mean.rename(columns={'MI_N': 'Average_MI_N'})
rdf = pd.merge(rdf, mean, on='control', how='inner')

mean = rdf.groupby('control')['D_N'].mean().reset_index()
mean = mean.rename(columns={'D_N': 'Average_D_N'})
rdf = pd.merge(rdf, mean, on='control', how='inner')

p1 = (
		ggplot(rdf,
			   aes(x='MI_N',
				   fill='control',
				   color='control')) +
		geom_histogram(position='identity',
					   alpha=0.5,
					   bins=25) +
		geom_vline(aes(xintercept='Average_MI_N',
					   color='control'),
				   linetype='dashed',
				   size=1) +
		labs(
			# title='Exp',
			x='Myocardial infarction / No. of Patients',
			y='Counts',
			fill='Control',
			color='Control'
		) +
		theme_minimal()
)

p2 = (
		ggplot(rdf,
			   aes(x='D_N',
				   fill='control',
				   color='control')) +
		geom_histogram(position='identity',
					   alpha=0.5,
					   bins=25) +
		geom_vline(aes(xintercept='Average_D_N',
					   color='control'),
				   linetype='dashed',
				   size=1) +
		labs(
			# title='Exp',
			x='Death from cardiovascular cause / No. of Patients',
			y='Counts',
			fill='Control',
			color='Control'
		) +
		theme_minimal()
)

g1 = pw.load_ggplot(p1, figsize=(3, 2))
g2 = pw.load_ggplot(p2, figsize=(3, 2))
g = (g2 | g1)
g.savefig("./img/nissen2007_data.pdf",
		  dpi=600,
		  bbox_inches='tight')


def run_analysis(p0s,
				 key,
				 xs,
				 ts,
				 n,
				 B,
				 tol,
				 maxit,
				 alpha,
				 string,
				 eps,
				 upper_alpha,
				 usecache):
	print('\n Data: {} \n'.format(string))

	xs = xs.reshape(-1)
	ts = ts.reshape(-1)
	assert np.all(ts > 0)
	assert np.all(xs >= 0)
	assert n > 0
	assert len(ts) == len(xs)
	assert len(ts) == n

	# Creating confidence interval by inverting
	# procedures for homogeneity testing with simple null

	key0, key1 = random.split(key, num=2)

	storage_path = './results/nissen2007_{}_cis.csv'.format(string)
	if usecache:
		cis = pd.read_csv(storage_path, encoding="utf-8")
		cis['test'] = cis['test'].astype('category')
		cis['lower'] = cis['lower'].astype(np.float64)
		cis['upper'] = cis['upper'].astype(np.float64)
	else:

		tests = [
			kravchuk.mean_homogeneity_test,
			kravchuk.local_minimax_homogeneity_test,
			kravchuk.l2_homogeneity_test,
			kravchuk.debiased_l2_homogeneity_test,
			empirical.plugin_homogeneity_test,
			partial(kravchuk.pearson_chi2_homogeneity_test, tol=tol),
			partial(kravchuk.LRT_homogeneity_test, tol=tol)
		]
		names = [
			'Mean',
			'Local Minimax',
			'$\ell_2$',
			'de-biased $\ell_2$',
			'Plug-in',
			'mod. Pearson\'s $\chi^2$',
			'mod. LRT'
		]

		cis = ci.evaluate_cis(key=key0,
							  alpha=alpha,
							  x=xs,
							  t=ts,
							  n=n,
							  tests=tests,
							  names=names,
							  tol=tol,
							  eps=eps,
							  maxit=maxit,
							  B=B,
							  p0s=p0s)

		print(cis)
		cis.to_csv(storage_path,
				   float_format="%.15f",
				   index=False)

	# Homogeneity testing with composite null

	storage_path = './results/nissen2007_{}_pvalues.csv'.format(string)
	if usecache:
		dfh = pd.read_csv(storage_path, encoding="utf-8")
		dfh['test'] = dfh['test'].astype('category')
		dfh['pvalue'] = dfh['pvalue'].astype(np.float64)
	else:
		test_statistics = [
			partial(kravchuk.modified_cochran_chi2_ts, tol=tol),
			partial(kravchuk.unbias_cochran_chi2_ts, tol=tol),
			partial(kravchuk.adaptive_ustat_ts, tol=tol)
		]
		names = [
			'mod. Cochran\'s $\chi^2$',
			'$R_1$',
			'$R_2$'
		]

		assert len(names) == len(test_statistics)

		dfh = pd.DataFrame(columns=['pvalue', 'test'])

		# tests calibrated for the finite sample

		keys_test_statistics = random.split(key1, num=len(test_statistics))

		for test_statistic_idx, test_statistic in enumerate(
				tqdm(test_statistics)):
			tqdm.write('\n Test {}'.format(names[test_statistic_idx]))

			pvalue = pvalues.get_pvalues(
				key=keys_test_statistics[test_statistic_idx],
				p0s=p0s,
				maxit=maxit,
				test_statistic=test_statistic,
				xs=xs,
				ts=ts,
				n=n,
				B=B,
				tol=tol,
				upper_alpha=upper_alpha)

			tqdm.write('\n pvalue {}'.format(pvalue))

			dfh = pd.concat(
				[dfh,
				 pd.DataFrame(
					 data={
						 'pvalue': pvalue,
						 'test': names[test_statistic_idx]},
					 index=[0])])

		# Asymptotic Cochran's chi-squared p-value

		quantiles = lambda alpha: np.array(
			chi2.ppf(q=1 - np.maximum(alpha, tol), df=n - 1) / (n - 1))
		cochran_chi2 = kravchuk.modified_cochran_chi2_ts(x=xs,
														  n=n,
														  t=ts,
														  tol=tol)
		pvalue = pvalues.get_pvalue(test_statistic_value=cochran_chi2,
									threshold=quantiles,
									tol=tol,
									maxit=maxit,
									upper_alpha=upper_alpha)

		dfh = pd.concat(
			[dfh,
			 pd.DataFrame(
				 data={
					 'pvalue': pvalue,
					 'test': 'Asymptotic mod. Cochran\'s $\chi^2$'},
				 index=[0])])

		dfh['pvalue'] = dfh['pvalue'].astype(np.float64)
		dfh['test'] = dfh['test'].astype('category')

		print(dfh)
		dfh.to_csv(storage_path,
				   float_format="%.15f",
				   index=False)

	return dfh, cis


seed = 0
key = random.key(seed=seed)
B = 10000
alpha = 0.05
eps = 0.012
tol = 1e-10
usecache = False
maxit = 100
p0s = np.arange(start=0, stop=0.501, step=0.001)

keys = random.split(key, num=4)

xs = np.array(df['R_MI'].values)
ts = np.array(df['R_N'].values)
print(ts)
n = len(ts)
dfh_R_MI, cis_R_MI = run_analysis(p0s=p0s,
								  usecache=usecache,
								  key=keys[0],
								  xs=xs,
								  ts=ts,
								  n=n,
								  B=B,
								  tol=tol,
								  maxit=maxit,
								  alpha=alpha,
								  eps=eps,
								  string='R_MI_N',
								  upper_alpha=0.1)

xs = np.array(df['R_D'].values)
ts = np.array(df['R_N'].values)
n = len(ts)
dfh_R_D, cis_R_D = run_analysis(p0s=p0s,
								usecache=usecache,
								key=keys[1],
								xs=xs,
								ts=ts,
								n=n,
								B=B,
								tol=tol,
								maxit=maxit,
								alpha=alpha,
								eps=eps,
								string='R_D_N',
								upper_alpha=0.1)

xs = np.array(df['C_MI'].values)
ts = np.array(df['C_N'].values)
n = len(ts)
dfh_C_MI, cis_C_MI = run_analysis(p0s=p0s,
								  usecache=usecache,
								  key=keys[2],
								  xs=xs,
								  ts=ts,
								  n=n,
								  B=B,
								  tol=tol,
								  maxit=maxit,
								  alpha=alpha,
								  eps=eps,
								  string='C_MI_N',
								  upper_alpha=0.5)

xs = np.array(df['C_D'].values)
ts = np.array(df['C_N'].values)
n = len(ts)
dfh_C_D, cis_C_D = run_analysis(p0s=p0s,
								usecache=usecache,
								key=keys[3],
								xs=xs,
								ts=ts,
								n=n,
								B=B,
								tol=tol,
								maxit=maxit,
								alpha=alpha,
								eps=eps,
								string='C_D_N',
								upper_alpha=0.5)

pvalues_ = pd.DataFrame(columns=['test', 'R_MI', 'R_D', 'C_MI', 'C_D'])

dfh_R_MI = dfh_R_MI.rename(columns={'pvalue': 'MI (Rosiglitazone)'})
dfh_R_D = dfh_R_D.rename(columns={'pvalue': 'D (Rosiglitazone)'})
dfh_C_MI = dfh_C_MI.rename(columns={'pvalue': 'MI (Control)'})
dfh_C_D = dfh_C_D.rename(columns={'pvalue': 'D (Control)'})

pvalues_ = pd.merge(dfh_R_MI, dfh_R_D, on='test', how='inner')
pvalues_ = pd.merge(pvalues_, dfh_C_MI, on='test', how='inner')
pvalues_ = pd.merge(pvalues_, dfh_C_D, on='test', how='inner')

pvalues_ = pvalues_.rename(columns={'test': 'Test'})

ordered_columns = ['Test',
				   'MI (Rosiglitazone)',
				   'MI (Control)',
				   'D (Rosiglitazone)',
				   'D (Control)']
pvalues_ = pvalues_.reindex(columns=ordered_columns)

print(pvalues_.round(3).to_latex(index=False,
								 float_format="%.3g"))

pvalues_.round(3).to_latex('./results/nissen2007_pvalues.tex',
						   float_format="%.3g",
						   index=False)


def row_convert(row):
	if row['lower'] < 0:
		return 'Rejected all'
	else:
		return "[{}, {}]".format(
			row['lower'], row['upper'], float_format="%.3g")


def merge_columns(df):
	# Combine the two columns into one column with intervals
	df = df.round(3)
	df['ci'] = df.apply(row_convert, axis=1)
	df = df.drop(['lower', 'upper'], axis=1)
	return df


cis_R_MI_ = merge_columns(cis_R_MI)
cis_R_D_ = merge_columns(cis_R_D)
cis_C_MI_ = merge_columns(cis_C_MI)
cis_C_D_ = merge_columns(cis_C_D)

cis_R_MI_ = cis_R_MI_.rename(columns={'ci': 'MI (Rosiglitazone)'})
cis_R_D_ = cis_R_D_.rename(columns={'ci': 'D (Rosiglitazone)'})
cis_C_MI_ = cis_C_MI_.rename(columns={'ci': 'MI (Control)'})
cis_C_D_ = cis_C_D_.rename(columns={'ci': 'D (Control)'})

cis = pd.merge(cis_R_MI_, cis_R_D_, on='test', how='inner')
cis = pd.merge(cis, cis_C_MI_, on='test', how='inner')
cis = pd.merge(cis, cis_C_D_, on='test', how='inner')

cis = cis.rename(columns={'test': 'Test'})

ordered_columns = ['Test',
				   'MI (Rosiglitazone)',
				   'MI (Control)',
				   'D (Rosiglitazone)',
				   'D (Control)']
cis = cis.reindex(columns=ordered_columns)

print(cis.round(3).to_latex(index=False,
							float_format="%.3g"))

cis.round(3).to_latex('./results/nissen2007_cis.tex',
					  float_format="%.3g",
					  index=False)