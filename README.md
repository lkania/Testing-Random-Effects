# Testing Random Effects for Binomial Data

## Abstract

In modern scientific research, small-scale studies with limited participants are
increasingly common. However, interpreting individual outcomes can be
challenging, making it standard practice to combine data across studies using
random effects to draw broader scientific conclusions. In this work, we
introduce an optimal methodology for assessing the goodness of fit between a
given reference distribution and the distribution of random effects arising from
binomial counts. For meta-analyses, we also derive optimal tests to evaluate
whether multiple studies consistently support a treatment’s effectiveness before
pooling the data. In all cases, we characterize the smallest separation between
the null and alternative hypotheses under the 1-Wasserstein distance that
ensures the existence of a valid and powerful test.

## Contents

<!-- TOC -->
* [Requirements](#requirements)
* [Simulations for goodness-of-fit testing (Section 3.4)](#simulations-for-goodness-of-fit-testing-section-34)
* [Application of goodness-of-fit testing: Modeling political outcomes from 1976 to 2004 (Appendix K)](#application-of-goodness-of-fit-testing-modeling-political-outcomes-from-1976-to-2004-appendix-k)
* [Simulations for homogeneity testing with a reference effect (Appendix J.1)](#simulations-for-homogeneity-testing-with-a-reference-effect-appendix-j1)
* [Simulations for homogeneity testing without a reference effect (Appendix J.2)](#simulations-for-homogeneity-testing-without-a-reference-effect-appendix-j2)
* [Application of homogeneity testing: Meta-analysis of safety concerns associated with rosiglitazone (Section 6)](#application-of-homogeneity-testing-meta-analysis-of-safety-concerns-associated-with-rosiglitazone-section-6)
<!-- TOC -->

## Requirements

Python 3.11 with the following packages

| Package      | Version |
|--------------|---------|
| jax          | 0.4.23  |
| jaxlib       | 0.4.23  |
| jaxopt       | 0.8.1   |
| scipy        | 1.11.4  |
| pandas       | 2.1.1   |
| tqdm         | 4.65.0  |
| plotnine     | 0.13.6  |
| patchworklib | 0.6.5   |

## Simulations for goodness-of-fit testing (Section 3.4)

All the images referenced in Section 3.4 can be reproduced by running:

```
python test_gof.py
```

The images are stored in the folder ```img```.

## Application of goodness-of-fit testing: Modeling political outcomes from 1976 to 2004 (Appendix K)

The following file must be downloaded and placed in the ```data``` folder:  
```
https://raw.githubusercontent.com/ramyakv/mle-for-population-params-binomial/refs/heads/master/electionDataMatrix.m
```

For your convenience, the file has already been placed in the ```data``` folder.

All the images and tables referenced in Appendix K can be reproduced by running:

```
python counties.py
```

The images are stored in the folder ```img``` while the tables are stored in
```results```.

## Simulations for homogeneity testing with a reference effect (Appendix J.1)

All the images referenced in Appendix J.1 can be reproduced by running:

```
python test_homogeneity_simple_null.py
```

The images are stored in the folder ```img```.

## Simulations for homogeneity testing without a reference effect (Appendix J.2)

All the images referenced in Appendix J.2 can be reproduced by running:

```
python test_homogeneity_composite_null.py
```

The images are stored in the folder ```img```.

## Application of homogeneity testing: Meta-analysis of safety concerns associated with rosiglitazone (Section 6)

The following information is needed for this section: Table 3 (Myocardial Infarctions
and Cardiovascular Deaths in Rosiglitazone Trials) of S. E. Nissen and K. Wolski.
Effect of Rosiglitazone on the Risk of Myocardial Infarction and Death from Cardiovascular Causes. New England Journal of Medicine, 356(
  24):2457–2471, 2007. (https://www.nejm.org/doi/10.1056/NEJMoa072761)

For your convenience, the table has been partially reproduced in the ```nisse2007.py```.

All the images and tables referenced in Section 6 can be reproduced by running:

```
python nissen2007.py
```

The images are stored in the folder ```img``` while the tables are stored in
```results```.