# -*- coding:utf-8 -*-

import numpy as np
import scipy.stats as sp_st

def pooled_var(*samples):
    """
    Computes the pooled variance of a set of samples.
    The variance of all populations is assumed equal.

    Parameters:
    -----------
    -samples: 1d-arrays
        A set of samples from populations with equal variance

    Returns:
    --------
    -pooledvar: float
        The pooled variance
    """

    pooledvar = 0
    ntot = 0
    for sample in samples:
        var = np.var(sample, ddof=1)
        n = len(sample)
        pooledvar += (n-1)*var
        ntot += n
    pooledvar /= (ntot-len(samples))
    return pooledvar


def ttest(sample1, sample2, equal_var=True, alternative="two-sided"):
    """
    Performs a two sample Student t-test to compare means of normal populations.
    Samples are assumed to be independent.
    Null hypothesis: Both populations have identical means.

    Parameters:
    ----------
    -sample1: 1d-array
        Sample from first population.
    -sample2: 1d-array
        Sample from second population.
    -equal_var: bool, optional
        If true (default), perform a standard two sample t-test, assuming that
        both variances are equal.
        If false, perform a Welch's t-test which does not assume equal
        variances.
    -alternative: string, optional
        If two-sided (default), the alternative hypothesis is that the means are
        not equal.
        If greater, the alternative hypothesis is that the mean of population 1
        is greater than the one of population 2.
        If less, the alternative hypohtesis is that the mean of population 2
        is greater than the one of population 1.

    Returns:
    --------
    -stat: float
        The test statistic
    -pvalue: float
        The pvalue associated to the statistic under the null hypothesis
    """
    try:
        assert isinstance(sample1, np.ndarray)
        assert isinstance(sample2, np.ndarray)
    except AssertionError:
        sample1 = np.array(sample1)
        sample2 = np.array(sample2)
        ttest(sample1, sample2, equal_var=equal_var, alternative=alternative)
    try:
        assert alternative in ["two-sided", "greater", "less"]
    except AssertionError:
        raise ValueError("alternative must be either two-sided or greater or" +\
            " less")

    mu1 = np.mean(sample1)
    mu2 = np.mean(sample2)
    n = sample1.size
    m = sample2.size

    if equal_var:
        sp = np.sqrt(pooled_var(sample1, sample2))
        stat = (mu1-mu2)/(sp * np.sqrt(1/n + 1/m))
        dof = n+m-2
    else:
        var1 = np.var(sample1, ddof=1)
        var2 = np.var(sample2, ddof=1)
        stat = (mu1-mu2)/(np.sqrt(var1/n + var2/m))
        dof = (var1/n + var2/m)**2 / ((var1/n)**2/(n-1) + (var2/m)**2/(m-1))

    if alternative=="two-sided":
        pvalue = 2*sp_st.t.cdf(-abs(stat), dof)
    elif alternative=="greater":
        pvalue = 1-sp_st.t.cdf(stat, dof)
    elif alternative=="less":
        pvalue = sp_st.t.cdf(stat, dof)

    return stat, pvalue


def ftest(sample1, sample2, alternative="two-sided"):
    """
    Performs a Fisher test to compare variances of two samples from normal
    populations.
    Samples are assumed to be independent.
    Null hypothesis: Both populations have identical variances.

    Parameters:
    ----------
    -sample1: 1d-array
        Sample from first population.
    -sample2: 1d-array
        Sample from second population.
    -alternative: string, optional
        If two-sided (default), the alternative hypothesis is that the variances
        are not equal.
        If greater, the alternative hypothesis is that the variance of 
        population 1 is greater than the one of population 2.
        If less, the alternative hypohtesis is that the variance of 
        population 2 is greater than the one of population 1.

    Returns:
    --------
    -stat: float
        The test statistic
    -pvalue: float
        The pvalue associated to the statistic under the null hypothesis
    """
    try:
        assert isinstance(sample1, np.ndarray)
        assert isinstance(sample2, np.ndarray)
    except AssertionError:
        sample1 = np.array(sample1)
        sample2 = np.array(sample2)
        ftest(sample1, sample2, alternative=alternative)

    var1 = np.var(sample1, ddof=1)
    var2 = np.var(sample2, ddof=1)
    n = sample1.size
    m = sample2.size

    stat = var1/var2

    if alternative=="two-sided":
        if stat > 1:
            pvalue = 1 - sp_st.f.cdf(stat, n-1, m-1) + \
                    sp_st.f.cdf(1/stat, m-1, n-1)
        else:
            pvalue = sp_st.f.cdf(stat, n-1, m-1) + \
                1 - sp_st.f.cdf(1/stat, m-1, n-1)
    elif alternative=="greater":
        pvalue = 1 - sp_st.f.cdf(stat, n-1, m-1)
    elif alternative=="less":
        pvalue = sp_st.f.cdf(stat, n-1, m-1)

    return stat, pvalue


def ztest(sample1, sample2, var1, var2, alternative="two-sided"):
    """
    Performs a two sample Z-test to compare means of normal populations.
    Samples are assumed to be independent.
    Null hypothesis: Both populations have identical means.

    Parameters:
    ----------
    -sample1: 1d-array
        Sample from first population.
    -sample2: 1d-array
        Sample from second population.
    -var1: float
        Known variance of population 1
    -var2: float
        Known variance of population 2
    -alternative: string, optional
        If two-sided (default), the alternative hypothesis is that the means are
        not equal.
        If greater, the alternative hypothesis is that the mean of population 1
        is greater than the one of population 2.
        If less, the alternative hypohtesis is that the mean of population 2
        is greater than the one of population 1.

    Returns:
    --------
    -stat: float
        The test statistic
    -pvalue: float
        The pvalue associated to the statistic under the null hypothesis
    """
    try:
        assert isinstance(sample1, np.ndarray)
        assert isinstance(sample2, np.ndarray)
    except AssertionError:
        sample1 = np.array(sample1)
        sample2 = np.array(sample2)
        ztest(sample1, sample2, var1=var1, var2=var2, alternative=alternative)
    try:
        assert alternative in ["two-sided", "greater", "less"]
    except AssertionError:
        raise ValueError("alternative must be either two-sided or greater or" +\
            " less")

    mu1 = np.mean(sample1)
    mu2 = np.mean(sample2)
    n = sample1.size
    m = sample2.size

    stat = (mu1-mu2)/(np.sqrt(var1/n + var2/m))

    if alternative=="two-sided":
        pvalue = 2*sp_st.norm.cdf(-abs(stat))
    else:
        pvalue = sp_st.norm.cdf(-abs(stat))

    return stat, pvalue


if __name__ == "__main__":
    pass