#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Support module for visualizing information about statistic ensembles.

Most of the routines present in this module use an input 'samples' of shape
'''
>>> samples.shape == (nObservable, nXrange, nSamples)
'''
where the first dimension corresponds to the number of observables
(dependent variable), the second dimension to the range of the independent 
variable values and the last dimension to the statistical ensemble.
"""
__version__ = 0.1
__author__ = "Christopher Körber"

#-------------------------------------------------------------------------------
# Numeric and statistic modules
import numpy as np
import scipy.stats as stats
import statsmodels.nonparametric.api as smnp

# Data management modules
import pandas as pd
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
def generatePseudoSamples(
    xRange,
    nSamples,
    expPars,
    mean=0.0,
    sDev=0.1,
):
  r"""
  Generates 3-dimensional pseudo statistical correlator data.

  The first dimension being different observables, the second the x-data
  of the correlator and the last the number of statistical samples.
  The correlator follow an exponential shape
  $$
    C_i(x) = \sum_{ a_{ij}, b_{ij} in expPars } \exp( - a_{ij} x - b_{ij} )
  $$
  and Gaussian noise is added.

  Parameters
  ----------
    xRange : array
      The number of x range for the exponential input.
      Length of array is second dimension of output.

    nSamples : int
      The number of statistical repetitions. Third dimension of array.

    expPars : 3-dimensional np.array
      The shape is (number of observables, number of exponentials, 2)
      The exponential parameters. E.g.,
        expPars = [ [(a11, b11), (a12, b12)], [(a21, b21), (a22, b22)], ... ]

    mean : double
      The mean of the Gaussian noise.

    sDev : double
      The standard deviation of the Gaussian noise

  Returns
  -------
    samples : 3-dimensional array
  """
  # Get array dimensions
  nX = len(xRange)
  nObs = expPars.shape[0]

  # Get Noise
  samples = np.random.normal(
    loc=mean, scale=sDev, size=nX*nObs*nSamples
  ).reshape(
    [nObs, nX, nSamples]
  )

  # Prepare for iteration
  xTranspose = xRange.reshape(-1, 1)
  # Add pseudo correlator to noise
  for nObs, obsPars in enumerate(expPars):
    for aij, bij in obsPars:
      samples[nObs] += np.exp(-aij*xTranspose-bij)

  return samples


#-------------------------------------------------------------------------------
def getStatisticsFrame(samples, nXStart=0, nXStep=1, obsTitles=None):
  r"""
  Computes a statistic frame for a given correlator bootstrap ensemble.

  This routine takes statistical data 'samples' (see parameters) as input.
  For each individual distribution within the sample data,
  this routine fits a Gaussian Probability Density Function (PDF) and computes
  Kernel Density Estimate (KDE).
  The output of this routine is a data frame, which contains the
  following information for each individual distribution of data within
  the samples array:
    * 'mean': the mean value of the distribution
    * 'sDev': the standard deviation of the individual distribution
    * 'kdeDiff': the relative vector norm of the KDE and the fitted PDF
        $$
          \sqrt{
            \int dx [ 2*(PDF_{KDE} - PDF_{FIT})/(PDF_{KDE} + PDF_{FIT}) ]^2
          }
        $$
    * 'Dn' and 'pValue': the statistic and the significance of the Hypothesis
          (normal distribution with given parameters)
          by the Kolmogorov-Smirnov test. of the Kolmogorov-Smirnov test
          (https://en.wikipedia.org/wiki/Kolmogorov-Smirnov_test).
  The data is classified by 'nX' and the observable name
  ('obsTitles' if present).

  Parameters
  ----------
    samples : array, shape = (nObservables, nXrange, nSamples)
      The statsitical HMC data.

    nXStart : int
      Index to nX dimension of samples array for plotting frames. Plots
      will start at this index.

    nXStep : int
      Stepindex to nX dimension of samples array for plotting frames. Only
      each 'nXStep' will be shown.

    obsTitles : None or list, length = nObservables
      Row titles for figure.

  Returns
  -------
    df : 'pandas.DataFrame'

  Note
  ----
    For the Kolmogorov-Smirnov test see 'scipy.stats.kstest' and for the KDE see
    '''python
    >>> import statsmodels.nonparametric.api as smnp
    >>> kde = smnp.KDEUnivariate(dist)
    >>> kde.fit(kernel="gau", bw="scott", fft=True, gridsize=100, cut=3)
    '''
  """
  # Allocate temp variables
  nObs, nXSize, _ = samples.shape
  if obsTitles is None:
    obsTitles = [r"O{0}".format(no) for no in range(nObs)]

  nXRange = np.arange(nXStart, nXSize, nXStep)

  data = []
  # Iterate correlators
  for nO, corrSample in enumerate(samples):
    # Iterate time steps
    for nX, dist in zip(nXRange, corrSample[nXStart::nXStep]):
      # Execute KS test
      mean, sDev = np.mean(dist), np.std(dist, ddof=1)
      ksRes = stats.kstest(dist, "norm", args=(mean, sDev))

      # Estimate KDE
      kde = smnp.KDEUnivariate(dist)
      kde.fit(kernel="gau", bw="scott", fft=True, gridsize=100, cut=3)

      # Compute integral difference between normal dist and KDE
      deltaX = kde.support[1] - kde.support[0]
      normal = stats.norm.pdf(kde.support, loc=mean, scale=sDev)
      kdeDiff = 2*(kde.density-normal)/(kde.density+normal)
      kdeDiffnorm = np.sqrt(np.sum(kdeDiff**2)*deltaX)

      # Store data
      data += [{
        "observable": obsTitles[nO],
        "nX": nX,
        "mean": mean,
        "sDev": sDev,
        "Dn": ksRes.statistic,
        "pValue": ksRes.pvalue,
        "kdeDiff": kdeDiffnorm,
      }]

  # Return frame
  return pd.DataFrame(
    data,
    columns=["observable", "nX", "mean", "sDev", "Dn", "pValue", "kdeDiff"]
  )




#-------------------------------------------------------------------------------
def getFluctuationFrame(
    dataFrame,
    valueKeys,
    collectByKeys,
    averageOverKeys=None,
):
  """Routine for computing the collective average and standard deviation
  information for specified keys in a data frame.

  Computes the mean and standard deviations for the 'valueKeys' over all keys
  no collected in 'collectByKeys'.
  Afterwards further averages over 'averageOverKeys':
  '''pseudo_code
  >>> avg[valueKey, collectAvgKey] = average(
  >>>   average( df[collectKey, valueKey, restKeys], restKeys)[keys, valueKeys],
  >>>   keys in averageOverKeys
  >>> )
  >>> std[valueKey, collectAvgKey] = average(
  >>>   std( df[collectKey, valueKey, restKeys], restKeys)[keys, valueKeys],
  >>>   keys in averageOverKeys
  >>> )
  '''
  for all valueKeys and collectAvgKeys, where
  '''pseudo_code
  >>> collectAvgKey in collectKey and not in averageOverKeys
  '''

  Parameters
  ----------
    dataFrame : 'pandas.DataFrame'
      Data frame which must contain the the values of the following keys.

    valueKeys : list of strings
      The target values for which the means and standard deviations will be
      computed.

    collectByKeys : list of strings
      Dependent columns for the data.
      These informations will be separated and not averaged out.

    averageOverKeys : None or list of strings
      If not None: This routines first computes the average and mean
      for the target values separated by by the 'collectByKeys' values.
      Afterwards, another average will be computed over these keys.

  Parameters
  ----------
    fluctFrame : 'pandas.DataFrame'


  Note
  ----
    'averageOverKeys' does not affect the average values when ignoring it in
    'collectByKeys': the 'avg_...' values are the same for
    '''pseudo_code
    >>> (collectByKeys = ['key1, key2'], averageOverKeys = ['key1, key2']) and
    >>> (collectByKeys = ['key1'], averageOverKeys = ['key1'])
    '''
    However, the standard deviation is affected by that.
  """
  # Compute standard deviation for bootstrap mean and sdev for non-grouped cols
  fluctFrame = dataFrame.groupby(collectByKeys)[valueKeys].mean()
  # give them proper labels
  fluctFrame.columns = ["avg_%s" % key for key in valueKeys]

  # Compute average standard deviation
  for key in valueKeys:
    fluctFrame["std_%s" % key] = dataFrame.groupby(collectByKeys)[key].std()

  # Compute further averaging over avgOverGroups
  if averageOverKeys is None:
    return fluctFrame
  else:
    collectAvgGroups = [el for el in collectByKeys if not el in averageOverKeys]
    return fluctFrame.groupby(collectAvgGroups).mean()
