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
# System modules
import re

# Numerics and statistics modules
import numpy as np
import scipy.stats as stats
import statsmodels.nonparametric.api as smnp

# Plotting modules
import matplotlib
import matplotlib.pylab as plt
import seaborn as sns

# Module for uncertainty treatment
import gvar as gv
#-------------------------------------------------------------------------------



#-------------------------------------------------------------------------------
def plotSamples(xRange, samples, ax=None, **kwargs):
  """
  Creates errorbar plot for all the correlator components in one frame.

  Parameters
  ----------
    xRange : array
      The number of x-values for sample input.
      Length of array is second dimension of samples.

    samples : 3-dimensional array
      Dimensions are the number of observables, the number of x-values and
      the number of statistical samples.

    ax : 'matplotlib.axes'
      Plot samples in this figure. If 'None', create new object.

    **kwargs: keyword arguments
      Will be passed to 'ax.errorbar'.

    Returns
    -------
      ax : 'matplotlib.axes'
  """
  # Get access if specified
  if ax is None:
    ax = plt.gca()

  # Make error bar plot for individual observables
  for nObs, corr in enumerate(samples):
    mean = corr.mean(axis=1)
    sdev = corr.std(axis=1, ddof=1)
    ax.errorbar(xRange, mean, yerr=sdev, label="Observable %i" % nObs, **kwargs)

  return ax


#-------------------------------------------------------------------------------
def plotSampleDistributions(
    samples,
    nXStart=0,
    nXStep=0,
    obsTitles=None,
    xRange=None
):
  """
  Creates a 'matplotlib.figure' which contains a grid of distribution plots.
  The data is plotted for all 'Observables' values on the x-axis and
  selected 'xRange' values on the y-axis.
  Each individual frame contains a distribution plot with a fitted PDF, Bins
  and KDE.

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

    xRange : None or iterable
      If 'None' creates a range of x-values from 'nXStart', 'nXSetp' and
      'samples.shape'.
      If specified, takes this as the x-range. make sure it agrees with the
      'samples.shape'.

  Returns
  -------
    fig : 'matplotlib.figure'
  """
  # Set up display
  font = "Times"
  sns.set(
    style="ticks",
    font_scale=0.5,
  )
  matplotlib.rcParams['mathtext.fontset'] = 'cm'
  font = {
    'family':'serif',
    'serif': ['times new roman'],
    'style': 'normal',
    'variant': 'normal',
    'weight': 'ultralight'
  }
  plt.rc('font', **font)

  # Number of columns and rows
  nObs = samples.shape[0]
  nX = samples.shape[1]
  # Access shifted range
  if xRange is None:
    xRange = np.arange(nXStart, nX, nXStep)
  nCols = len(xRange)
  samples = samples[:, nXStart::nXStep, :]

  if obsTitles is None:
    obsTitles = [r"$\mathcal{{O}}_{{{0}}}$".format(no) for no in range(nObs)]

  # Create the figure grid
  fig, axs = plt.subplots(
    dpi=400,
    figsize=(0.8*nObs, 0.4*nCols),
    nrows=nCols,
    ncols=nObs,
  )

  # Loop through columns and rows
  for no, (axRow, corrData) in enumerate(zip(axs.T, samples)):
    for nx, (ax, corrDataNx) in enumerate(zip(axRow, corrData)):

      # Set titles, row labels and legend
      if nx == 0:
        ax.set_title(obsTitles[no], y=1.2)
      if no == 0:
        ax.set_ylabel(xRange[nx])
      if nx == 0 and no == 0:
        legends = [r"$\mu$", "Bins", "PDF"]
      else:
        legends = [None, None, None]

      # Get mean and standard deviation of sample
      ## Plot an horizontal line at position
      mean, sdev = np.mean(corrDataNx), np.std(corrDataNx, ddof=1)
      ax.axvline(mean, color="black", lw=0.5, ls="--", label=legends[0])

      # Plot the histogram including KDE
      p = sns.distplot(
        corrDataNx,
        kde=True,
        norm_hist=True,
        ax=ax,
        kde_kws={"alpha":0.8, "lw":0.7},
        hist_kws={"histtype":"stepfilled"},
        label=legends[1]
      )

      # Plot the distribution estimate
      x = np.linspace(*p.get_xlim(), num=100)
      ax.plot(
        x,
        stats.norm.pdf(x, loc=mean, scale=sdev),
        ls="--", lw=1.0,
        label=legends[2]
      )

      # set legend if fist entry
      if nx == 0 and no == 0:
        ax.legend(
          loc="center",
          fontsize="xx-small",
          bbox_to_anchor=[-0.1, 1.5, 0, 0],
          frameon=True
        )

      # Further styling
      ## Remove y-axis
      ax.set(yticks=[])
      ## Set up x-axis to only contain mu values on top
      ax.set_xticks([mean])
      ## Create mu text
      mustr = r"$\mu ={mu} $".format(mu=gv.gvar(mean, sdev))
      if "e-" in mustr:
        mustr = re.sub("e-0*([1-9]+)", r"\cdot 10^{-\g<1>}", mustr)
      ax.set_xticklabels([mustr], fontdict={"size":"xx-small"})
      ax.xaxis.tick_top()
      ax.tick_params(direction='out', length=0, width=0.5, pad=-0.5, top=True)

      ## Reduce line width of visible axis
      ax.spines["bottom"].set_linewidth(0.5)
      ax.spines["top"].set_linewidth(0.)

  # Remove other remaining axis
  sns.despine(fig, left=True, right=True, top=False)
  # And adjust intermediate distances
  fig.subplots_adjust(wspace=0.05, hspace=0.45)

  return fig


#-------------------------------------------------------------------------------
def plotDistribution(dist):
  r"""Plots the fitted PDF, KDE and CDF as well as the PDF differences between
  fits, binning and KDE.
  The figure contains additional informations like:
    * Kolmogorov-Smirnov test statistics and P-values
    * The KDE difference defined by
      $$
        \Delta PDF(x)
        = 2*[PDF_{KDE}(x) - PDF_{FIT}(x)]/[PDF_{KDE}(x) + PDF_{FIT}(x)]
      $$
      and the integrated KDE difference is given by
      $$ \sqrt{ \int dx [\Delta PDF(x)]^2 } $$

  Parameters
  ----------
    dist : array or list, one dimensional

  Returns
  -------
    fig : 'matplotlib.figure'

  Note
  ----
    Abbreviations:
    * KDE : Kernel Density Estimate
    * PDF : Probability Density Function
    * CDF : Cumulative Density Function

    This routine uses seaborn to estimate the bins and KDE, scipy for the
    Kolmogorov-Smirnov test
    (https://en.wikipedia.org/wiki/Kolmogorov-Smirnov_test)
    and 'statsmodels' for estimating the KDE
    '''python
    >>> import statsmodels.nonparametric.api as smnp
    >>> kde = smnp.KDEUnivariate(data)
    >>> kde.fit(kernel="gau", bw="scott", fft=True, gridsize=100, cut=3)
    '''
    'seaborn' itself uses 'numpy' for binning where the number of bins is
    determined by the Freedman Diaconis Estimator
    (https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html).
  """
  # Create the figure
  fig, axs = plt.subplots(
    dpi=400,
    figsize=(3, 3),
    nrows=3,
    sharex=True,
    gridspec_kw={'height_ratios': [1, 3, 1]}
  )

  # Set up plot styles
  baseLineStyle = {
    "color": "gray",
    "lw": 0.5,
    "ls": "--",
    "zorder": -1
  }
  fitLineStyle = {
    "lw": 0.9,
    "color": "red",
    "label": "Fit"
  }
  kdeLineStyle = {
    "lw": 0.0,
    "marker": ".",
    "ms": 3,
    "color": "green",
    "label": "KDE",
  }
  histLineStyle = {
    "rwidth": 0.9,
    "label": "Bins",
  }
  styles = {
    "Base": baseLineStyle,
    "Fit": fitLineStyle,
    "KDE": kdeLineStyle,
    "Bins": histLineStyle,
  }

  # Compute distribution fits
  mean, sdev = np.mean(dist), np.std(dist, ddof=1)

  # Kolmogorov-Smirnov test
  ksRes = stats.kstest(dist, 'norm', args=(mean, sdev))

  # Estimate KDE and compare to normal
  kde = smnp.KDEUnivariate(dist)
  kde.fit(kernel="gau", bw="scott", fft=True, gridsize=100, cut=3)
  ## Get infinitesimal step size
  deltaX = kde.support[1] - kde.support[0]
  ## Compute fitted PDF
  normal = stats.norm.pdf(kde.support, loc=mean, scale=sdev)
  ## Compute difference
  kdeDiff = (2*(kde.density-normal)/(kde.density+normal))
  normKDEDiff = np.sqrt(np.sum(kdeDiff**2)*deltaX)

  # Set title
  axs[0].set_title(
    "KS Test result: Statistic = {stat:1.3f}, P-Value = {pvalue:1.3f}".format(
      stat=ksRes.statistic, pvalue=ksRes.pvalue
    ) + ",\nintegrated KDE difference = {normKDEDiff:1.3f}".format(
      normKDEDiff=normKDEDiff
    )
  )

  # Compute fits
  yb, xb = np.histogram(dist, bins="fd")

  #Plot PDF
  ax = axs[0]
  sns.distplot(
    dist,
    hist_kws=styles["Bins"],
    kde_kws=styles["KDE"],
    fit_kws=styles["Fit"],
    ax=ax,
    norm_hist=True,
    fit=stats.norm
  )
  ## Axis styling
  ax.axvline(mean, label=r"$\mu$", **baseLineStyle)
  ax.set_ylabel("PDF")
  ax.set_yticks([])
  ax.legend([])

  # CDFs
  ax = axs[1]
  styles["KDE"].update({"cumulative":True})
  styles["Bins"].update({"cumulative":True})
  ## Plot CDFs
  ecdf = sns.distplot(
    dist,
    hist_kws=styles["Bins"],
    kde_kws=styles["KDE"],
    ax=ax,
    norm_hist=True
  )
  ## Get the x-range
  lines = ecdf.get_lines()[0]
  xl = lines.get_xdata()
  ## Compute the fitted CDF
  cdf = stats.norm.cdf(xl, loc=mean, scale=sdev)
  ax.plot(xl, cdf, **fitLineStyle)
  ## Styling
  ax.set_ylabel("CDF")
  ax.axvline(mean, label=r"$\mu$", **baseLineStyle)
  ax.axhline(0.5, **baseLineStyle)
  ax.set_yticks(np.linspace(0.25, 1, 4))
  ax.legend(loc="upper left", frameon=True)

  # Difference plot
  ax = axs[2]
  for key in ["KDE", "Bins"]:
    styles[key].pop("cumulative")
    styles[key].pop("label")

  # Plot KDE difference
  ax.plot(kde.support, kdeDiff, **styles["KDE"])

  # Plot bin difference
  rwidth = styles["Bins"].pop("rwidth")
  styles["Bins"].pop("normed")
  midBin = (xb[1:]+xb[:-1])/2
  yb = yb/np.sum(yb*(xb[1:]-xb[:-1]))
  pdf = stats.norm.pdf(midBin, loc=mean, scale=sdev)
  diff = 2*(yb-pdf)/(yb+pdf)
  ax.bar(
    xb[:-1]+deltaX/2,
    diff,
    width=(xb[1:]-xb[:-1])*rwidth,
    align='edge',
    **styles["Bins"]
  )
  ax.set_ylabel(r"$\Delta$PDF")

  ax.set_ylim(min(-0.1, diff.min())*1.5, max(diff.max(), 0.1)*1.5)


  ## Styling
  ax.axvline(mean, **baseLineStyle)
  baseLineStyle["color"] = "black"
  baseLineStyle["ls"] = "-"
  ax.axhline(0, **baseLineStyle)

  # General styling
  for nax, ax in enumerate(axs):
    # Labels right
    ax.yaxis.set_label_position("right")
    # Ticks styling
    ax.tick_params(
      axis="both",
      direction='inout',
      width=0.5,
      length=2.5,
      top=(nax != 0)
    )
    # set line width
    for val in ax.spines.values():
      val.set_linewidth(0.5)

  # Remove line width for PDF plot
  for pos in ["left", "top", "right"]:
    axs[0].spines[pos].set_linewidth(0)

  ax.set_xlim(dist.min(), dist.max())

  # Adjust internal plot spacings
  plt.subplots_adjust(hspace=0.0)

  return fig


#-------------------------------------------------------------------------------
def errBarPlot(
    dataFrame,
    meanKey="mean",
    sDevKey="sDev",
    xKey="nBinSize",
    rowKey="observable",
    colKey="nX",
    colorKey="nSamples",
    errBarKwargs=None,
    shareY=False,
):
  """Creates a grid of errorbar plots.

  Each frame in the grid plot displays the 'meanKey' and its
  standard deviation 'sDevKey' over the independent variable 'xKey'.
  The columns of the grid are given by the 'colKey' entries and
  the rows are given by the 'rowKey'.
  The 'colorKey' plots decides which entries are shown in different
  colors within each frame.

  Parameters
  ----------
    dataFrame : 'pandas.DataFrame'
      This data frame must contain the values of the following keys.

    meanKey : string
      Name of the dataFrame key which will used for plotting the mean value for
      each frame of the grid.

    sDevKey : string
      Name of the dataFrame key which will used for plotting the standard
      deviation value for each frame of the grid.

    xKey : string
      Name of the dataFrame key which will used as the dependent variable for
      each frame of the grid.

    rowKey : string
      Name of the dataFrame key which will used as the rows of the plot grid.

    colKey : string
      Name of the dataFrame key which will used as the columns of the plot grid.

    colorKey : string
      Name of the dataFrame key which will used for discriminating different
      plots values within each frame.

    errBarKwargs : dict
      Parameters which will be passed to 'plt.errorbar'.
      These parameters will overwrite default values.

    shareY : boolean
      Specifies whether the y-entries shall be displayed on the same range
      (row wise).

  Returns
  -------
    graph : 'matplotlib.figure'

  Notes
  -----
    The y-axis scales are different for each frame and can generally not be
    compared.
  """
  # Check whether frame contains all columns
  for key in [rowKey, colKey, xKey, meanKey, sDevKey, colorKey]:
    if not key in dataFrame.columns:
      raise KeyError("Key %s not found in input frame" % key)

  # Set up the error bat plot
  errBarStyle = {
    "linestyle":"None",
    "marker":".",
    "ms":3,
    "lw":1,
    "elinewidth":0.5,
    "capthick":0.5,
    "capsize":0.5,
  }
  # Adjust by input keys
  if errBarKwargs:
    for key, val in errBarKwargs.items():
      errBarStyle[key] = val

  # Compute how much one has to shift plots for visualization
  ## Number of shifts
  colorEntries = dataFrame[colorKey].unique()
  nColors = len(colorEntries)

  ## Compute minimal independent variable distance
  xRange = dataFrame[xKey].unique()

  ## Loop through distances to get the minimal one
  dXmin = max(abs(xRange[-1] - xRange[0]), 0.1)
  for nx1, x1 in enumerate(xRange[:-1]):
    for x2 in xRange[nx1+1:]:
      if abs(x2-x1) < dXmin:
        dXmin = abs(x2-x1)
  dXmin /= 3

  ## Allocate shift of distances
  dX = {}
  for nEntry, entry in enumerate(colorEntries):
    dX[entry] = dXmin*(2*nEntry-nColors+1)*1./nColors

  ## Modify x cols
  df = dataFrame.copy()
  df[xKey] += df.apply(lambda col: dX[col[colorKey]], axis=1)

  # Create the facet grid for the mapping
  graph = sns.FacetGrid(
    data=df,
    row=rowKey,
    col=colKey,
    hue=colorKey,
    palette="Blues",
    sharex=True,
    sharey="row" if shareY else False,
  )
  ## and map the error bar plot
  graph.map(plt.errorbar, xKey, meanKey, sDevKey, **errBarStyle)

  # Change figure size
  graph.fig.set(
    dpi=500,
    figheight=2,
    figwidth=len(dataFrame[colKey].unique())*1./2
  )

  # Style individual plots
  for nax, ax in enumerate(graph.axes.flat):
    if not shareY:
      ax.set_yticks([])
    ## At most three ticks
    ax.set_xticks(np.linspace(
      dataFrame[xKey].min(), dataFrame[xKey].max(), 3, dtype=int
    ))
    ## Set the range
    ax.set_xlim(dataFrame[xKey].min()-1, dataFrame[xKey].max()+1)
    ## Set the ticks
    ax.tick_params(
      axis="both",
      direction='inout',
      width=0.5,
      length=2.5,
    )

    # Remove axis and ticks
    for pos in ["left", "top", "right"]:
      ax.spines[pos].set_linewidth(0)
    if shareY and nax % len(graph.axes[0]) == 0:
      ax.spines["left"].set_linewidth(0.5)
    else:
      ax.tick_params(
        axis="y",
        direction='inout',
        width=0.0,
        length=0.0,
      )
    ax.spines["bottom"].set_linewidth(0.5)

  # Adjust the margin titles and plot the mean of the means
  graph.set_titles("")
  means = dataFrame.groupby([rowKey, colKey])[meanKey].mean()
  for nCorr, (corrName, axRow) in enumerate(
      zip(dataFrame[rowKey].unique(), graph.axes)
  ):
    for nt, ax in  zip(dataFrame[colKey].unique(), axRow):
      if nCorr == 0:
        ax.set_title("{colKey}$ = {nt}$".format(nt=nt, colKey=colKey))
      ax.axhline(means[corrName, nt], color="black", ls="--", lw=0.5)

  # Set the labels
  graph.set_ylabels(meanKey)

  # Adjust the remaining margin titles
  for corrName, ax in zip(dataFrame[rowKey].unique(), graph.axes[:, -1]):
    ax.yaxis.set_label_position("right")
    ax.set_ylabel(corrName)

  graph.set_xlabels(xKey)
  graph.add_legend()

  # Adjust the intermediate plot spacing
  plt.subplots_adjust(wspace=0.1, hspace=0.05)

  return graph


#-------------------------------------------------------------------------------
def plotFluctuations(
    fluctFrame,
    valueKey,
    axisKey,
):
  """Routine for visualizing fluctuations of statistical data.

  Plots the average values and standard deviations for collective datasets
  in bar plots.

  Parameters
  ----------
    fluctFrame : 'pandas.DataFrame'
      A data frame containing fluctuation data.
      It must have the columns '["avg_{valueKey}", "std_{valueKey}"]' and
      'axisKey' must be specified in the indices.
      The easiest way to generate such a frame is using the
      'utilities.getFluctuationFrame' method.

    valueKey : string
      Name of the dataFrame key which dependence is analyzed.

    axisKey : string
      Name of the dataFrame key which will be displayed on the y-axis.
      Must be in the indices of the 'fluctFrame'

  Returns
  -------
    fig : 'matplotlib.figure'
  """
  # Temporarily store frame
  df = fluctFrame.copy()


  # Stack frame for overlapping plots
  stackedFrame = df.stack().reset_index()
  # Give more appropriate names
  stackedFrame.columns = stackedFrame.columns.tolist()[:-2] + [
    "value_type", "value"
  ]

  # Select specified data
  query = "{0} or {1}".format(*[
    "value_type == '{0}_{1}'".format(typeKey, valueKey)
    for typeKey in ["std", "avg"]
  ])
  stackedFrame.query(query, inplace=True)

  # Beautify (avg_obs -> AVG(obs))
  stackedFrame["value_type"] = stackedFrame.apply(
    lambda col: re.sub(
      "^std_", "STD(", re.sub("^avg_", "AVG(", col["value_type"])
    ) + ")",
    axis=1
  )

  # Figure out which columns index the labels
  labelColumns = stackedFrame.columns.tolist()
  labelColumns.remove(axisKey)
  labelColumns.remove('value')

  # Create combined label for overlapping plot
  stackedFrame["info"] = stackedFrame.apply(
    lambda row: ", ".join([
      "{key}={val}".format(key=key, val=row[key])for key in labelColumns
      if key != "value_type"
    ][::-1]),
    axis=1
  )

  # Create the figure
  fig, ax = plt.subplots(figsize=(3, 3./50*len(stackedFrame)), dpi=300)

  # Plot the average values
  sns.set_color_codes("pastel")
  graph = sns.barplot(
    x="value",
    y=axisKey,
    hue="info",
    data=stackedFrame[stackedFrame.value_type.str.startswith("AVG")],
    alpha=0.9,
    ax=ax,
    orient="h",
  )

  # Plot the standard deviations
  sns.set_color_codes("muted")
  sns.barplot(
    x="value",
    y=axisKey,
    hue="info",
    data=stackedFrame[stackedFrame.value_type.str.startswith("STD")],
    alpha=0.5,
    ax=ax,
    orient="h",
    edgecolor="black",
    linewidth=1,
  )

  # We need to create a custom legend. Thus we want to store some infos
  legend = {}
  leg = graph.get_legend()
  for patch, txt in zip(leg.get_patches(), leg.get_texts()):
    legend[txt.get_text()] = patch.get_facecolor()

  # Remove old legend
  ax.legend_.remove()

  # Now create blank fillers for the custom legend
  for key, val in legend.items():
    legend[key] = ax.fill_between([np.NaN], [np.NaN], color=val)

  # The STD patch
  legend["AVG"] = ax.fill_between(
    [np.NaN], [np.NaN],
    alpha=0.9,
    color="gray",
  )

  # The AVG patch
  legend["STD"] = ax.fill_between(
    [np.NaN], [np.NaN],
    alpha=0.5,
    facecolor="gray",
    edgecolor="black",
    linewidth=1,
  )

  # And create the new legend
  plt.legend(
    list(legend.values()),
    list(legend.keys()),
    loc="upper left",
    frameon=True,
    fontsize="xx-small",
    bbox_to_anchor=(1.01, 1.0)
  )

  # Remove axis
  ax.axvline(0, color="black")
  sns.despine(left=True)
  ax.tick_params(axis="y", which="both", left="off", right="off")

  # set title
  ax.set_title("Fluctuations of %s" % valueKey)

  # And return the figure
  return fig
