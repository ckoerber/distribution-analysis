#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests the support module for visualizing information about statistic ensembles.
"""
__version__ = 0.1
__author__ = "Christopher Körber"

import unittest
import numpy as np
import distalysis.utilities as ut

#===============================================================================
#     Extension of core abstract test
#===============================================================================
class TestGeneratePseudoSamples(unittest.TestCase):
  """Test class for 'distalysis.utilities.generatePseudoSamples' method.
  """

  #----------
  def __init__(self, *args, **kwargs):
    """Initializes the test class with given runtime parameters.
    """
    # Initialize the base classes
    super(TestGeneratePseudoSamples, self).__init__(*args, **kwargs)

    # Parameters
    self.nX = 32
    self.nSamples = 100

    # x-Range
    self.xRange = np.arange(self.nX)

    # Exponential parameters
    self.expPars = np.array([
      [(+1./self.nX, 0.5), (+4./self.nX, 1.0)],
      [(-1./self.nX, 1.5), (-4./self.nX, 4.0)],
    ])
    self.nC = self.expPars.shape[0]

    # Generate base samples
    self.samples = ut.generatePseudoSamples(
      self.xRange,
      self.nSamples,
      self.expPars
    )

  #----------
  def testSampleShape(self):
    """
    Test the shape of the output sample array.
    """
    self.assertEqual((self.nC, self.nX, self.nSamples), self.samples.shape)

#===============================================================================
#     Tests
#===============================================================================
if __name__ == "__main__":
  unittest.main()
