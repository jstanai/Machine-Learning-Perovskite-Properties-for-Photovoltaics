#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jared
"""

import numpy as np
from itertools import product
import matplotlib


matplotlib.rcParams.update({'font.size': 15.5})

suppressWarnings = True

"""
i/o

    Specify the path to csv files. featurePath is the output of 
    mongoDB_management, while featurePathExt is an optional path to an 
    external test set for model evaluation. dataCubeOutput controls the output 
    location of the model selection data.
"""

projectPath = './data'
featurePath = projectPath + '/features/d2_paper/d2_paper_24102018.csv'
featurePathExt = projectPath + '/features/dummy_paper5000_1per_26102018.csv'
dataCubeOutput = 'outputs/result.csv'
deltaH_qmpyFile = '/hull/compEnergy_qmdb_d3.csv'

"""
Feature Construction

    Controls the properties computed in the PDDF, and the minimum resolution 
    (dr) and maximum considered neighborhood (rmax). Models cannot have a 
    smaller dr or larger rmax without first computing a high-resolution feature
    database. This high-resolution data is rebinned based on the model 
    parameters. 
"""


properties = ['X', 'EI1', 'rp', 'rs', 'ra', 'EI2', 'l', 
              'h', 'fermi', 'ir', 'socc', 'pocc']

dr = 0.2
rmax = 15.1 

"""
Model Parameters

    This controls parameters of the machine-learning model. Gamma is the kernel
    hyper-parameter, while alpha is the regularization hyper-parameter. The 
    cube implements the space of models to consider. splitNumber is the number
    of test/train splits to perform for validation, while testPercent controls
    the fraction of data that should be used for validation in this split.
    
    Mode controls what kind of analysis should be performed:
        'default'   - performs analysis of featurePath data
        'ext'       - performs analysis of external dataset using model trained
                      on featurePath data
        'split-set' - performs analysis of featurePath data containing element
                      leaveOut using a model trained on featurePath data which
                      does not contain element leaveOut. Must specify leaveOut,
                      e.g. leaveOut = 'Cs'
                      
    targetLabel controls what the prediction column should be.
"""


mode = 'ext'
leaveOut = ['Cs'] 
includeHull = False
targetLabel = 'dir_gap'
splitNumber = 20
testPercent = 0.1

#gammaRange = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]) 
#alphaRange = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]) 

gammaRange = 1/(10**np.arange(4, 11))
alphaRange = gammaRange
 
# MODELS TO RUN M = (dr, cv, v_cutoff, sfm, rmax)
# DEFAULT MODEL (M_BEST FOR BANDGAP)
#   [0.2, 8, 0.001, 0.1, 15.0]


#cube = list(product(
#        np.array([0.2, 1.2, 2.2]), #dr
#        np.array([4, 8]),       # cv cross-fold-validation folds (int)
#        np.array([2e-2, 1e-2, 1e-3, 0.0]),  #v_cutoff variance threshold
#        np.array([0.4, 0.2, 0.1, 0.01, 0.0]), # sfm_threshold 
#        np.array([15.0, 10.0, 5.0]) #rmax
#        ))

cube = [[0.3, 4, 0.001, 0.01, 15.0]] 









