#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jared
"""
import pandas as pd
import numpy as np
import myConfig
import ml.preprocessor as preprocessor
import ml.feature as feature
import time
import ml.krr as krr

import warnings
from ml.helpers import warn 

def getData(mode):
    
    # DEFAULT MODE:
    #   dff is split into training and dev-sets to determine model.
    #   no external test set is used. 
    if mode == 'default':
        dff = pd.read_csv(myConfig.featurePath)
        dff = dff.copy().dropna(axis=0, how='any').reset_index()
      
    # EXT MODE:
    #   dff is split into training and dev-sets to determine model.
    #   dffExt is used as external test set.  
    if mode == 'ext':
        dff = pd.read_csv(myConfig.featurePath)
        dff = dff.copy().dropna(axis=0, how='any').reset_index()
    
        dffExt = pd.read_csv(myConfig.featurePathExt)
        dffExt = dffExt.copy().dropna(axis=0, how='any').reset_index()
    
    # SPLIT-SET MODE:
    #   dff data is partioned into two subsets: 
    #       dff does not contain specified leaveOut element.
    #       dffExt test set contains specified leaveOut element
    if mode == 'split-set':
        
        leaveOut = myConfig.leaveOut
        dff = pd.read_csv(myConfig.featurePath)
        dffExt = pd.read_csv(myConfig.featurePath)
        
        for i in leaveOut:
            dff = dff[dff['frac' + leaveOut] == 0.0]
            dffExt = dffExt[dffExt['frac' + leaveOut] > 0.0]
        
        dff = dff.copy().dropna(axis=0, how='any').reset_index()
        dffExt = dffExt.copy().dropna(axis=0, how='any').reset_index()
        
        print('no ' + leaveOut, '- set size: ', len(dff))
        print('contains ', '- set size: ', len(dffExt))
      
    # CONVEX HULL INCLUSION:
    #   Pulls data from qmpy and merges on crystal_id. New thermodynamic 
    #   calcultions must be performed to include these calculations for new
    #   data. See stability.py to perform these new calculations and update
    #   thermodynamic data for crystals.
    if myConfig.includeHull == True:
        
        deltaH_qmpy = pd.read_csv(myConfig.deltaH_qmpyFile)
        
        # train-dev set merge
        dff = pd.merge(dff, deltaH_qmpy[['crystal_id', 'deltaH_hull']], 
                       on=['crystal_id'])
        
        dff['hull_distance'] = dff.apply(lambda x: 
            (x['dH_formation'] - x['deltaH_hull']), axis=1)
        
        
        # test set merge
        if mode != 'default':
            dffExt = pd.merge(dffExt, 
                              deltaH_qmpy[['crystal_id', 'deltaH_hull']], 
                              on=['crystal_id'])
            
            dffExt['hull_distance'] = dffExt.apply(lambda x: 
                (x['dH_formation'] - x['deltaH_hull']),
                axis=1)
        
    return dff, dffExt

def main():

    if myConfig.suppressWarnings == True:
        warnings.warn = warn
    
    #
    #
    # SETUP ML PARAMETERS
    gammaRange = myConfig.gammaRange
    alphaRange = myConfig.alphaRange  
    pList = myConfig.properties
    pList = feature.getPropertyMixerLabels(pList)
    cube = myConfig.cube
    num = myConfig.splitNumber 
    
    #
    #
    # EXTRACT DATA AND CONFIGURE TARGETS
    dff, dffExt = getData(myConfig.mode)
    
    y_ext = dffExt[myConfig.targetLabel]
    y_labels = dff[myConfig.targetLabel]
    
    X = preprocessor.applyDFFilter(dff, filterType='primary')
    X_ext_in = preprocessor.applyDFFilter(dffExt, filterType='primary')
    
    if myConfig.includeHull: 
        X = X.drop(['deltaH_hull', 'hull_distance'], axis = 1)
        X_ext_in = X_ext_in.drop(['deltaH_hull', 'hull_distance'], axis = 1)
    
    #
    #
    # PRE-ALLOCATE DATAFRAME OUTPUT
    dc_cols = ['rms_train', 'rms_test', 'rel_train', 'rel_test', 'bestscore',
               'best_gamma', 'best_alpha', 'dr', 'cv', 'v_cutoff', 
               'sfm_threshold', 'rmax', 'num_features']
    
    dc = pd.DataFrame(index = range(len(cube)), columns = dc_cols)
    
    #
    #
    # VERBOSE OUTPUT
    b = 70
    print('*'*b + '\n Trials: ' + str(myConfig.splitNumber) + 
          '    Hyper-parameter grid size: ' + str(
          len(myConfig.alphaRange)*len(myConfig.gammaRange)) +
          '\n' + '*'*b 
          )

    #
    #
    # COMPUTE FITS FOR EACH MODEL PARAMETER SET IN 'CUBE'
    for ii, param in enumerate(cube):
        
        start = time.time()
        print('\n' + '-'*b + '\n Experiment ' + str(ii + 1) + '/' + 
              str(len(cube)) + ': M =' + str(param) + '\n' + '-'*b)
        
        
        userRange = np.arange(0.001, param[4], param[0])
        
        
        try:
            X_c, extraData = preprocessor.preprocess(X, userRange, pList, 
                                                     param[2], 
                                                     y_labels, 
                                                     param[3],
                                                     verbose = False)
        except:
            print('  No features found for M = ', param)
            continue 
        print('  Post-processing feature Count: ', len(X_c.columns))
        
        
        if myConfig.mode != 'default':
        
            try:
                X_ext = preprocessor.preprocessExtBinTransform(X_ext_in, 
                                                               userRange, 
                                                               pList)
            except:
                print('  No ext features found for M = ', param)
                continue
            
            X_ext = preprocessor.preprocessExtScales(X_ext, 
                                                 extraData['final_columns'], 
                                                 extraData['model'])
             
            result = krr.get_krr_performance(X_c, y_labels, 
                                             alphaRange, gammaRange, 
                                             num, param[1],
                                             X_ext = X_ext,
                                             y_ext = y_ext,
                                             getExampleLearningCurve = False,
                                             getExampleMLPlot = False,
                                             printErrorReports = False)
        else: 
            result = krr.get_krr_performance(X_c, y_labels, 
                                             alphaRange, gammaRange, 
                                             num, param[1],
                                             getExampleLearningCurve = False,
                                             getExampleMLPlot = False,
                                             printErrorReports = False)
        
        # AVERAGE OVER TRIALS 
        result = result.mean(axis=0)
        
        # ADD MODEL META DATA
        result['dr']            = param[0]
        result['cv']            = param[1]
        result['v_cutoff']      = param[2]
        result['sfm_threshold'] = param[3]
        result['rmax']          = param[4]
        result['num_features']  = len(X_c.columns)
            
        dc.loc[ii] = result
        
        end = time.time()
        print('  Experiment Time: ', round(end - start, 2), 's')
    
    # SAVE OUTPUT AND PRINT RESULTS
    dc.to_csv(myConfig.dataCubeOutput)
    
    print('-'*b + '\n Results: ' + '\n' + '-'*b)
    print(dc.head())
    print('*'*b)
    
if __name__ == '__main__':
    main()