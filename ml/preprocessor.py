#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 18:29:40 2018

@author: Jared
"""

# my imports
import plotter

# scikit imports
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn_pandas import DataFrameMapper
# normal imports
import numpy as np
from ast import literal_eval
import matplotlib.pyplot as plt
import pandas as pd

# scales
def preprocessExtScales(X, final_columns, model):
    
    Xt = model.transform(X) 
    X = pd.DataFrame(Xt,index=X.index, columns=X.columns)
        
    return(X[final_columns])
    
def preprocessExtBinTransform(X, userRange, pList):

    return transform_p_bin(X, userRange, pList)

def preprocess(X, userRange, pList, vthreshold, y, sfm_threshold,
               **kwargs):
    
    #plotter.plotFeature(X, 3, str(len(X.columns)) + ' Features (Raw)') 

    # TRANSFORM p_ VALUES TO NEW RANGE
    X = transform_p_bin(X, userRange, pList)

    # NORMALIZATION
    scaler = MaxAbsScaler(copy=True)  
    model = scaler.fit(X)
    Xt = model.transform(X)
    
    X = pd.DataFrame(Xt,index=X.index, columns=X.columns)
    
    # REDUCE BASED ON VARIANCE    
    try:
        X = VarianceThreshold_selector(X, vthreshold)  
    except:
        return
    
    Xnew = X.copy()  
    clf = ElasticNetCV(n_jobs = -1, cv=3, tol=0.001)
    sfm = SelectFromModel(clf, threshold = sfm_threshold) 
        
    sfm.fit(Xnew, y)
    feature_idx = sfm.get_support()
    X = X[X.columns[feature_idx]].copy()        

    return(X, {'final_columns' : X.copy().columns.tolist(),
               'model'         : model})

def VarianceThreshold_selector(data, threshold):
    
    selector = VarianceThreshold(threshold)
    selector.fit_transform(data)

    return data[data.columns[selector.get_support(indices=True)]] 

def transform_p_bin(df_in, histBins, pList):
    
    # parse out the additional features that wont go through bin transform   
    df_other = df_in[[i for i in df_in.columns if 'p_' not in i]]
    df_other = df_other.drop(['index'], axis = 1)

    super_out = []
    for i, p in enumerate(pList):
        
        # GET PROPERTY SUBSET OF DATAFRAME    
        #df = df_in[[j for j in df_in.columns if p == j.split('_')[-1][1:-1]]].copy()
        
        # Use for old 599 data and new 
        df = df_in[[j for j in df_in.columns if p == j.split('_')[-1]]].copy()
        
        if(len(df.columns)) == 0: continue
        # GET ALL START RADIAL DATA FROM df_in
        c = [literal_eval(name.split('_')[1] + '.' + name.split('_')[2])
             for name in df.columns]
       
        # BIN START DATA, USE AS MAP FOR SUMMING DATAFRAME COLUMNS     
        d = np.digitize(c, bins = histBins) # INDICES FOR BINS
        
        # GROUP df BY d MAP
        groups = df.groupby(d, axis = 1)   
        
        for key, item in groups: 
            summedGroups = pd.DataFrame(item.sum(axis = 1))        
            summedGroups.columns = [item.columns[0]]
            super_out.append(summedGroups)
       
    if(len(super_out) != 0):
        df_out = pd.concat(super_out, axis=1)  
        #if there are extra features we wish to include
        result = pd.concat([df_out, df_other], axis=1, join='inner')
    else: 
        result = df_other
    
    return(result)
    
#
#
# Initial discarding of features
def applyDFFilter(dff, **kwargs):
    

    if (kwargs['filterType'] == 'primary'):
        

        X = dff.drop([x for x in list(dff) if "_id" in x] + 
                     [x for x in list(dff) if "e_"  in x] + 
                     [x for x in list(dff) if "frac" in x] +
                     #[x for x in list(dff) if "p_" in x] +
                     #[x for x in list(dff) if "log" in x] +
                     ['total_energy', 'dir_gap', 'ind_gap', 
                      'dH_formation', 
                      'counts', 'la', 'lb', 'lc', 'em_hole', 'em_electron',
                      'volume', 
                      ],
                     axis=1)
        
    if (kwargs['filterType'] == 'old_primary'):
        

        X = dff.drop([x for x in list(dff) if "_id" in x] + 
                     [x for x in list(dff) if "e_"  in x] + 
                     [x for x in list(dff) if "frac" in x] +
                     #[x for x in list(dff) if "p_" in x] +
                     [x for x in list(dff) if "em" in x] +
                     ['total_energy', 'dir_gap', 'ind_gap', 
                      'dH_formation', 
                      'counts', 'la', 'lb', 'lc',
                      'volume', 
                      ],
                     axis=1)
        
    return X
 