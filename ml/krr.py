#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jared
"""

import sklearn.kernel_ridge as kr
import numpy as np
import pandas as pd
import plotter
from ml.errors import mean_relative_error as mre

from sklearn.metrics import mean_squared_error as mse
import myConfig 

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from ml.helpers import printProgressBar

def customKernelOption(X, Y, **kwargs):
    
    norm = np.linalg.norm((X - Y)) 
    
    return( np.exp(-kwargs['gamma']*norm**2) )
    
def getKernel(s, i, j, **kwargs):
    
    if s == 'custom':
        # Format: kernel_params = {'gamma' : j}
        return kr.KernelRidge(kernel = customKernelOption, 
                              alpha=i, kernel_params = kwargs['kernal_params'])
    else:
        return kr.KernelRidge(kernel = s, alpha=i, gamma = j)       

def get_trained_krr_model(X, y_labels, alphaRange, gammaRange, cv):
    
    scores_mean = []
    
    # Set Default Optimal Hyperparameters
    hyper_params_optimal = [gammaRange[0], alphaRange[0]]
    
    hyper_params = []
    best_score = 0
    
    for i in alphaRange:
        for j in gammaRange:            

            model = getKernel('rbf', i, j)
            scores = cross_val_score(model, X, y_labels, cv=cv)

            if scores.mean() >= best_score: 
                best_score = scores.mean()          #R^2 value           
                hyper_params_optimal = [j , i]
        
            hyper_params = hyper_params + [[j, i, 
                                            scores.mean(), 
                                            scores.std()*2]] # [*** why *2?]
            
            scores_mean = scores_mean + [scores.mean()]
    
    model = getKernel('rbf', hyper_params_optimal[1], hyper_params_optimal[0])   
    model.fit(X, y_labels)
    
    G = gammaRange.tolist()
    A = alphaRange.tolist()
    Z = scores_mean
    
    return({'model' : model, 
            'hyper_params_optimal' : hyper_params_optimal, 
            'scores' : scores_mean,
            'Z' : Z, 'G' : G, 'A' : A,
            'bestscore' : best_score})

def get_krr_performance(X, y_labels, alphaRange, gammaRange, 
                        num, cv, **kwargs):
    
    df_columns = ['rms_train', 
                  'rms_test', 
                  'rel_train',
                  'rel_test',
                  'bestscore', 
                  'best_gamma', 
                  'best_alpha']
    
    if ('X_ext' in kwargs) and ('y_ext' in kwargs):
        df_columns += ['rms_ext', 'rel_ext', 'yhat_ext']
    
    result = pd.DataFrame(index = range(num), columns = df_columns)
    
    for index, row in result.iterrows():
               
        printProgressBar(index + 1, num, prefix = '', 
                         suffix = str(index + 1) + '/' + str(num), 
                         decimals = 0, length = 50, 
                         fill = '#')

        train_x, test_x, train_y, test_y = train_test_split(
                                                X, y_labels, 
                                                test_size=myConfig.testPercent)
        
        model_data = get_trained_krr_model(train_x, train_y, 
                                           alphaRange, gammaRange, cv)
        
        yhat_test = model_data['model'].predict(test_x)
        yhat_train = model_data['model'].predict(train_x) 
        
        s = pd.Series({
                'rms_test'    : np.sqrt(mse(test_y, yhat_test)),
                'rms_train'   : np.sqrt(mse(train_y, yhat_train)),
                'rel_test'    : mre(test_y, yhat_test),
                'rel_train'   : mre(train_y, yhat_train),
                'bestscore'   : model_data['bestscore'],
                'best_gamma'  : model_data['hyper_params_optimal'][0],
                'best_alpha'  : model_data['hyper_params_optimal'][1],
                })
        
        if ('X_ext' in kwargs) and ('y_ext' in kwargs):
            
            yhat_ext = model_data['model'].predict(kwargs['X_ext'])
            s = s.append(pd.Series({
                'rms_ext'   : np.sqrt(mse(kwargs['y_ext'], yhat_ext)),
                'rel_ext'   : mre(kwargs['y_ext'], yhat_ext),
                'yhat_ext'  : np.array(yhat_ext)
                }))
    
        result.loc[index] = s
    
        if (kwargs['getExampleLearningCurve'] == True) and (index < 1):
            nset, train_scores, test_scores = learning_curve(
                                model_data['model'], 
                                X, y_labels, cv=cv, 
                                train_sizes=np.linspace(.1, 1.0, 20), 
                                verbose=0, scoring = 'r2')
            
            plotter.getLearningCurve(nset, train_scores, test_scores)
        
        if (kwargs['getExampleMLPlot'] == True) and (index < 1):
            
            plotter.getMLPlot(train_y, test_y, yhat_train, yhat_test, 
                              True, 
                              len(X.columns))
         
        #plotter.getGaussPlotExt(y_predict_ext)
        #plotter.getMLPlotExt(y_ext, yhat_ext, X_ext, flag, len(X_ext.columns))     
            
        if (kwargs['printErrorReports'] == True):
            print('Errors Report: ')
            print('  y Abs (test)', np.sqrt(mse(test_y, yhat_test)))
            print('  y Abs (train)', np.sqrt(mse(train_y, yhat_train)))
            print('  y Rel (test)', mre(test_y, yhat_test),'%')
            print('  y Rel (train)', mre(train_y, yhat_train),'%')
            
            if ('X_ext' in kwargs) and ('y_ext' in kwargs):   
                print('  y_ext Abs', np.sqrt(mse(kwargs['y_ext'], yhat_ext)))
                print('  y_ext Rel', mre(kwargs['y_ext'], yhat_ext),'%')
    
    return(result)
