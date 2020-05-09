#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 18:40:15 2018

@author: Jared
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
import myConfig
import pandas as pd
from ast import literal_eval
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import math
from pandas.plotting import scatter_matrix
import seaborn as sns
from scipy.optimize import curve_fit

def dbExamine(x, y):
    plt.scatter(x, y)
    
    plt.show()
    
def dfScatter(df):
    scatter_matrix(df, alpha=0.2, 
                   figsize=(6, 6), diagonal='kde')
    
    plt.show()
    

#group is of a group in of aggregated feature 
def getConvergencePlot(group, group2, baseEfix, baseK, title=None, savetitle = None,
                     **kwds):
    
    path = '/Users/Jared/Dropbox/Master Thesis/code/codeOutputs/'
    group3 = group.append(group2)

    #group is the better one here
    best = group[(group['e_densityMeshCutoff'] == 200. )
                   & (group['e_kb'] == 32)]
    
    
    
    choice = group[(group['e_densityMeshCutoff'] == 200. )
                   & (group['e_kb'] == 12)]
    
    bestGap = best.iloc[0]['dir_gap']
    
    choiceGap = choice.iloc[0]['dir_gap']
    
    print('Best: ', bestGap)
    print('Choice: ', choiceGap)
    print('abs ', bestGap - choiceGap)
    err = 100*abs(bestGap - choiceGap)/bestGap
    print('Error % of Best and Choice: ', err)

    
    mi = min(group3['dir_gap'])
    ma = max(group3['dir_gap'])

    
    #fig, ax = plt.subplots()
    #cm = plt.cm.get_cmap('RdYlBu')
    cm = plt.cm.get_cmap('viridis')
    
    #down triangles
    sc1 = plt.scatter(group['e_densityMeshCutoff'], 
                     [int(i) for i in group['e_kb']], 
                     c = group['dir_gap'], s = 600, 
                     marker = [(-1, -1), (-1, 1), (1, -1)],
                     vmin=(mi), vmax=(ma), cmap=cm)
    
    #up traingles
    sc2 = plt.scatter(group2['e_densityMeshCutoff'], 
                     [int(i) for i in group2['e_kb']], 
                     c = group2['dir_gap'], s = 600, 
                     marker = [(1,-1), (1, 1), (-1, 1)], 
                     vmin=(mi), vmax=(ma), cmap=cm)
    
     
    ax = plt.axes()
    ax.set_title(title)
    ax.tick_params(direction='in', top=True, right=True)
    #ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))
    #ax.set_yticklabels([12, 16, 20, 24, 28, 32])
    plt.yticks([4, 8, 12, 16, 20, 24, 28, 32])
    plt.xticks([50,75, 100, 125, 150, 175, 200, 225, 250])
    plt.ylabel('$k_{b}$ Value')
    plt.xlabel('Energy (Hartree)')
    #plt.title(title)
    #txt = 
    '''
    for i, txt in enumerate(group['dir_gap']):
        ax.annotate(round(txt, 2), 
                    xy = (group['e_densityMeshCutoff'][i], group['e_kb'][i]),
                    xytext=(0, .2),
                    textcoords='offset points'
                    )
    '''
    txs = np.linspace(mi, ma, num = 11)
    plt.colorbar(sc2, ticks = txs, format='%.3f', label = 'Bandgap (eV)')
 
    
    #plt.scatter(group['e_densityMeshCutoff'], group['e_kb'], 
    #            c = group['dir_gap'], s = 500)
    
    '''
    fig = plt.figure(figsize = (7,4))
    ax1 = fig.add_subplot(111)
    #ax1.set_title(A_t + B_t + C_t + ' Convergence', y=1.16)
    ax1.set_title(title[0], y=title[1])
    
    group2 = group[group['e_densityMeshCutoff'] == baseEfix].sort_values(by=['e_kb'])
    
    print(group)
    
    
    
    
    plt.scatter(range(len(group2)), group2['dir_gap'], color = 'orange')
    l1, = plt.plot(range(len(group2)), group2['dir_gap'], 
            color = 'orange', linestyle = '-',
            label = 'K-points')
    plt.xlabel('K-point Grid')
    plt.ylabel('Bandgap (eV)')
    #plt.xticks(range(len(group2)), ['3x6x3', '6x12x6', '10x20x10', 
    #       '12x24x12', '16x32x16'])
        
    ax2 = ax1.twiny()
    group3 = group[group['e_kb'] == baseK].sort_values(by=['e_densityMeshCutoff'])
    plt.scatter(range(len(group3)), group3['dir_gap'])
    l2, = plt.plot(range(len(group3)), group3['dir_gap'], 
                 linestyle = '--', label = 'Energy Cutoff')
    plt.xlabel('Density Mesh Cutoff Energy (Hartree)')
    plt.xticks(range(len(group3)), group3['e_densityMeshCutoff'])
        
    plt.legend(handles=[l1, l2])
    '''
    plt.savefig(path + title + '.png', pad_inches=1.2, dpi=500)
    plt.show()
    

# SHOW SOME PLOTS OF FEATURES
def plotFeature(X, i, title):
     
    savePath = '/Users/Jared/Dropbox/Master Thesis/code/codeOutputs/'
    
    print('BEFORE')
    plt.plot(range(0, len(X.loc[i])), X.loc[i])
    plt.title(title)
    plt.ylabel('Density in undetermined units')
    plt.xlabel('Index')
    plt.savefig(savePath + title + str(i) + '.png', pad_inches=0.2, dpi=600)
    plt.show()
    
def getMLPlot(train_y, test_y, y_predict_train, y_predict_test, flag, 
              nFeatures):
    
    flag = True
    train_y = 1000*train_y
    test_y = 1000*test_y
    y_predict_train = 1000*y_predict_train
    y_predict_test = 1000*y_predict_test
    
    if flag == True:

        savePath = '/Users/Jared/Dropbox/Master Thesis/code/codeOutputs/'
        
        my_dpi = 500
        fig = plt.figure(figsize=(5, 5), dpi=my_dpi)
        #ymax = 3.2
        #xmax = ymax
        ymin = 1.08*min(train_y) if min(train_y) <=0 else 0.92*min(train_y)
        ymax = 1.08*max(train_y) if max(train_y) >=0 else 0.92*max(train_y)
        xmax = ymax


        plt.ylabel('$E_{g}$ Prediction (eV)')
        plt.xlabel('$E_{g}$ (eV)')
        plt.title('Bandgap Prediction', y=1.04)

        plt.ylabel('$\Delta E_{hull}$ Prediction (meV/atom)')
        plt.xlabel('$\Delta E_{hull}$ (meV/atom)')
        plt.title('$\Delta E_{hull}$ Prediction', y=1.04)
        
        #plt.ylabel('$\Delta H_{f}$ Prediction (eV/atom)')
        #plt.xlabel('$\Delta H_{f}$ (eV/atom)')
        #plt.title('$\Delta H_{f}$ Prediction', y=1.04)
        
        plt.ylim(ymin, ymax)
        plt.xlim(ymin, xmax)
        
        xy = [ymin, ymax] #np.arange(ymin, ymax, .0005) 
        ax = plt.axes()
        #ax.grid()
        #plt.xticks(np.arange(ymin, ymax, 0.25))
        #plt.yticks(np.arange(round(ymin, 2), 
        #                     round(ymax + 0.1, 2)  + 0.25, 0.25))
        ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        ax.yaxis.set_major_locator(plt.MaxNLocator(6))
        
        ax.tick_params(direction='in', top=True, right=True)
        
        p0 = plt.plot(xy, xy, 'k', zorder=1)
        p1 = plt.scatter(train_y, y_predict_train, color='#00ccff',
                    marker='o', s=80, label='Train', zorder=2)
        p2 = plt.scatter(test_y, y_predict_test, color = '#ffb31a', 
                    marker='o', s=80, label = 'Test', zorder =3)
        #plt.legend(['Train', 'Test'])
        plt.legend(handles=[p1, p2])
        
        #plt.plot(xy, xy, 'k', alpha = 1.0)
        
        plt.savefig(savePath + 'paper_ML_example_d2_junk.png', 
                    bbox_inches="tight")
        flag = False
        #plt.savefig(savePath + 'ML_example_d1.png')
        plt.show()
 
def fitFunc(x, a, b):
    return a * x + b
    
def getTrendPlot1(dffExt, y_predict_ext, s,
                  ylabel, xlabel, title, scatter = True):
    
    #s = 'volume'
    #s = 'la'

    #y_predict_ext = pd.DataFrame(data = list(y_predict_ext), 
    #                             columns = ['dir_gap_fit'])
    
    #dffExt = dffExt.reset_index()
    #dffExt = dffExt.merge(y_predict_ext,
    #                      left_index=True, right_index = True)
    
    #dffExt = dffExt.join(y_predict_ext,
    #                     how = 'outer')
    

    
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    
    #print(dffExt.groupby([s]).size())
    y_cl = dffExt.groupby([s])['yhat_ext'].mean()
    x_cl = np.array([i for i in y_cl.index])
    #y_cl = y_cl.values
    
    x_cl_new = []
    y_cl_new = []
    for xi, x in enumerate(x_cl):
        if (x <= 962.1 and x >= 950):
            x_cl_new = x_cl_new + [x]
            y_cl_new = y_cl_new + [y_cl.values[xi]]

    dffExt = dffExt.sort_values(s)
    
    if scatter:
        plt.scatter(dffExt[s], 
                    dffExt['yhat_ext'],
                    marker = 'o',
                    alpha = 0.2)
    

    #p1, = plt.plot(x_cl, y_cl, linestyle = '-', lw = 2)
    bins = 25
    n, _ = np.histogram(dffExt[s], bins=bins)
    sy, _ = np.histogram(dffExt[s], bins=bins, weights=dffExt['yhat_ext'])
    mean = sy / n
    
    mean_x = (_[1:] + _[:-1])/2
    
    drop = ~np.isnan(mean)
    mean = mean[drop]
    mean_x = mean_x[drop]
    
    #print(np.nan_to_num(mean))
    #print((_[1:] + _[:-1])/2)
    plt.plot(mean_x, mean, '--', lw = 2, alpha = 1.0)
    '''
    
    ax1 = plt.axes()
    ax1.tick_params(direction='in', top=True, right=True)
    
    x_cl = x_cl_new
    y_cl = y_cl_new
    
    
    popt1, pcov1 = curve_fit(fitFunc, x_cl, y_cl)
    #x = np.linspace(min(x_cl), max(x_cl), 100)
    #plt.plot(x, fitFunc(x, *popt1), 'r--', lw = 4)
    x = np.linspace(800, 1500, 100)
    plt.plot(x, fitFunc(x, *popt1), 'r', lw = 2)
    
    plt.ylim(0, 2.0)
    plt.xlim(min(0.98*dffExt[s]), 1.01*max(dffExt[s]))
    
    #ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    ax1.yaxis.set_major_locator(plt.MaxNLocator(5))
    
    print(fitFunc(962.1, *popt1))
    print('**** The slope (meV): ', 1000*popt1[0])
    
    #1% frac M = [0.2, 8, 0.001, 0.1, 15.0] dummy_paper5000_1per_26102018junk
    slopes1 = np.array([8.01, 6.91, 8.01, 6.91, 5.96, 
                        8.01, 8.01, 8.01, 5.96, 8.01, 
                        8.01, 8.01, 8.01, 8.01, 5.96,
                        8.01, 8.01, 8.01, 8.01, 8.01,
                        8.01, 8.01, 8.01, 8.01, 8.01])
    slopes1_std = slopes1.std()
    slopes1_mean = slopes1.mean()
    print(slopes1_std, slopes1_mean)
    
    #0% frac M = [0.2, 8, 0.001, 0.1, 15.0]
    slopes1 = np.array([])
    slopes1_std = slopes1.std()
    slopes1_mean = slopes1.mean()
    print(slopes1_std, slopes1_mean)
    '''
    #plt.legend(handles = [p1])
    
    plt.tight_layout()
    #path = '/Users/Jared/Dropbox/Master Thesis/code/codeOutputs/'
    #plt.savefig(path + 'paper_dummyTrend5000_0per_volumeCsSnI3.png', dpi = 400, bbox_inches="tight")
    
    #plt.show()        
  
def getLearningCurve(train_sizes, train_scores, test_scores):
    
    savePath = '/Users/Jared/Dropbox/Master Thesis/code/codeOutputs/'
    
    my_dpi = 500
    fig = plt.figure(figsize=(5, 5), dpi=my_dpi)
    
    plt.title('Learning Curves', y = 1.04)
    plt.xlabel('$N_{subset}$')
    plt.ylabel('Score')
    
    ax = plt.axes()
    ax.tick_params(direction='in', top=True, right=True)
    print(train_sizes)
    x1 = train_sizes
    y1 = np.mean(train_scores, axis=1)
    y1s = np.std(train_scores, axis=1)
    
    x2 = train_sizes
    y2 = np.mean(test_scores, axis=1)
    y2s = np.std(test_scores, axis=1)
    
    plt.plot(x1, y1, '.b-',label = 'training')
    
    plt.fill_between(x1, y1-y1s, y1+y1s, alpha = 0.2, color = 'b', 
                     linewidth=0.0)
    
    plt.plot(x2, y2, '.r-', label = 'testing')
    
    plt.fill_between(x2, y2-y2s, y2+y2s, alpha = 0.2, color = 'r', 
                     linewidth=0.0)
    
    plt.legend(loc = 4)
    
    plt.tight_layout()
    plt.savefig(savePath + 'paper_learnCurve_d2.png', pad_inches=0.2, dpi=500)
    plt.show()

def getMLPlotExt(y_ext, y_predict_ext, X_ext, flag, 
              nFeatures):
    
    dffExt = pd.read_csv(myConfig.featurePathExt)
    dffExt = dffExt.copy().dropna(axis=0, how='any').reset_index()
    
    getTrendPlot1(dffExt, y_predict_ext, 'fracNa')
    
    
    # Pb bandgap graph
    flagComp = False
    if flagComp == True:
        
        #ctuple = [c[k] for k in key]
        compounds = []
        for index, row in dffExt.iterrows():
            #row['counts']
            
            counter = row['counts']
            counter = counter.split('(')[1].split(')')[0]
            counter = literal_eval(counter)
            
            A_t = ''
            B_t = ''
            C_t = ''
            for i in counter:   
            
                if i == 'Cs' or i == 'Rb' or i == 'Na' or i == 'K':
                    A = i
                    Ai = str(counter[i])
                    if counter[i] == 1: Ai = ''
                    A_t += A + '$_{' + Ai + '}$'
                    
                if i == 'Sn' or i == 'Ge' or i == 'Pb':
                    B = i
                    Bi = str(counter[i])
                    if counter[i] == 1: Bi = ''
                    B_t += B + '$_{' + Bi + '}$'
                if i == 'Br' or i == 'Cl' or i == 'I':
                    C = i
                    Ci = str(counter[i])
                    if counter[i] == 1: Ci = ''
                    C_t += C + '$_{' + Ci + '}$'
            
            compounds += [A_t + B_t + C_t]
    
    
        fig, ax1 = plt.subplots()
        plt.title('Pb Bandgap Predictions by Composition')
        plt.ylabel('Bandgap (eV)')
        x = range(len(compounds))
        plt.xticks(x, compounds, rotation = 90)
        #plt.xlabel('CsSnI$_{3}$')
        fig.set_size_inches(8, 5)
        #ax1 = plt.axes()
        ax1.scatter(x, y_predict_ext, marker = 'o', color = 'xkcd:blue', label='Predicted')
        ax1.scatter(x, y_ext, marker = 'o', color = 'xkcd:red', label = 'DFT')
        plt.legend()
        ax1.plot(x[1:7], y_predict_ext[1:7], linestyle = '--', color = 'xkcd:blue')
        ax1.plot(x[1:7], y_ext[1:7], linestyle = '--', color = 'xkcd:red')
        
        ax1.tick_params(direction='in', top=True, right=True)
        ax1.yaxis.set_major_locator(plt.MaxNLocator(5))
        
        plt.tight_layout()
        path = '/Users/Jared/Dropbox/Master Thesis/code/codeOutputs/'
        
        
        plt.savefig(path + 'predictPb1.png', dpi = 400, bbox_inches="tight")
        #plt.show()
        #print(dffExt['counts'])
        
        #cols = [x for x in list(X_ext) if "frac" in x]
        #print(cols)
    
    flagComp = False
    if flagComp == True:

        savePath = '/Users/Jared/Dropbox/Master Thesis/code/codeOutputs/'
        #plt.scatter(y_ext, y_predict_ext, alpha=0.8, color='#00ccff',
        #            marker='o', s=80)
        
        my_dpi = 500
        fig = plt.figure(figsize=(5, 5), dpi=my_dpi)
        
        ymin = 1.08*min(y_ext) if min(y_ext) <=0 else 0.92*min(y_ext)
        ymax = 1.08*max(y_ext) if max(y_ext) >=0 else 0.92*max(y_ext)
        xmax = ymax

        print(len(y_ext))
        plt.ylabel('$E_{g}$ Prediction (eV)')
        plt.xlabel('$E_{g}$ (eV)')
        plt.title('Bandgap Prediction', y=1.04)
        
        bandgapCs = np.array([0.213, ])
        #plt.ylabel('$\Delta H_{f}$ Prediction (eV/atom)')
        #plt.xlabel('$\Delta H_{f}$ (eV/atom)')
        #plt.title('$\Delta H_{f}$ Prediction', y=1.04)
        
        #plt.ylabel('ML Prediction (eV)')
        #plt.xlabel('Bandgap (eV)')
        #err = round(sqrt(mean_squared_error(y_ext, y_predict_ext)),3)
        #plt.title('Bandgap Prediction')
        
        
        plt.ylim(ymin, ymax)
        plt.xlim(ymin, xmax)
        
        #plt.legend(['Train', 'Test'])
        
        
        plt.ylim(ymin, ymax)
        plt.xlim(ymin, xmax)
        

        ax = plt.axes()
        xy = [ymin, ymax]
        #ax.grid()
        #plt.xticks(np.arange(ymin, ymax, 0.25))
        #plt.yticks(np.arange(round(ymin, 2), 
        #                     round(ymax + 0.1, 2)  + 0.25, 0.25))
        #ax.xaxis.set_major_locator(plt.MaxNLocator(12))
        #ax.yaxis.set_major_locator(plt.MaxNLocator(12))
        ax.tick_params(direction='in', top=True, right=True)
        ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        ax.yaxis.set_major_locator(plt.MaxNLocator(6))
        p0 = plt.plot(xy, xy, 'k', zorder=1)
        p1 = plt.scatter(y_ext, y_predict_ext, color='#00ccff',
                    marker='o', s=80, label='Train', zorder=2)

        #plt.legend(['Train', 'Test'])
        #plt.legend(handles=[p1])

        #xy = np.arange(0, 3.2, .005)
        #plt.plot(xy, xy, 'k', alpha = 0.75)
        #plt.show()
        plt.savefig(savePath + 'paper_predict_noCsnoRb_form.png', dpi=500, bbox_inches="tight")
    
        
def getGaussPlotExt(d):

    e = pd.read_csv('/Users/Jared/Dropbox/Master Thesis/' + 
          'Data/ExternalFeaturesDB/ext_tmp.csv')
    #x = np.random.normal(size=100)
    
    print(type(d))
    

    elist = (e['dir_gap'].values)#/12.0
    print(elist, type(elist))
    

    ax = sns.jointplot(d, elist, kind="kde")
    ax.set_axis_labels("Volume Predicted", "Volume Set")
    #ax.set_ylim(0,1)

    plt.show()
    #sns.distplot(d);     
                 
def parallel_coordinates(frame, class_column, cols=None, ax=None, color=None,
                     use_columns=False, xticks=None, colormap=None, 
                     title = None, cblabel = None, savetitle = None,
                     **kwds):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib import ticker
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    n = len(frame)
    class_col = frame[class_column]
    class_min = np.amin(class_col)
    class_max = np.amax(class_col)

    if cols is None:
        df = frame.drop(class_column, axis=1)
    else:
        df = frame[cols]

    #used_legends = set([])
    fig = plt.figure(1, figsize=(9,4.5))
    ncols = len(df.columns)
    Colorm = plt.get_cmap(colormap)

    #gs0 = gridspec.GridSpec(1, 2, height_ratios=[1], 
    #                       width_ratios=[0.9, 0.1],)
    #gs0.update(left=0.05, right=0.95, bottom=0.08, top=0.93, 
    #           wspace=0.0, hspace=0.03)

    #gs1 = GridSpec(3, 3)
    #gs1.update(left=0.05, right=0.48, wspace=0.05)
#gs00 = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs0[0])

    gs = gridspec.GridSpec(1,ncols - 1, 
                           height_ratios=[1], 
                           width_ratios=[1]*(ncols - 1))
    gs.update(left=0.05, right=0.85, wspace=0.0)
    
    gs_cb = gridspec.GridSpec(1,1, 
                           height_ratios=[1], 
                           width_ratios=[1])
    gs_cb.update(left=0.92, right=0.95)
    
    
    
    #fig, axes = plt.subplots(1, ncols - 1, sharey=False, figsize=(8,5))
    
    x = [i for i, _ in enumerate(df.columns)]
    
    if title is not None:    
        plt.suptitle(title, fontsize=16)
    
    min_max_range = {}
    cols = df.columns
    for col in cols:
        min_max_range[col] = [df[col].min(), df[col].max(), np.ptp(df[col])]
        df[col] = np.true_divide(df[col] - df[col].min(), np.ptp(df[col])) 
    
    for i in range(ncols - 1):
        ax = plt.subplot(gs[0,i])
        ax.set_ylim([-0.1,1.1])
        for idx in df.index:
            kls = class_col.iat[idx]
            
            ax.plot(x, df.loc[idx, cols], linewidth=2, alpha = 0.5,
                    color=Colorm((kls - class_min)/(class_max-class_min)))
            
        ax.set_xlim([x[i], x[i+1]])
        
        '''
        if i == (ncols - 1):
            ax = plt.twinx(ax)
            dim = ncols - 1
            ax.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
            set_ticks_for_axis(dim, ax, min_max_range, df, cols, ticks=6)
            ax.set_xticklabels([cols[-2], cols[-1]])
        '''
            

    #print(gs.get_subplot_params())
    for i in range(ncols - 1):    
        ax = plt.subplot(gs[0,i])               
    #for i, ax in enumerate(axes):
        ax.xaxis.set_major_locator(ticker.FixedLocator([i]))
        set_ticks_for_axis(i, ax, min_max_range, df, cols, ticks=6)
        ax.set_xticklabels([cols[i]])
        
    
    # Move the final axis' ticks to the right-hand side
    last_ax = plt.subplot(gs[0,ncols - 2])
    ax = plt.twinx(last_ax)
    #ax.set_ylim([-0.1,1.1])
    #ax = plt.twinx(axes[-1])
    #dim = len(axes)
    dim = ncols - 1
    ax.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
    set_ticks_for_axis(dim, ax, min_max_range, df, cols, ticks=6)
    ax.set_xticklabels([cols[-2], cols[-1]])
    
    
    # Remove space between subplots
    #plt.subplots_adjust(right = 0.75)
    

        
    #a = twiny(ax)
    #print(ax.get_aspect())
    #[0.85, 0.1, 0.03, 0.73]
    #divider = make_axes_locatable(axes)
    #cbaxes = divider.append_axes("right", size="5%", pad=0.45)
    #plt.colorbar(cax = cbaxes, cmap = Colorm)
    #cbaxes = fig.add_axes([0.85, 0.12, 0.03, 0.76]) 
   
    #cb = plt.colorbar(ax1, cax = cbaxes) 
    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes("right", size="5%", pad=0.35)
    #cbar_ax = fig.add_axes([0.85, 0.0, 0.05, 0.8])
    
    

    #ax = fig.add_subplot(231)
    bounds = np.linspace(class_min,class_max,10)
    #ax = plt.gca()
    
    
    
    #cax = plt.subplot(gs[0,ncols - 1])
    cax = plt.subplot(gs_cb[0,0])
    #pos1 = cax.get_position()
    #pos2 = [pos1.x0 + 0.1, pos1.y0,  pos1.width, pos1.height] 
    #cax.set_position(pos2)
    #cax,_ = mpl.colorbar.make_axes(ax)
 
    cb = mpl.colorbar.ColorbarBase(cax, cmap=Colorm, 
                                   spacing='proportional', ticks=bounds, 
                                   boundaries=bounds, format='%.2f', 
                                   label = cblabel)
    #cb.set_label(cblabel)
    plt.gcf().subplots_adjust(right=0.15)
    
    
    #plt.tight_layout(h_pad = 2)
    if savetitle is not None:
        plt.savefig(savetitle, dpi=400, bbox_inches="tight")
    plt.show()
    return fig


def set_ticks_for_axis(dim, ax, min_max_range, df, cols, ticks):
    min_val, max_val, val_range = min_max_range[cols[dim]]
    step = val_range / float(ticks-1)
    
    
    tick_labels = [round(min_val + step * i, 4) for i in range(ticks)]
    norm_min = df[cols[dim]].min()
    norm_range = np.ptp(df[cols[dim]])
    norm_step = norm_range / float(ticks-1)
    ticks = [round(norm_min + norm_step * i, 8) for i in range(ticks)]

    ax.set_ylim([-0.1,1.1])
    ax.yaxis.set_ticks(ticks)
    ax.set_yticklabels(tick_labels)
   
def varianceBias(df, arr, cmap, title = None, cbformat = None, cbmin = None,
                 cbmax = None, savetitle = None):
    
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib import ticker
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    cblabel = arr[2]
    
    df = df.sort_values(by = arr[0], ascending = False)
    #df = df.sort_values(by = 'num_features', ascending = False)
    
    class_col = df[arr[2]]
    
    if cbmin != None:    
        class_min = cbmin
    else:
        class_min = df[arr[2]].min()
        
    if cbmax != None:    
        class_max = cbmax
    else:
        class_max = df[arr[2]].max()
    
    print(class_max)
    
    #fig = plt.figure(1, figsize=(5,5))
    my_dpi = 500
    fig = plt.figure(figsize=(7, 5), dpi=my_dpi)
    
    Colorm = plt.get_cmap(cmap)
    xl = range(len(df))
    #xl = df['num_features'] #range(len(df))
    
    
    gs = gridspec.GridSpec(1,1, 
                           height_ratios=[1], 
                           width_ratios=[1])
    gs.update(left=0.15, right=0.79, wspace=0.0)
    
    gs_cb = gridspec.GridSpec(1,1, 
                           height_ratios=[1], 
                           width_ratios=[1])
    gs_cb.update(left=0.82, right=0.85)
    
    
    ax = plt.subplot(gs[0,0])
    ax.tick_params(direction='in', top=True, right=True)
    
    for j, idx in enumerate(df.index):
        kls = class_col.iat[idx]
        
        c = (kls - class_min)/(class_max-class_min)
        
        c = 1.00 if c > 1 else c
            
        
        ax.scatter(xl[j], df.loc[idx, arr[1]], alpha = 0.7, marker = 'o',
                    color=Colorm((kls - class_min)/(class_max-class_min)))
        ax.scatter(xl[j], df.loc[idx, arr[0]], alpha = 0.7, marker = '+',  
                    color=Colorm((kls - class_min)/(class_max-class_min)))
            
        #ax.set_xlim([x[i], x[i+1]])

    p1 = plt.scatter([], [], marker='o', color = 'k', label='Train')
    p2 = plt.scatter([], [], marker='+', color = 'k', label='Test')
    #ax.legend(handles=[p1, p2])

    row = df[df[arr[0]] == df[arr[0]].min()]
    s = 'M$_{best}$ = $(' + \
        str(row['v_cutoff'].values)[1:-1] + ', ' + \
        str(row['cv'].values)[1:-1] + ', ' + \
        str(row['dr'].values)[1:-1] + ', ' + \
        str(row['sfm_threshold'].values)[1:-1] + ', ' + \
        str(row['rmax'].values)[1:-1] + ')^{T}$'
    

    if title is not None:    
        #plt.suptitle(title)
        ax.set_title(title, y=1.04)
        
    if cbformat is not None:  
        cbformat = cbformat
    else:
        cbformat = '%.4f'
        
    ax.legend(handles=[p1, p2])
    
    ax.set_xlabel('(Sorted) Model Index')
    ax.set_ylabel('RMSE Bandgap (eV)')
    bounds = np.linspace(class_min,class_max,10)
    cax = plt.subplot(gs_cb[0,0])
    cb = mpl.colorbar.ColorbarBase(cax, cmap=Colorm, 
                                   spacing='proportional', ticks=bounds, 
                                   boundaries=bounds, format = cbformat, 
                                   #label = cblabel, 
                                   label = 'Number of Features',
                                   extend='max')
    
    
    
    
        
    

    #+ ', ' + str(row['dr']) + \
    #    ', ' + str(row['sfm_threshold']) + ', ' + str(row['rmax'])
        
    ax.annotate('$M_{best}$', xy=(len(df) - 1, 
                df[arr[0]].min()), 
                xytext=(0.82*len(df), 1.45*df[arr[0]].min()),
                arrowprops=dict(facecolor='black', 
                                #shrink=0.005, 
                                arrowstyle="->"),
                )
                
    if savetitle is not None:
        plt.savefig(savetitle, bbox_inches="tight")
        #plt.savefig(savetitle)

    plt.show()
    
    '''
    #x = complexityMeasure
    xl = range(len(df))
    plt.scatter(xl, df[arr[1]], color = cmap, label = arr[1])
    plt.scatter(xl, df[arr[0]], color = cmap, label = arr[0])
    #plt.plot(xl, test, linestyle = '--', color = 'r')
    #plt.plot(df[arr[1]], color = 'b')
    #plt.plot(xl, train, linestyle = '--', color = 'b')
    plt.legend()
    plt.show()
    '''
    

# GEt correlation stuff
'''
X_corr = pd.read_csv('/Users/Jared/Dropbox/Master Thesis/code/codeOutputs/X_corr.csv', index_col = 0)
print(len(X_corr))
    
    #plt.matshow(X_corr)
    #X_corr.fillna(value=np.nan, inplace=True)
    
ax = sns.heatmap(X_corr, robust = True)
    
    # turn the axis label
#for item in ax.get_yticklabels():
#        item.set_rotation(0)
    
#for item in ax.get_xticklabels():
#        item.set_rotation(90)
plt.savefig('/Users/Jared/Dropbox/Master Thesis/code/codeOutputs/seabornPandas.png', dpi=400)
plt.show()
'''