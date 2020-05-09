#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jared
"""
import numpy as np
import pandas as pd
import myConfig
import matplotlib.pyplot as plt
from ast import literal_eval
from plotter import getTrendPlot1
from matplotlib.pyplot import figure


df = pd.read_csv(myConfig.extOutput)

dffExt = pd.read_csv(myConfig.featurePathExt)
dffExt = dffExt.copy().dropna(axis=0, how='any').reset_index() 


y_predict_ext = df['yhat_ext']

print('Num dummy crystals: {}'.format(len(y_predict_ext)))
print([n for n in dffExt.columns if 'p_' not in n])

s = 'fracCl'

dffExt['yhat_ext'] = df['yhat_ext']

ylabel = '$E_{g}$ (eV)'



getTrendPlot1(dffExt, y_predict_ext, s,
                  ylabel = ylabel,
                  xlabel = s,
                  title = 'Trend')
plt.show()

'''
s = 'volume'
g = dffExt.groupby('fracCl')
for i, group in g:
    getTrendPlot1(group, y_predict_ext, s,
                  ylabel = ylabel,
                  xlabel = s,
                  title = 'Trend',
                  scatter = False)
plt.show()
'''
    
s = 'fracCs'
g = dffExt.groupby('fracSn')
for i, group in g:
    getTrendPlot1(group, y_predict_ext, s,
                  ylabel = ylabel,
                  xlabel = s,
                  title = 'Trend',
                  scatter = False)
plt.show()

'''
print(dffExt[['fracCs', 'fracRb', 'fracK', 'fracNa',
              'fracSn' , 'fracGe',
              'fracCl', 'fracI', 'fracBr', 'yhat_ext']].head(10))
'''
    
g = dffExt.groupby([
              'fracCs', 'fracRb', 'fracK', 'fracNa',
              'fracSn' , 'fracGe',
              'fracCl', 'fracI', 'fracBr'])
    
x = []
y = []
x_all = []
y_all = []
for (gr, gi) in g:
    
    labels = ['Cs', 'Rb', 'K', 'Na', 'Sn', 'Ge',
              'Cl', 'I', 'Br']
    #print(gr)
    sarr = []
    for i, n in enumerate(gr):
        
        if i < 6: 
            m = 1
        else:
            m = 3
        
        if n != 0:
            #if n == 1.0:
            sarr.append(labels[i] + '$_{' +  str(int(4*m*n)) + '}$')
            #else:
                #sarr.append(labels[i] + '$_{' +  str(4*m*n) + '}$')
        
    #print(sarr, gr)
    
    x += [''.join(sarr)]
    y.append(gi['yhat_ext'].mean())
    
    x_all += [''.join(sarr)]*len(gi)
    y_all += gi['yhat_ext'].tolist()
    
print(len(x_all), len(x))
    
fig = plt.figure(figsize=(13, 4), dpi=200)
#(Atomic 3%, Lattice 10%)
#plt.title('Stability Trends')
plt.title('Direct Bandgap Trends')
#plt.ylabel('$\Delta E_{hull}$ (meV/atom)')
plt.ylabel('$E_{g}$ (eV)')
plt.xticks(rotation=90)
plt.scatter(x, y)
#figure(num=None, figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')
plt.savefig('/Users/Jared/Documents/test.png', bbox_inches='tight')
plt.show()
'''
plt.title('Bandgap Trends (Atomic 5%, Lattice 5%)')
plt.ylabel('E$_{g}$ (eV)')
plt.xticks(rotation=90)
plt.scatter(x_all, y_all)
figure(num=None, figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')
'''