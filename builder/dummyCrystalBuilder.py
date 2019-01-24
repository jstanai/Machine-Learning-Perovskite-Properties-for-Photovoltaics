#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 17:58:16 2018

@author: Jared
"""

import pandas as pd
import perovskiteBuilder as ab
from ast import literal_eval

import random

def mixFracCoords(fracCoords, cf):
    
    for find, fx in enumerate(fracCoords):
        r1 = float(random.uniform(-50, 50))/cf
        r2 = float(random.uniform(-100, 100))/cf
        r3 = float(random.uniform(-50, 50))/cf
        
        #r1 = 0
        #r2 = 0
        #r3 = 0
        
        fracCoords[find] = [fx[0]+r1,fx[1]+r2,fx[2]+r3]
    
    return(fracCoords) 

def processDummyCrystals(df):

    df_row = df.head(1).copy() #enforce df is single row
    
    #
    #
    # lattice multiplication:    
    mnum = 5000
    
    #m = [[1.065, 1.065, 1.065]]*mnum
    m = [[1., 1., 1.]]*mnum
    
    c = 5*400 #400 = 25% deviation
    for mind, mx in enumerate(m):
        #r1 = float(random.randrange(-100, 100))/c
        #r2 = float(random.randrange(-100, 100))/c
        #r3 = float(random.randrange(-100, 100))/c
        
        r1 = float(random.uniform(-100, 100))/c
        r2 = float(random.uniform(-100, 100))/c
        r3 = float(random.uniform(-100, 100))/c
        
        print(r1)
        m[mind] = [mx[0]+r1,mx[1]+r2,mx[2]+r3]
     

    
    n = 1 # number of crystals at same composition to generate
    fnum = 1 # number of randomizations for each crystal
    cf = 25*400 # randomization of lattice position
    
    # set to default fracCoords
    fracCoords = str(ab.getFracCoords())
    df_row['fractional_Coordinates'] = fracCoords
    
    df_dummy = pd.concat([df_row]*n, ignore_index=True)
    df_dummy.reset_index()
    df_dummy = df_dummy.copy()
    
    # possible elements
    Asite = ['Rb', 'Cs', 'K', 'Na']
    Bsite = ['Sn', 'Ge']
    Xsite = ['Cl', 'Br', 'I'] 
    #test compound trend
    Asite = ['Cs']
    Bsite = ['Sn']
    Xsite = ['I']

    start = 0
    resolution = [4, 4, 6]
    df1 = ab.generateCrystals(n, Asite, Bsite, Xsite, resolution,
                           fracCoords, getAll = True, 
                           halfCell = True, symmetry = False, start = start)
    
    print('size df1', len(df1))
    
    
    # fill with mixed elements
    elementsdf = df1['e_list'].copy()
    for ind, i in enumerate(elementsdf):
        df_dummy.loc[ind, 'elements'] = str(ab.convertElementsShort2Long(
                                            literal_eval(i)))
    
    
    # repeat for each volume (len(m) times)
    df_dummy = pd.concat([df_dummy]*mnum, ignore_index=True)
    
    for mi, mv in enumerate(m):
        lattice = str([mv[0]*11.7, 0.0, 0.0, 0.0, mv[1]*5.4, 0.0, 0.0, 0.0, mv[2]*11.7]) 
        lattice = str([mv[0]*2*6.219, 0.0, 0.0, 0.0, mv[1]*6.219, 0.0, 0.0, 0.0, mv[2]*2*6.219]) 
        df_dummy.loc[n*mi:n*(mi + 1) - 1, 'cellPrimVectors_end'] = lattice
    
    # repeat for random fracCoord fnum times
    df_dummy = pd.concat([df_dummy]*fnum, ignore_index=True)
    
    for i in range(fnum):
        fracCoords = mixFracCoords(ab.getFracCoords(), cf)
        fracCoords = str(fracCoords)
    
        df_dummy.loc[n*mnum*i:n*mnum*(i + 1) - 1, 'fractional_Coordinates'] = fracCoords
    
    
    
    
    
    print(df_dummy[['elements', 'cellPrimVectors_end', 'fractional_Coordinates']].head())

    return(df_dummy)