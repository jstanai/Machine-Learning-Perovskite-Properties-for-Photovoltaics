#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 17:58:16 2018

@author: Jared
"""

import pandas as pd
from ast import literal_eval
import random
from builder import dummyCrystalConfig as dc
from builder import perovskiteBuilder as ab

def mixFracCoords(fracCoords, cf):
    
    for find, fx in enumerate(fracCoords):
        
        fracCoords[find] = [fx[0]+float(random.uniform(-cf, cf)),
                            fx[1]+float(random.uniform(-cf*2, cf*2)),
                            fx[2]+float(random.uniform(-cf, cf))]
    
    return(fracCoords) 

def processDummyCrystals(df):
    
    df_row = df.head(1).copy() #enforce df is single row
    
    #
    #
    # lattice multiplication:    
    mnum = dc.lattice_mul
    m0 = [[1., 1., 1.]]
    m = m0*mnum

    c = dc.lattice_variation_percent/100 # percent lattice vector deviation
    for mind, mx in enumerate(m):
        
        m[mind] = [mx[0]+float(random.uniform(-c, c)),
                   mx[1]+float(random.uniform(-c, c)),
                   mx[2]+float(random.uniform(-c, c))]
      
    n = dc.lattice_comp_num # number of crystals at same composition to generate
    fnum = dc.fnum # number of randomizations for each crystal
    cf = dc.atomic_position_variation_percent/100 
    
    # set to default fracCoords
    fracCoords = str(ab.getFracCoords())
    df_row['fractional_Coordinates'] = fracCoords
    
    
    
    Asite = dc.Asite
    Bsite = dc.Bsite
    Xsite = dc.Xsite

    resolution = dc.resolution
    df1 = ab.generateCrystals(n, Asite, Bsite, Xsite, resolution,
                              fracCoords, getAll = True, 
                              halfCell = True, symmetry = False, start = 0)
    
    df_dummy = pd.concat([df_row]*len(df1), ignore_index=True)
    df_dummy.reset_index()
    df_dummy = df_dummy.copy()
    
    print('now here')
    print(df1['e_list'])
    # fill with mixed elements
    elementsdf = df1['e_list'].copy()
    for ind, i in enumerate(elementsdf):
        df_dummy.loc[ind, 'elements'] = str(ab.convertElementsShort2Long(
                                            literal_eval(i)))
    
    
    # repeat for each volume (len(m) times)
    df_dummy = pd.concat([df_dummy]*mnum, ignore_index=True)
    
    li = dc.lattice_init
    
    for mi, mv in enumerate(m):
        lattice = str([mv[0]*li[0], 0.0,         0.0, 
                       0.0,         mv[1]*li[1], 0.0, 
                       0.0,         0.0,         mv[2]*li[2]]) 
     
        df_dummy.loc[n*mi:n*(mi + 1) - 1, 'cellPrimVectors_end'] = lattice
    
    # repeat for random fracCoord fnum times
    df_dummy = pd.concat([df_dummy]*fnum, ignore_index=True)
    
    for i in range(fnum):
        fracCoords = mixFracCoords(ab.getFracCoords(), cf)
        fracCoords = str(fracCoords)
    
        df_dummy.loc[n*mnum*i:n*mnum*(i + 1) - 1, 'fractional_Coordinates'] = fracCoords
    
    #print(df_dummy[['elements', 
    #                'cellPrimVectors_end', 
    #                'fractional_Coordinates']].head())

    print('Number of Dummy Crystals Generated: {}'.format(len(df_dummy)))
    print(df_dummy['elements'], df_dummy.shape)
    return(df_dummy)