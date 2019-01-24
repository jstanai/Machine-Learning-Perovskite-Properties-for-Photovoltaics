#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jared
"""

import myConfig
import numpy as np
import pandas as pd

from collections import Counter
from ast import literal_eval

from numpy import cos, sin, sqrt
import math
import stability as stability

from ml import elements as el
from ml import integrater as g

def getCompFeature(df):

    # Compute Integrator Range
    laMax, lbMax, lcMax = printMaxMinCells(df)
    r_cutoff = myConfig.rmax
    r_cut = [1 + math.floor(r_cutoff/laMax),
             1 + math.floor(r_cutoff/lbMax), 
             1 + math.floor(r_cutoff/lcMax)]
    
    # Fundamental elemental properties to lookup and include
    #properties = ['X', 'EI1', 'rp', 'rs', 'ra', 'EI2',
    #              'l', 'h', 'fermi', 'ir', 's_occ', 'p_occ']

    properties = myConfig.properties
    properties = getPropertyMixerLabels(properties)
    
    dr = myConfig.dr
    rmax = myConfig.rmax
    r = np.arange(0.001, rmax, round(dr, 4)) 

    p_labels = get_radial_labels('p_', properties, r, dr)
    
    # EXPERIMENTAL DFT LABEL META-DATA
    experimental_params = ['e_ka', 'e_kb', 'e_kc', 
                           'e_pseudo_basis', 'e_XC', 'e_spin',
                           'e_electronTemperature', 
                           'e_densityCutoff', 
                           'e_densityMeshCutoff']
    
    # PRE-ALLOCATE FEATURE DATAFRAME      
    df_columns = ['crystal_id',
                  'counts',
                  'fracSn', 'fracGe',                       # B site
                  'fracCs', 'fracRb', 'fracNa', 'fracK',    # A site
                  'fracCl', 'fracBr', 'fracI',              # X site
                  'dH_formation',
                  'volume', 
                  'dir_gap', 
                  'ind_gap', 
                  'total_energy',
                  'em_hole', 'em_electron',
                  'la', 'lb', 'lc'
                  ] + experimental_params + p_labels 
                  
    feature = pd.DataFrame(index = range(df.shape[0] - 1), 
                           columns = df_columns)
    
    print(len(feature.columns))
    print('Computing ', len(feature.columns), 
          ' features for ', len(df), ' crystals.')
    
    # COMPUTE FEATURE ROW BY ROW (FOR EACH CRYSTAL)
    start_index = df.index[0]
    for index, row in df.iterrows():
        
        # Progress of crystals in threads
        print('Crystal ' + str(index + 1) + ' in thread ' + 
              str(start_index + 1) + ' - ' + str(len(df) + df.index[0]))
        
        # Parse experimental data into row
        experimental_params = get_experimental_params(row) 
        
        # get compositions of mixing
        counts = Counter(el.convertElementsLong2Short(
                         literal_eval(row['elements']))) 

        div = 4.0 # 4.0 for 2x1x2 supercells, 8.0 for 2x2x2 supercells
            
        fracSn = counts['Sn']/div
        fracGe = counts['Ge']/div
        fracCs = counts['Cs']/div
        fracRb = counts['Rb']/div
        fracNa = counts['Na']/div
        fracK  = counts['K']/div
        fracI  = counts['I']/(div*3)
        fracCl = counts['Cl']/(div*3)
        fracBr = counts['Br']/(div*3)
        
        # GET FORMATION ENERGY
        mu = stability.getMuCorrectedDFT2()
        dH_formation, junk = stability.calculateDeltaH_formation(
                                row['elements'], 
                                row['totalEnergy'], mu)

        # GET CONVEX HULL ENERGY
        #dH_hull = stability.getCrystalOQMDData

        # LATTICE VECTORS
        v = literal_eval(row['cellPrimVectors_end']) 

        # CONVERT FRAC COORDS TO REAL COORDS
        real_coords, la, lb, lc = convertFracCoords2(v, 
                                        row['fractional_Coordinates'])
        volume = la*lb*lc
        
        # EFFECTIVE MASS
        s = row['effectiveMass_hole']
        em_hole = literal_eval(s[0:-3])
        s = row['effectiveMass_electron']
        em_electron = literal_eval(s[0:-3])
                 
        elements = literal_eval(row['elements'])
        elements = el.convertElementsLong2Short(elements)
        
        # PDDF FAST IMPLEMENTATION
        p_array2 = []
        element_map = g.pdf_element_density(elements, real_coords,
                                            r, dr, la, lb, lc, r_cut)       
        
        for pair in properties:
            
            d = np.array([0.0])
            
            for atom in element_map:
                num = el.getElementFeature(properties = properties)[atom][pair]
                density = [num*count for count in element_map[atom]]
                d = d + density
                
            p_array2 = p_array2 + list(d)
                
        p_array2 = [np.float32(i) for i in p_array2] # SAVE DF MEMORY
        
        feature.loc[index] = [str(row['crystal_id']), 
                              counts,
                              fracSn, fracGe, 
                              fracCs, fracRb, fracNa, fracK,
                              fracCl, fracBr, fracI, 
                              dH_formation,
                              volume, 
                              row['directGap'], 
                              row['indirectGap'],
                              row['totalEnergy'],
                              em_hole, em_electron,
                              la, lb, lc
                             ] + experimental_params + p_array2   
                    
    return(feature)

def get_radial_labels(tag, properties, r, dr):

    r = r[:-1] #SO THAT ONLY UNTIL LAST NUMBER IN R IS COMPUTED
    
    p_labels = []
    for radius in r:

        rdr = '%.1f' % (dr + radius)
        radius = '%.1f' % radius
            
        a = radius.split('.')
        c = rdr.split('.')
        label = tag + a[0] + '_' + a[1] + '_' + c[0] + '_' + c[1] + '_'
        
        p_labels += [label + str(p) for p in properties]
    
    return p_labels 

def get_experimental_params(row):
    
    # Check if value exists and place NaN if not (for multiple dataframes)
    #if 'ka' not in row: row['ka'] = [np.NaN]
    #if 'kb' not in row: row['kb'] = [np.NaN]
    #if 'kc' not in row: row['kc'] = [np.NaN]
    
    if (pd.isnull(row['ka']) == False):
        ka = [literal_eval(str(row['ka']))]
    else:
        ka = [np.NaN]

    if (pd.isnull(row['kb']) == False):
        kb = [literal_eval(str(row['kb']))]
    else:
        kb = [np.NaN]

    if (pd.isnull(row['kc']) == False):
        kc = [literal_eval(str(row['kc']))]
    else:
        kc = [np.NaN] 

    # will need to update if new dataframes dont contain this info
    pseudo_basis = [row['pseudo_basis']]
    XC = [row['XC']]
    spin = [row['spin']]
    electronTemperature = [row['electronTemperature']]
    densityCutoff = [row['densityCutoff']]
    densityMeshCutoff = [row['densityMeshCutoff']]
    
    return (ka + kb + kc + pseudo_basis + XC + spin + 
            electronTemperature + densityCutoff + densityMeshCutoff)
    
def convertFracCoords1(la, lb, lc, coords):
    #assumes Orthohombric unit cell (no angles!)
    
    coords = literal_eval(coords)

    for index, i in enumerate(coords):
        coords[index] = [i[0]*la, i[1]*lb, i[2]*lc]
    
    return(coords)
    
def convertFracCoords2(v, coords):
    # does not assume Orthohombric cell (includes angles!)
    
    a = np.array(v[0:3])
    b = np.array(v[3:6])
    c = np.array(v[6:9])
    
    la = np.linalg.norm(a)
    lb = np.linalg.norm(b)
    lc = np.linalg.norm(c)
        
    alpha = math.acos(b.dot(c)/(lc*lb))
    beta  = math.acos(a.dot(c)/(la*lc))
    gamma = math.acos(b.dot(a)/(lb*la))  
     
    coords = literal_eval(coords)

    omega = la*lb*lc*sqrt(1 - cos(alpha)**2 - cos(beta)**2 - cos(gamma)**2 + \
                          2*cos(alpha)*cos(beta)*cos(gamma))
    
    for index, i in enumerate(coords):
        coords[index] = [i[0]*la + 
                         i[1]*lb*cos(gamma) + 
                         i[2]*lc*cos(beta), 
                         i[1]*lb*sin(gamma) + 
                         i[2]*lc*((cos(alpha) - 
                                  (cos(beta)*cos(gamma)))/sin(alpha)),
                         i[2]*omega/(la*lb*sin(gamma))
                        ]
                          
    return(coords, la, lb, lc)
   
def getPropertyMixerLabels(properties):
    """
    Compute the labels for the properties being used. Can be configured to 
    include other algebraic functions.
    
    Parameters
    ----------
    properties : list
        list of strings, elemental properties to consider

    Returns
    -------
    list
        The stringified algebraic combinations considered for this set of 
        properties
    """
    
    praw = properties #fundamental elemental properties
    
    '''
    # get EXP(X), X^(-1), LN(1 + X), X^2. Do not compute inverse for s_occ and
    # p_occ as they contain 1/0 division errors!
    #exp = ['(exp(' + p + '))' for p in properties]
    inverse = ['(1/' + p + ')' for p in properties]
    #ln = ['(log(1 + ' + p + '))' for p in properties]
    p = ['(' + p + ')' for p in properties]
    
    #properties = exp + inverse + ln + p
    properties = inverse + p
 
    # get X*Y for all X, Y in properties
    #pair_properties_1 = list(permutations(properties, 2))
    pair_properties_2 = list(combinations_with_replacement(properties, 2))

    m = [pair[0] + '*' + pair[1] 
         for pair in pair_properties_2] + praw
    '''
    m = praw
    
    return(m)

def printMaxMinCells(df):
    """
    Computes the maximum and minimum lattice vectors in the crystal dataset,
    which are used to compute the bounds of the integrator.py function for
    a given rmax. For 2x1x2 supercells of dimension 10x5x10 for example, 
    a rmax of 15 would require a grid of supercells 5x7x5 in dimension to 
    integrate over.
    
    Parameters
    ----------
    df : DataFrame
        Crystals database (output from ncParser4.py)

    Returns
    -------
    float
        Maximum lattice vector values for la, lb, and lc
    """
    laMax = 0
    lbMax = 0
    lcMax = 0
    laMin = 100
    lbMin = 100
    lcMin = 100
    
    for index, row in df.iterrows():
        df_l_check = literal_eval(row['cellPrimVectors_end'])
        if df_l_check[0] >= laMax: laMax = df_l_check[0]
        if df_l_check[4] >= lbMax: lbMax = df_l_check[4]
        if df_l_check[8] >= lcMax: lcMax = df_l_check[8]
        if df_l_check[0] <= laMin: laMin = df_l_check[0]
        if df_l_check[4] <= lbMin: lbMin = df_l_check[4]
        if df_l_check[8] <= lcMin: lcMin = df_l_check[8]

    #print('laMax ', laMax, 'laMin', laMin)
    #print('lbMax ', lbMax, 'lbMin', lbMin)
    #print('lcMax ', lcMax, 'lcMin', lcMin)
    
    return laMax, lbMax, lcMax 