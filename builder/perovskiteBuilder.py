#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Jared
"""

import random
from ase.visualize import view
from ase import Atoms
from collections import Counter
import pandas as pd
import csv
import math
import matplotlib.pyplot as plt
import itertools
from builder import perovskiteBuilderConfig as c

def main():

    fractional_coordinates = getFracCoords() 

    df1 = generateCrystals(c.num, c.Asite, c.Bsite, c.Xsite, c.resolution,
                           fractional_coordinates, getAll = c.getAll, 
                           halfCell = False, symmetry = False, 
                           start = c.start)
    
    print('Number of Crystals Requested: ', c.num)
    print('Number of Crystals Generated (Filtering duplicates): ', len(df1))
     
    data_count_i = 0 + c.start
    
    for chunk in getdf_chunks(df1, 500) :
        
        df1 = chunk
        df1 = df1.sample(frac=1)
        
        data_count_f = data_count_i + len(chunk)
        
        fname2 = c.fname.split('.')[0] + '_' + \
                 str(data_count_i) + '-' + str(data_count_f - 1) + '.csv'
               
        data_count_i = data_count_f
        
        header = {'lattice' : c.latticeType, 
                  'a' : c.pVector[0],
                  'b' : c.pVector[1],
                  'c' : c.pVector[2],
                  'fractional_coordinates' : fractional_coordinates}
      
        with open(c.path + fname2, 'w') as f: 
            writer = csv.writer(f)
            for key, value in header.items():
                writer.writerow([key, value])
                
            print('Writing file of length', len(df1), 'to CSV')
            df1.to_csv(f, index = None)
    
def getdf_chunks(df, num):
    """
    Breaks DataFrame into chunks.
    
    Returns
    -------
    array
        Array of DataFrames
    
    """
    c = round(len(df)/num)
    df_chunks = [df.loc[i*num : (i+1)*num - 1, : ] for i in range(c + 1) if
                 len(df.loc[i*num : (i+1)*num - 1, : ]) != 0]
    
    return(df_chunks)

def generateElements(Asite, Bsite, Xsite, res, **kwargs):
    """
    Generate random element vector for crystal.
    
    Parameters
    ----------
    Asite : array
        A-site elements.
    Bsite : array
        B-site elements.
    Xsite : array
        X-site elements.
    res : array
        resolution (e.g. [4, 4, 4])
    
    Returns
    -------
    array
        VNL-ATK compatiable array of perovskite elements organized as
        [A-site atoms, B-site atoms, X-site atoms]
    
    """

    n_a = res[0] 
    n_b = res[1]
    n_x = res[2] 
    
    if('start' in kwargs):
        start = kwargs['start']
    else:
        start = 0
    
    aNum = 8
    bNum = 8
    xNum = 24
    if kwargs['halfCell'] == True:
        aNum = 4
        bNum = 4
        xNum = 12
    
    # GETS ALL POSSIBLE CONFIGURATION FOR ALL 
    # MACROSTATES (Realizations of mixing fraction)
    # -> not the full configuration space as it is subject to mixing fraction
    #    contraints!
    if kwargs['getAll'] == True:
        print('Getting all configs...')
        Alist = list(itertools.combinations_with_replacement(
                    list(range(len(Asite))), n_a))    
    
        Alist2 = []
        for e in Alist:
            Alist2 += [i for i in itertools.permutations(e)]
        Alist = list(set(Alist2))
        
        Asites = []
        for pair in Alist:
            Asite_p = [Asite[e] for e in pair]
            
            if kwargs['symmetry'] == True:
                Asites = Asites + [Asite_p*int(aNum/n_a)]
            else:
                Asites = Asites + [random.sample(Asite_p*(int(aNum/n_a)), aNum)]
                
        Blist = list(itertools.combinations_with_replacement(
                    list(range(len(Bsite))), n_b))

        Blist2 = []
        for e in Blist:
            Blist2 += [i for i in itertools.permutations(e)]
            
        Blist = list(set(Blist2))

        Bsites = []
        for pair in Blist:
            Bsite_p = [Bsite[e] for e in pair]  
            
            if kwargs['symmetry'] == True:
                Bsites = Bsites + [Bsite_p*int(bNum/n_b)]
            else:
                Bsites = Bsites + [random.sample(Bsite_p*(int(bNum/n_b)), bNum)]
        
        Xlist = list(itertools.combinations_with_replacement(
                    list(range(len(Xsite))), n_x))
        
        Xlist2 = []
        for e in Xlist:
            Xlist2 += itertools.permutations(e)
           
        Xlist = list(set(Xlist2))
        
        Xsites = []
        
        for pair in Xlist:
            Xsite_p = [Xsite[e] for e in pair]  
            
            # USE THIS OPTION TO SPECIFY SIMPLE ARRAY [A A B, A A B...] 
            # patterns, rather than samples
            if kwargs['symmetry'] == True:
                Xsites = Xsites + [Xsite_p*int(xNum/n_x)]
            else:
                Xsites = Xsites + [random.sample(Xsite_p*(int(xNum/n_x)), xNum)]
        
        Elist = list(itertools.product(Asites, Bsites, Xsites))
        Elist_full = [pair[0] + pair[1] + pair[2] for pair in Elist]
        print('Number of Possibilities: ', len(Elist_full))     
        
        return pd.DataFrame(Elist_full[start:])
    
    Asite_p = [random.randrange(0,len(Asite)) for x in range(n_a)]
    Asite_p = [Asite[Asite_p[x]] for x in range(n_a)]
    Asite_p = random.sample(Asite_p*(int(8/n_a)), 8)
    #Asite_p = random.sample(Asite_p*(int(8/n_a)), 4) #half cell

    Bsite_p = [random.randrange(0,len(Bsite)) for x in range(n_b)]   
    Bsite_p = [Bsite[Bsite_p[x]] for x in range(n_b)]
    Bsite_p = random.sample(Bsite_p*(int(8/n_b)), 8)
    #Bsite_p = random.sample(Bsite_p*(int(8/n_b)), 4) #half cell

    Xsite_p = [random.randrange(0,len(Xsite)) for x in range(n_x)]
    Xsite_p = [Xsite[Xsite_p[x]] for x in range(n_x)]
    Xsite_p = random.sample(Xsite_p*(int(24/n_x)), 24) #MAP 1/8 RESOLUTION TO 24 POSITIONS
    #Xsite_p = random.sample(Xsite_p*(int(24/n_x)), 12) # half cell
    
    return(Asite_p + Bsite_p + Xsite_p)

def viewCrystal(elements, pos, **kwargs):
    """
    View Crystal using ASE.
    
    Parameters
    ----------
    elements : array
        list of elements
    pos :
        the 2nd param
    """
    pos = [[ 10 * i for i in inner ] for inner in pos]
    atoms = Atoms(elements, positions=pos, cell=[10, 10, 10], pbc=[1, 1, 1])
    
    view(atoms)
    

def generateCrystals(numCrystalTries, Asite, Bsite, Xsite, 
                     res, fractional_coordinates, **kwargs):
    """
    Create Crystals for ingestion into VNL-ATK DFT scripts.
    
    Parameters
    ----------
    numCrystalTries : int
        Number of crystals to generate.
    Asite : array
        A-site elements.
    Bsite : array
        B-site elements.
    Xsite : array
        X-site elements.
    res : array
        resolution (e.g. [4, 4, 4])
    fractional_coordinates : array
        Fractional Coordinate list.
    
    Returns
    -------
    DataFrame
        DataFrame of perovskite crystals.
    
    """
    
    d = []
    
    if(kwargs['getAll'] == True):
        elementsDF = generateElements(Asite, Bsite, Xsite, 
                                      res, 
                                      getAll = kwargs['getAll'], 
                                      halfCell = kwargs['halfCell'],
                                      symmetry = kwargs['symmetry'],
                                      start = kwargs['start'])
        
        elementsDF = elementsDF.sample(frac=1) #randomize
        elementsDF = elementsDF.reset_index(drop = True)
        
        for i in range(min(numCrystalTries, len(elementsDF))):
            elements = list(elementsDF.loc[i,:])
            counts = Counter(elements) 
            counts['e_list'] = str(elements)
            d = d + [counts]            
        
    else:
        for i in range(numCrystalTries):
            elements = generateElements(Asite, Bsite, Xsite, 
                                        res, 
                                        getAll = kwargs['getAll'],
                                        halfCell = kwargs['halfCell'])
            
            counts = Counter(elements)     
            counts['e_list'] = str(elements)
            d = d + [counts]
     
    # Process duplicates, NaNs and Data
    
    
    df1 = pd.DataFrame(data = d) 
    
    
    
    df1 = df1.fillna(0) 
    dup_check = ['e_list']
    df1 = df1.drop_duplicates(dup_check) 
    
    return(df1)

def getFracCoords():
    """
    Returns fractional coordinates for cubic supercells.
    
    Returns
    -------
    array
        Array of fractional coordinates.

    """
    # 2x2x2 Supercell
    fractional_coords_full = [[ 0.  ,  0.  ,  0.  ],
                              [ 0.  ,  0.  ,  0.5 ],
                              [ 0.  ,  0.5 ,  0.  ],
                              [ 0.  ,  0.5 ,  0.5 ],
                              [ 0.5 ,  0.  ,  0.  ],
                              [ 0.5 ,  0.  ,  0.5 ],
                              [ 0.5 ,  0.5 ,  0.  ],
                              [ 0.5 ,  0.5 ,  0.5 ],
                              [ 0.25,  0.25,  0.25],
                              [ 0.25,  0.25,  0.75],
                              [ 0.25,  0.75,  0.25],
                              [ 0.25,  0.75,  0.75],
                              [ 0.75,  0.25,  0.25],
                              [ 0.75,  0.25,  0.75],
                              [ 0.75,  0.75,  0.25],
                              [ 0.75,  0.75,  0.75],
                              [ 0.25,  0.25,  0.  ],
                              [ 0.25,  0.25,  0.5 ],
                              [ 0.25,  0.75,  0.  ],
                              [ 0.25,  0.75,  0.5 ],
                              [ 0.75,  0.25,  0.  ],
                              [ 0.75,  0.25,  0.5 ],
                              [ 0.75,  0.75,  0.  ],
                              [ 0.75,  0.75,  0.5 ],
                              [ 0.  ,  0.25,  0.25],
                              [ 0.  ,  0.25,  0.75],
                              [ 0.  ,  0.75,  0.25],
                              [ 0.  ,  0.75,  0.75],
                              [ 0.5 ,  0.25,  0.25],
                              [ 0.5 ,  0.25,  0.75],
                              [ 0.5 ,  0.75,  0.25],
                              [ 0.5 ,  0.75,  0.75],
                              [ 0.25,  0.  ,  0.25],
                              [ 0.25,  0.  ,  0.75],
                              [ 0.25,  0.5 ,  0.25],
                              [ 0.25,  0.5 ,  0.75],
                              [ 0.75,  0.  ,  0.25],
                              [ 0.75,  0.  ,  0.75],
                              [ 0.75,  0.5 ,  0.25],
                              [ 0.75,  0.5 ,  0.75]]


    # 2x1x2 Supercell (a = 2b = c)
    fractional_coords_half = [[ 0.  ,  0.  ,  0.  ],
                              [ 0.  ,  0.  ,  0.5 ],
                              [ 0.5 ,  0.  ,  0.  ],
                              [ 0.5 ,  0.  ,  0.5 ],
                              [ 0.25,  0.5 ,  0.25],
                              [ 0.25,  0.5 ,  0.75],
                              [ 0.75,  0.5 ,  0.25],
                              [ 0.75,  0.5 ,  0.75],
                              [ 0.25,  0.5 ,  0.  ],
                              [ 0.25,  0.5 ,  0.5 ],
                              [ 0.75,  0.5 ,  0.  ],
                              [ 0.75,  0.5 ,  0.5 ],
                              [ 0.  ,  0.5 ,  0.25],
                              [ 0.  ,  0.5 ,  0.75],
                              [ 0.5 ,  0.5 ,  0.25],
                              [ 0.5 ,  0.5 ,  0.75],
                              [ 0.25,  0.  ,  0.25],
                              [ 0.25,  0.  ,  0.75],
                              [ 0.75,  0.  ,  0.25],
                              [ 0.75,  0.  ,  0.75]]
    
    return fractional_coords_half

def generateDegreeOfHope():
    
    """
    Plots number of available macrostates for given composition resolutions.
    """
    
    # Crystal Resolutions
    n_a = [1, 2, 4, 4, 8] # divisor of 8
    n_b = [1, 2, 4, 4, 8] # divisor of 8
    n_x = [1, 2, 3, 4, 8] # divisor of 24
    
    n_label = [1, 0.5, 0.33, 0.25, 0.125]
    
    #num balls
    cval = []
    for index, i in enumerate(n_a):
        n_balls_a = n_a[index]
        n_balls_b = n_b[index]
        n_balls_x = n_x[index]
    
        nb = [n_balls_a, n_balls_b, n_balls_x]
        #buckets
        m_a = 4
        m_b = 2
        m_x = 3
    
        mb = [m_a, m_b, m_x]
    
        c = 1
        for index, b in enumerate(nb):
            ct = math.factorial(b + mb[index] - 1)
            cb = math.factorial(b)*math.factorial(mb[index] - 1)
            c = c*(ct/cb)
        cval = cval + [c]
    
    plt.plot(n_label, cval)
    plt.show
    print(' n_a: ', n_a, '\n', 'n_b: ', n_b, '\n', 'n_x: ', n_x, '\n', 
          'possible values: ', cval)

def getElementDict():
    """
    Element name dictionary.
    
    Returns
    -------
    dict
        Long name keys, short name values
    """    

    
    return({'Rubidium' : 'Rb', 
             'Caesium':  'Cs', 
             'Germanium' : 'Ge', 
             'Tin' : 'Sn', 
             'Iodine' : 'I', 
             'Bromine' : 'Br', 
             'Chlorine' : 'Cl',
             'Sodium': 'Na',
             'Potassium': 'K'})

def convertElementsLong2Short(elements):
    """
    Convert long element names to abbreviations.
    
    Parameters
    ----------
    first : array of strings
        Long element names
    
    Returns
    -------
    array
        Short element names
    """  
    
    if len(elements[0]) == 2: return(elements)
    
    eDict = getElementDict()

    for n, i in enumerate(elements):
        elements[n] = eDict[i]

    return(elements)

def convertElementsShort2Long(elements):
    """
    Convert element abbreviations to long names.
    
    Parameters
    ----------
    first : array of strings
        Short element names
    
    Returns
    -------
    array
        Long element names
    """   
    eDict = getElementDict()
    eDict = dict((v, k) for k, v in eDict.items())
    
    for n, i in enumerate(elements):
        elements[n] = eDict[i]

    return(elements)
    
def getSpaceGroup(elements, coords, la, lb, lc):
    
    print(elements)
    
    #coords = getFracCoords()
    
    crystal = Atoms([elements[0]]*20, coords, 
                    pbc = [1, 1, 1])
    
    lattice = [[la,  0, 0 ],
               [ 0, lb, 0 ],
               [ 0,  0, lc]]
    
    cell = (lattice,
            coords,
            crystal.get_atomic_numbers())
    
    spacegroup = spglib.get_spacegroup(cell, symprec=1e-3)
    print(spacegroup, crystal.get_atomic_numbers())
    
    return(spacegroup)
   
 
def nballs_mbuckets(n, m):
    return math.factorial(n + m - 1)/(math.factorial(n)*math.factorial(m-1))
 
    

def getTotalConfigurations(r):
    
    
    totalA_positions = 4
    totalB_positions = 4
    totalX_positions = 12
    
    #bucket arrangement:
    # m: number of element possibilities
    # n: number of lattice positions needing assignment
   
    pA = nballs_mbuckets(r[0], 4)
    pB = nballs_mbuckets(r[1], 2)
    pX = nballs_mbuckets(r[2], 3)
    print(pA)
    # group sizes
    pAg = int(totalA_positions/r[0])
    pBg = int(totalB_positions/r[1])
    pXg = int(totalX_positions/r[2])
    
    for i in range(int(pA)):  
        n = i
        g = pAg
        bigN = totalA_positions
        
        numArrangements = int(math.factorial(bigN)/
                          (math.factorial(n*g)*math.factorial(bigN - n*g)))
        
        print('n', n, 
              'groupSize', pAg,  
              'n*g', n*g, 
              'totalPositions', bigN, 
              'numA', numArrangements)

if __name__ == '__main__':
    main()