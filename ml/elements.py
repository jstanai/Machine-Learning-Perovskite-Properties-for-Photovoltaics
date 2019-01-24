#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Jared
"""
import pandas as pd
import math
import numpy as np
from math import exp
from math import log
import time
from itertools import combinations_with_replacement
import myConfig

def getElementDict():
    
    eDict = {'Rubidium' : 'Rb', 
             'Caesium':  'Cs', 
             'Germanium' : 'Ge', 
             'Tin' : 'Sn', 
             'Iodine' : 'I', 
             'Bromine' : 'Br', 
             'Chlorine' : 'Cl',
             'Sodium': 'Na',
             'Potassium': 'K',
             'Hydrogen': 'H',
             'Lead': 'Pb',
             'Nitrogen': 'N',
             'Carbon' : 'C'}
    
    return(eDict)
    
def convertElementsLong2Short(elements):
    
    if len(elements[0]) == 2: return(elements)
    
    eDict = getElementDict()

    for n, i in enumerate(elements):
        elements[n] = eDict[i]

    return(elements)

def convertElementsShort2Long(elements):
    
    eDict = getElementDict()
    eDict = dict((v, k) for k, v in eDict.items())
    
    for n, i in enumerate(elements):
        elements[n] = eDict[i]

    return(elements)
    
def getElementFeature(**kwargs):
    
    a = 0.529177249 # au to angstrom
    
    # https://journals.aps.org/prb/pdf/10.1103/PhysRevB.85.104104
    # https://en.wikipedia.org/wiki/Atomic_radii_of_the_elements_(data_page)
    # https://www.ptable.com/
    
    # atomic radius: empirical : https://aip.scitation.org/doi/10.1063/1.1725697
    
    # ir values come from https://pubs.acs.org/doi/suppl/10.1021/jacs.7b09379/suppl_file/ja7b09379_si_001.pdf
    # (table 2 supplementary information)
    # ir is crystal radius from abulafia.mt.ic.ac.uk/shannon/ptable.php
    #   Cs, Rb, Na, K are 12 fold coordinated
    #   Sn, Ge are 6 fold coordinated
    eDict = {'Cs': {'rs'    : 1.66*a, 
                    'rp'    : 3.08*a,
                    'ra'    : 2.60, # atomic radius wikipedia atomic radius table
                    'X'     : 0.79, # electronegativity (Pauling Scale)
                    'EI1'   : 3.89386, #eV/atom
                    'EI2'   : 23.15689, #eV/atom
                    'h'     : 7.353e-6, #HOMO Level eV relative to fermi (multiply by -1!)
                    'l'     : 3.371, #LUMO Level eV relative to fermi!
                    'fermi' : 1.149, #multiply by -1!
                    'ir'    : 2.02, #ang (crystal ionic raddi shannon)
                    'socc' : 1, 
                    'pocc' : 0,
                    },
             'Rb': {'rs'    : 1.44*a,
                    'rp'    : 2.86*a,
                    'ra'    : 2.35,
                    'X'     : 0.89,              
                    'EI1'   : 4.177, #eV/atom
                    'EI2'   : 27.2798, #eV/atom
                    'h'     : 4.145e-5,
                    'l'     : 2.972,   
                    'fermi' : 1.519,
                    'ir'    : 1.86, # ang
                    'socc' : 1, 
                    'pocc' : 0,
                    },
             'Cl': {'rs'    : 0.49*a, 
                    'rp'    : 0.59*a,
                    'ra'    : 1.00,
                    'X'     : 3.16,              
                    'EI1'   : 12.96778, #eV/atom
                    'EI2'   : 23.817, #eV/atom
                    'h'     : 4.162e-2,
                    'l'     : 1.351e1,  
                    'fermi' : 8.604,
                    'ir'    : 1.67, #ang
                    'socc' : 2, 
                    'pocc' : 5,
                    },
             'I':  {'rs'    : 0.71*a, 
                    'rp'    : 0.87*a,
                    'ra'    : 1.40,
                    'X'     : 2.66,              
                    'EI1'   : 10.45133, #eV/atom
                    'EI2'   : 19.13141, #eV/atom
                    'h'     : 4.161e-2,
                    'l'     : 1.054e1, 
                    'fermi' : 6.574,
                    'ir'    : 2.06, #ang
                    'socc' : 2, 
                    'pocc' : 5,
                    },
             'Br': {'rs'    : 0.57*a, 
                    'rp'    : 0.70*a,
                    'ra'    : 1.15,
                    'X'     : 2.96,              
                    'EI1'   : 11.81423, #eV/atom
                    'EI2'   : 21.796, #eV/atom
                    'h'     : 4.130e-2,
                    'l'     : 1.415e1,         
                    'fermi' : 7.796,
                    'ir'    : 1.82, # ang 
                    'socc' : 2, 
                    'pocc' : 5,
                    },
             'Sn': {'rs'    : 0.78*a, 
                    'rp'    : 1.06*a,
                    'ra'    : 1.45,
                    'X'     : 1.96,              
                    'EI1'   : 7.34412, #eV/atom
                    'EI2'   : 14.63228, #eV/atom
                    'h'     : 6.848,
                    'l'     : 1.792e-2,
                    'fermi' : 3.516,        
                    'ir'    : 1.20, # ang https://pubs.acs.org/doi/suppl/10.1021/jacs.7b09379/suppl_file/ja7b09379_si_001.pdf
                    'socc' : 2, 
                    'pocc' : 2,
                    },
             'Ge': {'rs'    : 0.64*a, 
                    'rp'    : 0.90*a,
                    'ra'    : 1.25,
                    'X'     : 2.01,              
                    'EI1'   : 7.898, #eV/atom
                    'EI2'   : 15.93507, #eV/atom
                    'h'     : 7.847,
                    'l'     : 1.793e-2,  
                    'fermi' : 3.552,
                    'ir'    : 0.87, # ang
                    'socc' : 2, 
                    'pocc' : 2,
                    },
             'K':  {'rs'    : 1.34*a, 
                    'rp'    : 2.68*a,
                    'ra'    : 2.20,
                    'X'     : 0.82,              
                    'EI1'   : 4.340557, #eV/atom
                    'EI2'   : 31.632, #eV/atom
                    'h'     : 1.658e1,
                    'l'     : 3.166e-6, 
                    'fermi' : 1.673,
                    'ir'    : 1.78, # ang 
                    'socc' : 1, 
                    'pocc' : 0,
                    },
             'Na': {'rs'    : 1.01*a, 
                    'rp'    : 2.35*a,
                    'ra'    : 1.80,
                    'X'     : 0.93,              
                    'EI1'   : 5.13861, #eV/atom
                    'EI2'   : 47.282, #eV/atom
                    'h'     : 2.585e1,
                    'l'     : 2.882e-5, 
                    'fermi' : 2.436,
                    'ir'    : 1.53, 
                    'socc' : 1, 
                    'pocc' : 0,
                    },
            }

    eDict2 = {'H': {'rs'    : 0, # does not exist
                    'rp'    : 0, # does not exist
                    'ra'    : 0.25,
                    'X'     : 2.2,   
                    'EI1'   : 13.597925, #eV/atom 
                    'EI2'   : 0, # does not exist
                    'h'     : 0, # does not exist
                    'l'     : 4.1324e-4, 
                    'fermi' : 6.4947,
                    'ir'    : 0, # does not exist (no crystal)
                    'socc' : 1, 
                    'pocc' : 0,
                    },
             'N':  {'rs'    : 0.33*a, 
                    'rp'    : 0.21*a,
                    'ra'    : 0.65,
                    'X'     : 3.04,              
                    'EI1'   : 14.533819, #eV/atom
                    'EI2'   : 29.6, #eV/atom
                    'h'     : 4.079525e-4,
                    'l'     : 1.56486e1, 
                    'fermi' : 7.1394, 
                    'ir'    : 0, # does not exist (no crystal)
                    'socc' : 2, 
                    'pocc' : 3,
                    }, 
             'C':  {'rs'    : 0.39*a, 
                    'rp'    : 0.25*a,            
                    'ra'    : 0.70,
                    'X'     : 2.55,              
                    'EI1'   : 11.260782, #eV/atom
                    'EI2'   : 24.382987, #eV/atom
                    'h'     : 8.461953,
                    'l'     : 1.791621e-2, 
                    'fermi' : 5.341208,
                    'ir'    : 0, # does not exist (no crystal)
                    'socc' : 2, 
                    'pocc' : 2,
                    },
             'Pb': {'rs'    : 0.96*a, 
                    'rp'    : 1.13*a,
                    'ra'    : 1.80,
                    'X'     : 1.87,              
                    'EI1'   : 7.416673, #eV/atom
                    'EI2'   : 15.033377, #eV/atom           
                    'h'     : 8.435864,
                    'l'     : 1.792086e-2, 
                    'fermi' : 3.209548,
                    'ir'    : 1.33, # ang
                    'socc' : 2, 
                    'pocc' : 2,
                    }, 
                }
    
    eDict.update(eDict2)

        
    if ('properties' in kwargs):
        eDict = propertyAddMixer(eDict, kwargs['properties'])
    
    return(pd.DataFrame(eDict))
   
def propertyAddMixer(eDict, properties):
     
    properties_all = properties
        
    element_list = [e for e in eDict]
    
    for element in element_list:
        
        # Find a smarter way to do this
        rs = eDict[element]['rs']
        rp = eDict[element]['rp']
        ra = eDict[element]['ra']
        X = eDict[element]['X']
        EI1 = eDict[element]['EI1']
        EI2 = eDict[element]['EI2']
        h = eDict[element]['h']
        l = eDict[element]['l']
        fermi = eDict[element]['fermi']
        ir = eDict[element]['ir']   
        socc = eDict[element]['socc'] 
        pocc = eDict[element]['pocc']  
        
        #MUST BE IN SAME ORDER AS THE LIST USED IN myConfig FILE!!!!
        #properties = myConfig.propertyAddMixer
        #properties = [X, EI1, rp, fermi]
        
        properties = [X, EI1, rp, rs, ra, EI2, l, 
                      h, fermi, ir, socc, pocc]
         
        #properties = [X, EI1, ir, s_occ, p_occ]
        
        praw = properties
        
        '''
        #exps = [math.exp(p) for p in properties]
        #for e, p in enumerate(properties):
        #    if p == 0:
        #        print(element, p)
        #        properties[e] = np.nan
              
        #inverse = [1.0/p for p in properties]
        
        inverse = [0.0]*len(properties)
        for e, p in enumerate(properties):
            try:
                inverse[e] = 1.0/p
            except:
                inverse[e] = 0.0
        
        
        #for e, i in enumerate(inverse):
        #    if np.isnan(i):
        #        inverse[e] = 0.0
             
        #ln = [math.log(1 + p) for p in properties]
        p = properties
        
        #properties = exps + inverse + ln + p
        properties =  inverse + p
        
        pair_properties = list(combinations_with_replacement(properties, 2))
        
        m = [pair[0]*pair[1] for pair in pair_properties] + praw
        '''
        
        m = praw
        
        for key, val in zip(properties_all, m):
                eDict[element][key] = val
                
                #if val > 100000:
                #    print(element, key, val)
                
         
       
    return(eDict)
        

def getPropertyMixerLabels(properties):
    praw = properties
    # GET EXP(X), X^(-1), LN(1 + X), X^2

    #exp = ['(exp(' + p + '))' for p in properties]
    inverse = ['(1/' + p + ')' for p in properties]
    #inverse = ['(1/' + p + ')' for p in properties if (p.find('_occ') == -1)]
    #ln = ['(log(1 + ' + p + '))' for p in properties]
    p = ['(' + p + ')' for p in properties]
    
    #properties = exp + inverse + ln + p
    properties = inverse + p
    # GET X*Y FOR ALL X, Y IN PROPERTIES
    #pair_properties_1 = list(permutations(properties, 2))
    pair_properties_2 = list(combinations_with_replacement(properties, 2))

    m = [pair[0] + '*' + pair[1] 
         for pair in pair_properties_2] + praw
    #d = [pair[0] + '/' + pair[1] for pair in pair_properties_1 
    #     if (pair[0] != pair[1] and
    #         pair[1].find('_occ') == -1)]
    
    return(m)

    
  
#def getPropertyMixer(properties):
    
    # Get multiplicative pairs
    #pair_properties = list(permutations(properties, 2))
    
    #for pair in pair_properties    
    
def getMu():
    # chemical potential mu (ev/atom) from 
    # https://www.nature.com/articles/npjcompumats201510/tables/3  
    mu = {'Na' : -1.212,
          'K'  : -1.097,
          'Rb' : -0.963,
          'Cs' : -0.855,
          'Br' : -1.317,
          'Cl' : -1.465,
          'I'  : -1.344,
          'Ge' : -4.624,
          'Sn' : -3.895,
          'H'  : -3.394,
          'N'  : -8.122,
          'C'  : -9.217,
          'Pb' : -3.704
         }
    
    return mu
 
# scaled correction 
def getMuCorrectedDFT1():  
    
    # corrected mu (see thesis)
    mu = {'Na' : -1078.192,
          'K'  : -773.077,
          'Rb' : -665.270,
          'Cs' : -557.126,
          'Br' : -301.018,
          'Cl' : -359.035,
          'I'  : -2400.335,
          'Ge' : -1932.695, 
          'Sn' : -1697.547,
          'H'  : -1000, # false -> need to update
          'N'  : -1000, # false -> need to update
          'Pb' : -1000, # false -> need to update
          'C'  : -1000, # false -> need to update
         }

    return mu

# addition correction
def getMuCorrectedDFT2():  
    
    # corrected mu (see thesis)
    mu = {'Na' : -1159.145 + 0.091,
          'K'  : -773.077,
          'Rb' : -665.270,
          'Cs' : -557.126,
          'Br' : -367.073 + 0.289,
          'Cl' : -446.037 + 0.355,
          'I'  : -2695.019 + 0.165,
          'Ge' : -1932.695, 
          'Sn' : -1746.360 + 0.112,
          'H'  : -1000, # false -> need to update
          'N'  : -1000, # false -> need to update
          'Pb' : -1000, # false -> need to update
          'C'  : -1000, # false -> need to update
         }

    return mu

def getMuDFT():
    # get DFT calculated 'mu' (ev/atom)
    # energy = total energy / # of atoms in configuration
    # no spin included
    mu = {'Na' : -1159.145,
          'K'  : -773.077,
          'Rb' : -665.270,
          'Cs' : -557.126,
          'Br' : -367.073,
          'Cl' : -446.037,
          'I'  : -2695.019,
          'Ge' : -1932.695, 
          'Sn' : -1746.360,
          'H'  : -1000, # false -> need to update
          'N'  : -1000, # false -> need to update
          'Pb' : -1000, # false -> need to update
          'C'  : -1000, # false -> need to update
         }
    
    '''
    # same params as real data
    mu = {'Na' : -1159.159285,
          'K'  : -773.08861,
          'Rb' : -665.28203,
          'Cs' : -557.13367,
          'Br' : -367.0724775,
          'Cl' : -446.0363425,
          'I'  : -2695.0189125,
          'Ge' : -1932.654825, 
          'Sn' : -1746.33044,
          'H'  : -1000, # false -> need to update
          'N'  : -1000, # false -> need to update
          'Pb' : -1000, # false -> need to update
          'C'  : -1000, # false -> need to update
         }
    '''

    return mu