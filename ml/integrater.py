#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Jared
"""
from itertools import product
import numpy as np
import math
from collections import Counter
            
def getDist(p1, p2):
    c2 = [(p2[j] - p1[j])**2 for j, i in enumerate(p1)]
    return(math.sqrt(sum(c2)))

def pdf_element_density(elements, positions, r, dr, la, lb, lc, r_cut):
    
    el = Counter(elements) 
    
    elements_d_c_total = {}
     
    for element in el:      
         
        pos = [positions[i] for i, e in enumerate(elements) if e == element]
        els = [e for e in elements if e == element] 
        
        d_c_total = [0]*len(r) 
        
        ea_bound = np.arange(-r_cut[0], r_cut[0] + 1, 1)*la
        eb_bound = np.arange(-r_cut[1], r_cut[1] + 1, 1)*lb
        ec_bound = np.arange(-r_cut[2], r_cut[2] + 1, 1)*lc
        
        # FOR EACH ATOM LOCATION IN CELL
        for i, e1 in enumerate(elements):           
            
            # FIND DENSITY FROM ELEMENT e2
            for j, e2 in enumerate(els):
                
                p1 = pos[j]
                
                ea = p1[0] + ea_bound
                eb = p1[1] + eb_bound
                ec = p1[2] + ec_bound
                
                p = list(product(ea, eb, ec))
            
                d = [getDist(positions[i], p_try) for p_try in p]
                d_c = [di for di in np.histogram(d, bins = r)[0]]
                d_c_total = [d_c_total[i] + d_c[i] for i, v in enumerate(d_c)]
            
        
            elements_d_c_total[element] = d_c_total    
            
            totalA = len(elements) # TOTAL NUMBER OF P1-CONTAINING ELEMENTS (NUMBER OF E1)
            
            # NORMALIZING FUNCTION f(m): m is metric 
            volume_factors = [(4./3)*math.pi*((ri + dr)**3 - ri**3) for ri in r]

            normalized_P2_density = [v/(totalA*volume_factors[i]) for 
                                     i, v in enumerate(d_c_total)]
        
            elements_d_c_total[element] = normalized_P2_density
            
    return(elements_d_c_total)
