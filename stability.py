#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 15:23:46 2018

@author: Jared
"""



from collections import Counter
import pymongo
import pandas as pd
from ast import literal_eval
from ml.elements import *
#import machineLearner as ml #get rid of if doing oqmd database
#from qmpy import * #use python 2.7!
from matplotlib import pyplot as plt
import math
#import mysql.connector
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.patches as mpatches
import matplotlib
import matplotlib.gridspec as gridspec


    # ENERGY OF FORMATION
    # dH = totalEnergy - sum(i,x){x*E_i}, x number of atoms of that type
    
    # STABILITY  (E_HULL)
    # dH_stab = dH - dH_hull
    # dH_hull (ev/atom), but calculated a different way than our energy of formation
    # We need 
    
    # Access Syntax for direct acces to DB 
'''
    cnx = mysql.connector.connect(user='root', password='Twinkles',
                                  host='127.0.0.1',
                                  database='qmpy_jared')
    cursor = cnx.cursor()
    cursor.execute("USE qmpy_jared;")
    cursor.close()
    cnx.close()
'''
    
    # DEFINITIONS FOR OQMD DATA
'''
    space = 'Cs-Sn-Br'  
    comp = 'CsSnBr3'
    space = PhaseSpace(space)
    energy, phase = space.gclp(comp)
    compute_stability
    print(energy, phase)
'''

def main():
    
    matplotlib.rcParams.update({'font.size': 15.5})
    # QUICK LOAD TO AVOID CALCULATION
    path = '/Users/Jared/Dropbox/Master Thesis/code/codeOutputs/'
    deltaH_qmpy = pd.read_csv(path + 'compEnergy_qmdb_d3.csv') 
    print('qmpy ', len(deltaH_qmpy))
    mng_client = pymongo.MongoClient('localhost', 27017)
    db = mng_client['perovskites']
    
    # GET AGGREGATED CRYSTAL DATA FROM MONGODB
    df =  pd.DataFrame(list(db['qw_outputs_aggregated'].find()))
    #df = pd.read_csv('/Users/Jared/Dropbox/Master Thesis/Data/crystalDB3/aggregated_features_14092018.csv')
    df_features = pd.read_csv('/Users/Jared/Dropbox/Master Thesis/Data/featureDB2/d2_paper_24102018.csv')

    '''
    plt.ylabel('$E_{gap}$ (eV)')
    plt.xlabel('Iodine Mixing Fraction')
    plt.title('Iodine Bandgap Trend')
    
    s = 'fracI'
    s2 = 'dir_gap'
    y_cl = df_features.groupby([s])[s2].mean()
    x_cl = np.array([i for i in y_cl.index])
    y_cl = y_cl.values
    
    plt.scatter(df_features[s], df_features[s2], alpha = 0.2)
    p1, = plt.plot(x_cl, y_cl, linestyle = '-', lw = 2, label = 'D$_{3}$')
    ax1 = plt.axes()
    ax1.yaxis.set_major_locator(plt.MaxNLocator(6))
    #plt.legend(handles = [p1])
    plt.tight_layout()
    path = '/Users/Jared/Dropbox/Master Thesis/code/codeOutputs/'
    plt.savefig(path + 'dummyTrend_realI.png', dpi = 400, bbox_inches="tight")
    plt.show()
    '''

    
    #df = df.dropna(axis = 0)
    dff = df.drop(df[df['nIterations'] >= 201].index).copy()
    dff = dff.drop(df[df['crystal_id'] == 1526850748].index).copy()
    df = dff.drop(df[df['crystal_id'] == 1526752626].index).copy()
    print('here', len(df))
    #deltaH_qmdb = getCrystalOQMDData(df)
    
    # MY CALCULATED FORMATION ENERGY
    mu = getMuCorrectedDFT2()
    deltaH2_formation = getDeltaH_formation(df, mu)
    mu = getMuDFT()
    deltaH_formation = getDeltaH_formation(df, mu)

    #df_delta = pd.DataFrame(deltaH_formation, columns = 'dH_formation')
    #deltaH_formation.to_csv('/Users/Jared/Dropbox/Master Thesis/code/codeOutputs/df_formation.csv')
    #plotDeltaH_formation(list(deltaH_formation['deltaH_formation']))
    

    

    
    # GEOMETRIC FORMATION ENERGY (BASED ON FIT)
    #deltaH_geo = getDeltaH_geo(df)
    #deltaH_geo.to_csv('/Users/Jared/Dropbox/Master Thesis/' + 
    #                'code/codeOutputs/deltaH_geo.csv')
    deltaH_geo = pd.read_csv('/Users/Jared/Dropbox/Master Thesis/' + 
                    'code/codeOutputs/deltaH_geo.csv')
    print('geo', len(deltaH_geo))
    #plotDeltaH_geo(list(deltaH_geo['deltaH_geo']))
    
    
    # comparison of geometric approach fidelity
    '''
    plt.plot(deltaH_geo['descriptor'], deltaH['deltaH'], 'o')
    plt.xlabel('$(t + \mu)^{\eta}$')
    plt.ylabel('$\Delta H_{f}$ (eV/atom)')
    plt.title('Formation Energy vs. Geometric Factor')
    plt.show()
    '''
    
    #error associated with SG15 basis set
    #delta = ((10.78 + 8.19 + 7.69 + 0.19)*(4/20) + 
    #        (4.35 + 8.07)*(4/20) + 
    #        (1.9 + 6.03 + 5.53)*(8/20)) 
    
    # MERGE ALL DATA
    result = pd.merge(deltaH_formation, deltaH_qmpy, on=['crystal_id'])
    result = pd.merge(result, deltaH_geo, on=['crystal_id'])
    result= pd.merge(result, df_features, on=['crystal_id'])

    result_corrected = pd.merge(deltaH2_formation, deltaH_qmpy, on=['crystal_id'])
    result_corrected = pd.merge(result_corrected, deltaH_geo, on=['crystal_id'])
    result_corrected = pd.merge(result_corrected, df_features, on=['crystal_id'])
    sresult = result_corrected
    '''
    result = result[result.crystal_id != 1519471915]
    result = result[result.crystal_id != 1519608323]
    result = result[result.crystal_id != 1519429441]
    result = result[result.crystal_id != 1520265350]
    result = result[result.crystal_id != 1520268226]
    result = result[result.crystal_id != 1520334800]
    result = result[result.crystal_id != 1520343157]
    result = result[result.crystal_id != 1520349833]
    result = result[result.crystal_id != 1520411007]
    result = result[result.crystal_id != 1520429554]
    result = result[result.crystal_id != 1520442584]
    result = result[result.crystal_id != 1520483780]
    '''

    # big plot
    
    my_dpi = 500
    fig = plt.figure(figsize=(5, 5), dpi=my_dpi)
    
    m = np.array((list(result['deltaH_formation'] - result['deltaH_hull'])))
    m = m.mean()
    m = 0.150 # 100 mev line

    ymin = 1.12*min(result['deltaH_hull']) if min(result['deltaH_hull']) <=0 else 0.88*min(result['deltaH_hull'])
    ymax = 1.12*max(result['deltaH_hull']) if max(result['deltaH_hull']) >=0 else 0.88*max(result['deltaH_hull'])
    xmax = ymax

    plt.ylim(ymin, ymax)
    plt.xlim(ymin, xmax)
    
    xy = [min(result['deltaH_hull']), max(result['deltaH_hull'])]
    xy = [ymin, ymax]
    p1, = plt.plot(xy, xy, color = 'k', label = '$E_{hull}$') 
    p0c, = plt.plot(result['deltaH_hull'],
                    result_corrected['deltaH_formation'], 'o', 
                    alpha = 0.5, color = 'r', label = '$\mu_{corrected}$')
    
    p0, = plt.plot(result['deltaH_hull'],
                   result['deltaH_formation'], 'o', 
                   alpha = 0.5, label = '$\mu$')
    
    
    #p1, = plt.plot(xy, xy, color = 'k', label = '$E_{hull}$') 
    
    #xy = [min(result['deltaH_hull']), max(result['deltaH_hull'])]
    
    #p2, = plt.plot(xy, [i + m for i in xy], alpha = 1.0, 
    #               color = 'k', 
    #               label = '$\Delta E_{hull}$ = 100 meV',
    #               linestyle = '--', linewidth = 3.0) 
    
    plt.xlabel('$\Delta H_{f, OQMD}$ (eV/atom)')
    plt.ylabel('$\Delta H_{f}$ (eV/atom)')
    plt.title('Convex Hull Distance', y = 1.04)
    
    plt.legend(handles = [p0c, p0, p1])
    
    ax1 = plt.axes()
    ax1.xaxis.set_major_locator(plt.MaxNLocator(6))
    ax1.yaxis.set_major_locator(plt.MaxNLocator(6))
    ax1.tick_params(bottom = True, top = True, left = True, right = True, 
                direction = 'in')
    
    plt.savefig(path + 'paper_oqmdb_new1.png', dpi=400, bbox_inches="tight")
    plt.show()
    '''
    # hist plot
    c, d, e = plt.hist(list(result['deltaH_formation'] - result['deltaH_hull']), bins = 21)
    plt.setp(e, edgecolor='w', lw=1, alpha = 0.7)
    #plt.title('Stability of ' + str(len(result)) + ' Compounds')
    #plt.xlabel('$E_{hull}$ distance (eV)')
    #plt.ylabel('Count')
    
    c, d, e = plt.hist(
            list(result_corrected['deltaH_formation'] - 
                 result['deltaH_hull']), bins = 21, color = 'r')
    plt.setp(e, edgecolor='w', lw=1, alpha = 0.7)
    plt.title('D$_{3}$ Hull Distance')
    plt.xlabel('$\Delta E_{hull}$ (eV)')
    plt.ylabel('Count')
    
    ax1 = plt.axes()
    ax1.tick_params(bottom = True, top = True, left = True, right = True, 
                direction = 'in')
    
    plt.savefig(path + 'oqmdb_new1.png', dpi=400, bbox_inches="tight")
    
    plt.show()
    
    '''
    
    
    
    #sresult = result_corrected.copy() #result_corrected[['fracCl','fracBr', 
                               # 'fracI', 'fracCs', 
                                #'fracRb', 'fracNa', 
                                #'fracK', 'fracSn', 
                               # 'fracGe', 'deltaH_hull']]
    #plt.scatter(result['fracCl'], result['deltaH_hull'])
    #print(sresult['t'])
    print(len(sresult))
    
    #
    #
    # lattice validity
    t1 = 2*(sresult['lb'].values)/(sresult['la'].values)
    t2 = 2*(sresult['lb'].values)/(sresult['lc'].values)
     
    '''
    blue_patch = mpatches.Patch(color='blue', label='2*lb/la')
    red_patch = mpatches.Patch(color='red', label='2*lb/lc')
     
    c2, d2, e2 = plt.hist(t1, bins = 21, color = 'b')
    plt.setp(e2, edgecolor='w', lw=1, alpha = 0.7)
    
    c1, d1, e1 = plt.hist(t2, bins = 21, color = 'r')
    plt.setp(e1, edgecolor='w', lw=1, alpha = 0.7)
    
    plt.legend(handles=[blue_patch, red_patch])
    
    plt.title('D$_{3}$ Perovskite Validity')
    plt.xlabel('Lattice Vector Ratio')
    plt.ylabel('Count')
    plt.show()
    '''
    
    sresult['hullDistance'] = list(result_corrected['deltaH_formation'] - 
                                   result_corrected['deltaH_hull'])
    sresult['deltaH_formation'] = list(result_corrected['deltaH_formation'])
    
    
    
    '''
    #
    #
    # goldshmitd vs dhhull
    plt.scatter(sresult['t'].values, sresult['hullDistance'].values)
    plt.show()
    
    #
    #
    # goldschmidt validity
    #plt.hist(sresult['t'].values)
    c1, d1, e1 = plt.hist(sresult['t'].values, bins = 21)
    plt.setp(e1, edgecolor='w', lw=1)
    plt.title('D$_{3}$ Perovskite Validity')
    plt.xlabel('Goldschmidt Tolerance Factor')
    plt.ylabel('Count')
    plt.show()
    '''   
    
    
    
    
    
    plt.ylabel('$\Delta E_{hull}$ (eV)')
    plt.xlabel('Sodium Mixing Fraction')
    plt.title('Sodium $\Delta E_{hull}$ Trend')
    
    s = 'fracNa'
    s2 = 'hullDistance'
    y_cl = sresult.groupby([s])[s2].mean()
    x_cl = np.array([i for i in y_cl.index])
    y_cl = y_cl.values
    
    plt.scatter(sresult[s], sresult[s2], alpha = 0.2)
    plt.plot(x_cl, y_cl, linestyle = '-', lw = 2, label = 'D$_{3}$')
    ax1 = plt.axes()
    ax1.yaxis.set_major_locator(plt.MaxNLocator(6))
    ax1.xaxis.set_major_locator(plt.MaxNLocator(6))
    
    ax1.tick_params(bottom = True, top = True, left = True, right = True, 
                direction = 'in')
    #plt.legend(handles = [p1])
    plt.tight_layout()
    path = '/Users/Jared/Dropbox/Master Thesis/code/codeOutputs/'
    plt.savefig(path + 'dummyTrend_realNa.png', dpi = 400, bbox_inches="tight")
    plt.show()
    
    
    print(xx)
    
    
    
    
    # run each of these with d3 data
    
    s = 'dir_gap'
    sname = '$E_{gap}$ (eV)'
    fname = 'eGap'
    
    '''
    s = 'deltaH_hull'
    s = 'hullDistance'
    sname = '$\Delta E_{hull}$ (eV)'
    fname = 'dh_hull'
    
    s = 'deltaH_formation'
    sname = '$\Delta H_{f}$ (eV/atom)'
    fname = 'dh_form'
    
    # goldschmidt
    s = 't' #'dir_gap'
    sname = '$t$'
    fname = 'gold'
    
    #lattice
    sresult['t1'] = t2
    s = 't1' #'dir_gap'
    sname = '2*lb/la'
    fname = '2lbla'
    '''
    glist = [g for g in sresult.groupby(['fracCl'])[s]]
    print(type(sresult[s].values[0]))
    y_cl = sresult.groupby(['fracCl'])[s].mean()
    y_cl_sd = sresult.groupby(['fracCl'])[s].std()
    x_cl = np.array([i for i in y_cl.index])
    y_cl = y_cl.values
 
    y_br = sresult.groupby(['fracBr'])[s].mean()
    y_br_sd = sresult.groupby(['fracBr'])[s].std()
    x_br = np.array([i for i in y_br.index])
    y_br = y_br.values
    
    y_i = sresult.groupby(['fracI'])[s].mean()
    y_i_sd = sresult.groupby(['fracI'])[s].std()
    x_i = np.array([i for i in y_i.index])
    y_i = y_i.values
    
    y_cs = sresult.groupby(['fracCs'])[s].mean()
    y_cs_sd = sresult.groupby(['fracCs'])[s].std()
    x_cs = np.array([i for i in y_cs.index])
    y_cs = y_cs.values
    
    y_rb = sresult.groupby(['fracRb'])[s].mean()
    y_rb_sd = sresult.groupby(['fracRb'])[s].std()
    x_rb = np.array([i for i in y_rb.index])
    y_rb = y_rb.values
    
    y_na = sresult.groupby(['fracNa'])[s].mean()
    y_na_sd = sresult.groupby(['fracNa'])[s].std()
    x_na = np.array([i for i in y_na.index])
    y_na = y_na.values
    
    y_k = sresult.groupby(['fracK'])[s].mean()
    y_k_sd = sresult.groupby(['fracK'])[s].std()
    x_k = np.array([i for i in y_k.index])
    y_k = y_k.values
    
    y_sn = sresult.groupby(['fracSn'])[s].mean()
    y_sn_sd = sresult.groupby(['fracSn'])[s].std()
    x_sn = np.array([i for i in y_sn.index])
    y_sn = y_sn.values
    
    y_ge = sresult.groupby(['fracGe'])[s].mean()
    y_ge_sd = sresult.groupby(['fracGe'])[s].std()
    x_ge = np.array([i for i in y_ge.index])
    y_ge = y_ge.values
    
    y = (sresult['deltaH_hull'].values)
    
    #scatter_matrix(sresult, alpha=0.2, figsize=(6, 6), diagonal='kde')
    
    #f, ax = plt.subplots(3, sharey = True)
    
    plt.figure(figsize = (5,15.2))

    gs1 = gridspec.GridSpec(3, 1)
    gs1.update(wspace=0.0, hspace=0.0)
    cs = 8
    alpha = 0.3
    
    ax1 = plt.subplot(gs1[0])
    ax1.tick_params(bottom = True, top = True, left = True, right = True, 
                direction = 'in')
    
    ax1.set_ylabel(sname)
    ax1.scatter(x_cl, y_cl, color = 'C0')
    ax1.plot(x_cl, y_cl, linestyle = '--', label = 'Cl', color = 'C0')
    ax1.errorbar(x_cl, y_cl, yerr=y_cl_sd, 
                 capsize = cs, fmt='none', color = 'C0', alpha = alpha)
    
    ax1.scatter(x_br, y_br, color = 'C1')
    ax1.plot(x_br, y_br, linestyle = '--', label = 'Br', color = 'C1')
    ax1.errorbar(x_br, y_br, yerr=y_br_sd, 
                 capsize = cs, fmt='none', color = 'C1', alpha = alpha)
    
    ax1.scatter(x_i, y_i, color = 'C2')
    ax1.plot(x_i, y_i, linestyle = '--', label = 'I', color = 'C2')
    ax1.errorbar(x_i, y_i, yerr=y_i_sd, 
                 capsize = cs, fmt='none', color = 'C2', alpha = alpha)
    
    label = 'X-Site'
    ax1.annotate(label, (0.78, 0.87), xycoords='axes fraction', va='center')
    ax1.legend(loc = 2)
    ax1.set_title('D$_{3}$ Stability Trends')
    #ax1.xlabel('Mxing Fraction')
    #ax1.ylabel(sname)
    #plt.savefig(path + fname + '-x-site-stability.png', dpi=400, bbox_inches="tight")
    #plt.show()
    
    
    
    ax2 = plt.subplot(gs1[1])
    ax2.tick_params(bottom = True, top = True, left = True, right = True, 
                direction = 'in')
    ax2.set_ylabel(sname)
    ax2.scatter(x_sn, y_sn, color = 'C0')
    ax2.plot(x_sn, y_sn, linestyle = '--', label = 'Sn', color = 'C0')
    ax2.errorbar(x_sn, y_sn, yerr=y_sn_sd, 
                 capsize = cs, fmt='none', color = 'C0', alpha = alpha)
    
    ax2.scatter(x_ge, y_ge, color = 'C1')
    ax2.plot(x_ge, y_ge, linestyle = '--', label = 'Ge', color = 'C1')
    ax2.errorbar(x_ge, y_ge, yerr=y_ge_sd, 
                 capsize = cs, fmt='none', color = 'C1', alpha = alpha)
    
    label = 'B-Site'
    ax2.annotate(label, (0.78, 0.87), xycoords='axes fraction', va='center')
    ax2.legend(loc = 2)
    #plt.title('D$_{3}$ Stability Trends (B-site)')
    #plt.xlabel('Mixing Fraction')
    #plt.ylabel(sname)
    #plt.savefig(path + fname + '-b-site-stability.png', dpi=400, bbox_inches="tight")
    #plt.show()
    
    ax3 = plt.subplot(gs1[2])
    ax3.tick_params(bottom = True, top = True, left = True, right = True, 
                direction = 'in')
    ax3.set_ylabel(sname)
    ax3.scatter(x_rb, y_rb, color = 'C0')
    ax3.plot(x_rb, y_rb, linestyle = '--', label = 'Rb', color = 'C0')
    ax3.errorbar(x_rb, y_rb, yerr=y_rb_sd, 
                 capsize = cs, fmt='none', color = 'C0', alpha = alpha)
    
    ax3.scatter(x_cs, y_cs, color = 'C1')
    ax3.plot(x_cs, y_cs, linestyle = '--', label = 'Cs', color = 'C1')
    ax3.errorbar(x_cs, y_cs, yerr=y_cs_sd, 
                 capsize = cs, fmt='none', color = 'C1', alpha = alpha)
    
    ax3.scatter(x_na, y_na, color = 'C2')
    ax3.plot(x_na, y_na, linestyle = '--', label = 'Na', color = 'C2')
    ax3.errorbar(x_na, y_na, yerr=y_na_sd, 
                 capsize = cs, fmt='none', color = 'C2', alpha = alpha)
    
    ax3.scatter(x_k, y_k, color = 'C3')
    ax3.plot(x_k, y_k, linestyle = '--', label = 'K', color = 'C3')
    ax3.errorbar(x_k, y_k, yerr=y_k_sd, 
                 capsize = cs, fmt='none', color = 'C3', alpha = alpha)
    
    
    label = 'A-Site'
    ax3.annotate(label, (0.78, 0.87), xycoords='axes fraction', va='center')
    ax3.legend(loc = 2)
    plt.xlabel('Mixing Fraction')
    
    #plt.title('D$_{3}$ Stability Trends (A-site)')
    #plt.xlabel('Mixing Fraction')
    #plt.ylabel(sname)
    #plt.savefig(path + fname + '-a-site-stability.png', dpi=400, bbox_inches="tight")
    plt.savefig(path + fname + '-t-trend.png', dpi=400, bbox_inches="tight")
    plt.show()
    
    #plt.scatter(x_br, y)
    #plt.scatter(x_i, y)
    #axarr[0].plot(x, y)
    #axarr[0].set_title('Sharing X axis')
    #axarr[1].scatter(x, y)
    #ax[0].set_title('Simple plot')
    
    #plt.scatter(x, y)

    '''
    c, d, e = plt.hist(list(deltaH_formation['deltaH_formation']), bins = 21)
    plt.setp(e, edgecolor='w', lw=1)
    plt.title('Formation Energy of ' + str(len(deltaH_formation)) + ' Compounds')
    plt.xlabel('$\Delta H_{f}$ (eV/atom)')
    plt.ylabel('Count')
    plt.savefig(path + 'oqmdb_new2.png', dpi=400, bbox_inches="tight")
    plt.show()
    '''
    
    
    # VARIANCE FROM CUBIC STRUCTURE
    '''
    plt.plot(1000*(result['deltaH_formation'] - result['deltaH_hull']), 'og')
    plt.title('Energy above $E_{hull}$ (meV)')
    plt.show()
    
    plt.plot(deltaH_geo['t'], 'p')
    plt.title('Goldschmidt Tolerance Factor')
    plt.show()
    
    plt.plot(100*abs(1 - deltaH_geo['b/a']), 'p')
    plt.title('b/a lattice distortion percent (%)')
    plt.show()
    
    plt.plot(100*abs(1 - deltaH_geo['c/a']), 'p')
    plt.title('c/a lattice distortion percent (%)')
    plt.show()
    '''
    
def calculateOQMDData(row_elements):
    
    counts = Counter(convertElementsLong2Short(
                         literal_eval(row['elements'])))
    
    comp = ''
    space = ''
    s = 0.0
    for el in counts:
        comp += str(el) + str(counts[el])
        s += counts[el]
            
        space += str(el) + '-'
          
    space = space[0:-1] 
        
def getCrystalOQMDData(df):
      
    #df = df.copy().head()
    fname = 'compEnergy_qmdb_d3_junk.csv'
    path = '/Users/Jared/Dropbox/Master Thesis/code/codeOutputs/' 
    
    print('Computing compositional energy')
 
    # PRE-ALLOCATE DATAFRAME        
    df_columns = ['crystal_id',
                  'totalEnergy_hull',
                  'deltaH_hull',
                  'phase_sum',
                  'pEnergy',
                  'compSum',
                  'stablePhase']
                  
    compEnergy = pd.DataFrame(index = range(df.shape[0]), 
                              columns = df_columns)
      
    deltaH = []
    for index, row in df.iterrows():
        
        print(index + 1, 'out of', len(df))
        
        counts = Counter(convertElementsLong2Short(
                         literal_eval(row['elements'])))
    
        # build composition A1B2C3, and space A-B-C strings for qmpy
        comp = ''
        space = ''
        s = 0.0
        for el in counts:
            comp += str(el) + str(counts[el])
            s += counts[el]
            
            space += str(el) + '-'
          
        space = space[0:-1] 
        
        
        space = PhaseSpace(space)
        energy, phase = space.gclp(comp)
        #mu = getMuDFT()
        mu = getMuCorrectedDFT2()
        
        phase_sum = 0
        pEnergy = []
        compSum = []
        for p in phase: # CsI, GeI2 are phases of CsGeI3 (dummy example)
            
            comp_sum = 0
            for el in p.comp: # each element, ex: Cs, I, in CsI
                # total energy * number of that element in this compostion
                comp_sum += mu[str(el)]*p.comp[el] 
                
            pEnergy += [p.energy] # the energy of that phase (eV/atom)
            compSum += [comp_sum] # the total energy of that phase's constituents
            # still need the formation energy... but I don't calculate it for all these 
            # compounds.. that is what we are trying to extrapolate
            
            
            # NEED TO NORMALIZE BY NUMBER OF ATOMS
            phase_sum += comp_sum # phase_sum is total energy of all phases,
            #should just be same as compSum from deltaH_formationenergy
            
        #plt.plot(pEnergy, compSum)   
            ##total_energy_estimated += sum(
             #       [mu[str(el)]*int(p.comp[el])*int(phase[p]) 
             #        for el in p.comp])
        
        #print(total_energy_estimated/s)
        
        deltaH += [energy/s]
        
        stablePhase = (len(phase) == 1)
    
        compEnergy.loc[index] = [str(row['crystal_id']), 
                                 energy, 
                                 energy/s, 
                                 phase_sum,
                                 str(pEnergy),
                                 str(compSum),
                                 stablePhase]
    
        print(str(pEnergy))
        
    with open(path + fname, 'w') as f: 
        
        compEnergy.to_csv(f, index = None)

    #plt.show()
    
    return(deltaH)
    
        
    
def plotDeltaH_formation(deltaH):
    
    c, d, e = plt.hist(deltaH, bins = 20)
    plt.setp(e, edgecolor='w', lw=1)
    plt.xlabel('$\Delta H_{f}$ (eV/atom)')
    plt.ylabel('Number of Compounds')
    #plt.axvline(x=0.0, color='r', linestyle='-')
    plt.title('Formation Energy of Database')
    plt.plot() 
    plt.show()    
    
def plotDeltaH_geo(deltaH):
    
    # positive H stable
    
    plt.plot(deltaH, 'o')
    plt.xlabel('Index')
    plt.ylabel('deltaH (Predicted)')
    plt.title('Stability Prediction: $-1.987 + 1.66(t + \mu)^{\eta}$')
    plt.axhline(y=0.0, color='r', linestyle='-')
    stableNum = len([a for a in deltaH if a >= 0])
    
    print('\n\n ' + str(stableNum) + ' stable compounds of ' + str(len(deltaH)) + 
          ' total. (' + str(round(100*stableNum/len(deltaH), 2)) + '%)')
    savePath = '/Users/Jared/Dropbox/Master Thesis/code/codeOutputs/'
    plt.savefig(savePath + 'stability01.png', pad_inches=0.4, 
                dpi=200, bbox_inches='tight')
    plt.plot()  
    plt.show()
    
    
    c, d, e = plt.hist(deltaH, bins = 21)
    print('here2')
    plt.setp(e, edgecolor='w', lw=1)
    plt.xlabel('$\Delta H_{d}$ (eV/atom)')
    
    plt.ylabel('Number of Compounds')
    plt.axvline(x=0.0, color='r', linestyle='-')
    plt.title('Stability Prediction: $-1.987 + 1.66(t + \mu)^{\eta}$')
    plt.plot() 
    plt.savefig(savePath + 'stability02.png', pad_inches=0.4, 
                dpi=200, bbox_inches='tight')
    plt.show()
    
def getDeltaH_geo(df):
    
    # SPECIFY POSSIBLE SITE OCCUPATIONS
    Asites = ['Cs', 'Rb', 'Na', 'K']
    Bsites = ['Sn', 'Ge']
    Xsites = ['I', 'Cl', 'Br']
    
    # IONIC RADIAL PROPERTY SPECIFICATION FOR CALCULATIONS 
    # PRE-ALLOCATE DATAFRAME        
    
    df_columns = ['crystal_id', 'deltaH_geo', 
                  't', 'mu', 'eta', 'descriptor',
                  'b/a', 'c/a']
                  
    deltaH_df = pd.DataFrame(index = range(df.shape[0]), 
                             columns = df_columns)
    
    for index, row in df.iterrows():
        
        #print('GEO: ', index, ' of ', len(df) - 1)
        
        # GET VOLUME
        v = literal_eval(row['cellPrimVectors_end']) # GET LATTICE VECTORS
        la = v[0]
        lb = v[4]
        lc = v[8]
        volume = la*lb*lc
        
        # GET ELEMENT COUNTS
        counts = Counter(convertElementsLong2Short(
                         literal_eval(row['elements'])))
        
        # GET IONIC RADII OF ELEMENTS
        
        #properties = ['X', 'EI1', 'rp', 'rs', 'ra', 'EI2',
        #              'l', 'h', 'fermi', 'ir']
    
        #properties = ml.getPropertyMixerLabels(properties)
        A_ir = []
        B_ir = []
        X_ir = []
        for e in counts:
            v = counts[e] # NUMBER OF ELEMENTS OF THIS TYPE
            if e in Asites:
                A_ir += [getElementFeature()[e]['ir']]*v
            if e in Bsites:
                B_ir += [getElementFeature()[e]['ir']]*v
            if e in Xsites:
                X_ir += [getElementFeature()[e]['ir']]*v

        # GET EFFECTIVE IONIC RADII BY AVERAGING OVER ALL
        # A, B, AND X SITE ATOMS
        Aeff = sum(A_ir)/4.0
        Beff = sum(B_ir)/4.0
        Xeff = sum(X_ir)/12.0
        
        # GET TOTAL VOLUME OF ELEMENTS
        A_volumes = sum([(4./3)*math.pi*r**3 for r in A_ir])
        B_volumes = sum([(4./3)*math.pi*r**3 for r in B_ir])
        X_volumes = sum([(4./3)*math.pi*r**3 for r in X_ir])
        
        # GOLDSCHMIDT TOLERANCE  
        t = (Aeff + Xeff)/(math.sqrt(2.0)*(Beff + Xeff))  
        
        # OCTAHEDRAL
        mu = Beff/Xeff 
        
        # PACKING FACTOR
        eta = sum([A_volumes, B_volumes, X_volumes])/volume
        
        descriptor = (t + mu)**eta
        # PREDICTED DECOMPOSITION ENERGY (POSITIVE MEANS STABLE)
        #https://pubs.acs.org/doi/suppl/10.1021/
        #jacs.7b09379/suppl_file/ja7b09379_si_001.pdf
        deltaH_geo = -1.987 + 1.660*(t + mu)**eta #(eV)
        
        deltaH_df.loc[index] = [row['crystal_id'], deltaH_geo,
                                t, mu, eta, descriptor, 
                                2*lb/la, lc/la]

        # since deltaH_p positive means stable
    print('done')
    return deltaH_df

def calculateDeltaH_formation(row_elements, row_totalEnergy, mu):
    # GET ELEMENT COUNTS
    counts = Counter(convertElementsLong2Short(
                     literal_eval(row_elements)))  
    
    #mu = getMuDFT() #eV

    
    mus = [mu[el]*counts[el] for el in counts]       
    muSum = sum(mus) #eV

    deltaH_result = (row_totalEnergy - muSum)/len(literal_eval(row_elements))

    return deltaH_result, muSum

def getDeltaH_formation(df, mu):
    
    
    # PRE-ALLOCATE DATAFRAME        
    df_columns = ['crystal_id', 
                  'totalEnergy', 
                  'deltaH_formation', 
                  'compSum_formation']
                  
    deltaH_df = pd.DataFrame(index = range(df.shape[0]), 
                             columns = df_columns)
    
    for index, row in df.iterrows():
              
        #print('FORMATION: ', index, ' of ', len(df) - 1)
        
        '''
        # GET ELEMENT COUNTS
        counts = Counter(convertElementsLong2Short(
                         literal_eval(row['elements'])))
                
        #if(counts['Br'] == 12 and counts['Cs'] == 4 and counts['Sn'] == 4):
        
        mu = getMuDFT()
        mus = [mu[el]*counts[el] for el in counts]
           
        muSum = sum(mus)
        
        #print(row['directGap'], volume, row['totalEnergy'],-1*mus)
        #print(row['volume'])
        #print(counts, (row['totalEnergy'] - mus)/20.)
        deltaH_result = (row['totalEnergy'] - muSum)/20.
        ''' 
        
        deltaH_result, muSum = calculateDeltaH_formation(row['elements'],
                                                         row['totalEnergy'],
                                                         mu)
            
        
        # GET VOLUME
        '''
        v = literal_eval(row['cellPrimVectors_end']) # GET LATTICE VECTORS
        la = v[0]
        lb = v[4]
        lc = v[8]
        volume = la*lb*lc
        '''
        
        deltaH_df.loc[index] = [row['crystal_id'], 
                                row['totalEnergy'],
                                deltaH_result, 
                                muSum]
        
    path = '/Users/Jared/Dropbox/Master Thesis/code/codeOutputs/'
    deltaH_df.to_csv(path + 'formationEnergy.csv')
    
    return deltaH_df  




if __name__ == '__main__':
    main()