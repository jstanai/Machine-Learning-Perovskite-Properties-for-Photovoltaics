#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Jared
"""
import csv
import itertools
from vnl import jobTemplates as jb 
import pandas as pd
from ast import literal_eval
from builder import perovskiteBuilder



# FILE CONFIGURATION

# file name
mastername = 'builderOutput_0-9'
path = './outputs/' # file location path
fname = mastername + '.csv'
jobFile = mastername + '_job.py'


outputFolder = path + 'jobs/' 
outputFile = outputFolder + jobFile.split('.')[0] + '_complete_'

dfFilter = False
startFilter = 0
stopFilter = 0

kpoints = 'na=6, nb=12, nc=6,'
densityMesh = '200*Hartree'



def main():
 
    headerSize = 6
    params = {}
    
    f = open(path + fname, 'r')

    reader = csv.reader(open(path + fname, 'rt'))
        
    keyCount = 0
    for row in reader:
        key = row[0]
        params[key] = (row[1])
        keyCount += 1
        if keyCount == headerSize - 1: break               
    f.close()

    # BUILD DATA FILE FROM DATAFRAME
    df1 = pd.read_csv(path + fname, skiprows = headerSize - 1)
    
    # For large dataFile  
    start = 500 #index in filename to start at
    stop = 998 #index in filename to stop at 
    n = 1 #number to do
    
    for i in range(n):
        num = stop - start + 1

        if dfFilter:
            df = df1.loc[startFilter:stopFilter,:].copy()
        else:
            df = df1.copy()

        df[['e_list']] = df['e_list'].apply(literal_eval)
        df['e_list'] = df['e_list'].apply(perovskiteBuilder.convertElementsShort2Long)

        f = open(outputFolder + jobFile.split('.')[0] + '_' + \
             str(start) + '-' + str(stop) + '.py', 'w')   
        
        index = 0
        
        for el in df['e_list']:
            el = str(el)
            el = el.replace("'", "")
                
            outputFile_edit = outputFile + str(start + index) + '_'
            
            #f.write(jb.getTemplateGGAQuantis(el, params, outputFile,  
            #        kpointsQuantis, densityMesh))
            f.write(jb.getTemplateGGA(el, params, outputFile_edit,  
                    kpoints, densityMesh))       
            #f.write(jb.getTemplatePW(el, params, outputFile,  
            #        kpoints, densityMesh))
            #f.write(jb.getTemplateLDA(el, params, outputFile, 
            #        kpoints, densityMesh))
            f.write(jb.addOptimizeGeometry(outputFile_edit, outputFile_edit))
            f.write(jb.addTotalEnergy(outputFile_edit))
            f.write(jb.addForces(outputFile_edit))
            f.write(jb.addStress(outputFile_edit))
            f.write(jb.addBandStructure(outputFile_edit))
            #f.write(jb.addBandStructurePW(outputFile_edit))       
            f.write(jb.addEffectiveMass(outputFile_edit))
            
            index += 1      
        f.close()
        
        start += num
        stop += num

if __name__ == '__main__':
    main()
    
    