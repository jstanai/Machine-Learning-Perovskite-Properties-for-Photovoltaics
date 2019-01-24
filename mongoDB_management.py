#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jared
"""

import pandas as pd
import pymongo
import json
from os import listdir
from os.path import isfile, join
import multiprocessing as mp
import numpy as np
import dbConfig
import myConfig
import dummyCrystalBuilder as dc
from feature import getCompFeature
from feature import printMaxMinCells

def import_content(db, filename, collection):

    data = pd.read_csv(filename)
    data = data.dropna()
    data_json = json.loads(data.to_json(orient='records'))
    db[collection].insert_many(data_json)   
    
def update_database(db, folder, collection):
    
    filepaths = [f for f in listdir(folder) if 
                 (isfile(join(folder, f)) and f.endswith('.csv'))]

    db[collection].delete_many({})
    
    for filename in filepaths:
        import_content(db, folder + filename, collection)  

    print('Loading ' + str(db[collection].count()) + 
          ' items from ' + collection + '...')
     
    db[collection].aggregate([
        {
            "$lookup": {
                "from": collection,
                "localField": "crystal_id",    
                "foreignField": "crystal_id",  
                "as" : "fromItems"
            }
        },
        {
            "$replaceRoot": { "newRoot": { "$mergeObjects": 
                             [ { "$arrayElemAt": [ "$fromItems", 0 ] }, 
                            "$$ROOT" ] } }
        },
        { "$project": { "fromItems": 0 } }, 
        { "$out": collection + "_aggregated" }
    ]) 

def parallelize(df, numProcesses, func):
    
    df_split = np.array_split(df, numProcesses) 
    pool = mp.Pool(processes=numProcesses)
    
    results = pool.map(func, df_split)

    pool.close()
    pool.join()
    
    results_df = pd.concat(results)
    return results_df

def process_features(db, **kwargs):
    
    df =  pd.DataFrame(list(db['qw_outputs_aggregated'].find()))
    
    if dbConfig.dummy == True:
        df = dc.processDummyCrystals(df)  
    
    print('Processing Features... ')
    if kwargs['numProcesses'] > 1:
        feature = parallelize(df, kwargs['numProcesses'], getCompFeature)
    else:
        feature = getCompFeature(df) 
    
    if dbConfig.saveFeatures == True:
        feature.to_csv(dbConfig.saveFeaturesPath + 
                       dbConfig.saveFeaturesFile, index=False)
    
    print('Done.')

def getDB():
    
    client = pymongo.MongoClient(dbConfig.host, dbConfig.port)
    return(client['perovskites'])

def main():    
    
    db = getDB()

    update_database(db, dbConfig.crystalDBFolder, 'qw_outputs')
    process_features(db, numProcesses = 4)  
    update_database(db, dbConfig.featureDBFolder, 'features')  
    

if __name__ == "__main__":
    
    main()
    
    
