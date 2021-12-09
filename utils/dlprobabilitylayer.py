# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 09:57:22 2021

@author: ink
"""
import os
import numpy as np
import gdal
import pandas as pd
from raster_io import array_to_tiff
from sklearn.ensemble import RandomForestClassifier

def secondmax_of_list(list1):
    # list of numbers
    # list1 = [10, 20, 4, 45, 99]
     
    # new_list is a set of list1
    new_list = set(list1)
     
    # removing the largest element from temp list
    new_list.remove(max(new_list))
     
    # elements in original list are not changed
    # print(list1)
     
    return max(new_list)
def distance_of_max_smax(probarray):
    # input probarray is 6*512*512
    class_num, rows, cols=probarray.shape
    #transform to 512*512*6
    bands = np.dstack([probarray[i] for i in range(class_num)])
    
    distarr = np.zeros((rows,cols))
    
    for x in range(cols):
        for y in range(rows):
            data = set(bands[x][y])
            firstmax = max(data)
            data.remove(max(data))
            secondmax = max(data)
            dis = firstmax-secondmax
            distarr[x][y]=dis
    return distarr

def find_similar_probability(distarray,percentile):
    # input probarray is 512*512
    rows, cols=distarray.shape
    
    mark = np.zeros((rows,cols), dtype=bool)
    threshold = np.quantile(distarray, percentile)
    for x in range(cols):
        for y in range(rows):
            dis = distarray[x][y]
            if dis<threshold:
                mark[x][y]=1
    return mark

def mark_hard_to_classify_by_probarray():
    root_dir = 'E:/tmp/ergc_test/unet_predict'
    file_list = list(filter(lambda x: x.startswith('prob') and x.endswith('.tif') , os.listdir(root_dir)))
    for filename in file_list:
        file = os.path.join(root_dir, filename)
        ds = gdal.Open(file)
        
        array = ds.ReadAsArray()
        distarray = distance_of_max_smax(array)
        mark = find_similar_probability(distarray, 0.2)
        mark=mark.astype('int8')
        array_to_tiff(mark, ds.GetProjection(), ds.GetGeoTransform(), 1, os.path.join(root_dir,filename.replace('prob','maxd')))

def probabilities_of_falsepred():
    root_dir = 'E:/tmp/ergc_test/unet_predict'
    file_list = list(filter(lambda x: x.startswith('falsepred') and x.endswith('.tif') , os.listdir(root_dir)))
    for filename in file_list:
        file = os.path.join(root_dir, filename)
        ds = gdal.Open(file)
        
        mask = ds.ReadAsArray()
        probf = file.replace('falsepred','prob')
        ds1=gdal.Open(probf)
        probarray=ds1.ReadAsArray()
        
        class_num, rows, cols=probarray.shape
        #transform to 512*512*6
        bands = np.dstack([probarray[i] for i in range(class_num)])
        
        tmp = np.zeros((rows,cols,3))
        for x in range(cols):
            for y in range(rows):
                if mask[x][y]==0:
                    dlist = list(bands[x][y])
                    dlist.sort(reverse=True)
                    tmp[x][y]=dlist[:3]
        tmp = np.transpose(tmp,(2,0,1))
        array_to_tiff(tmp, ds.GetProjection(), ds.GetGeoTransform(), 3, os.path.join(root_dir,filename.replace('falsepred','max3c')))
def probability_falsepred_classifier():
    root_dir = 'E:/tmp/ergc_test/unet_predict'
    file_list = list(filter(lambda x: x.startswith('prob') and x.endswith('924_1386.tif') , os.listdir(root_dir)))
    feat=[]
    train_y=[]
    for filename in file_list:
        # filename prob_top_potsdam_2_10_RGBIR_924_1386.tif
        file = os.path.join(root_dir, filename)
        ds = gdal.Open(file)
        probarray = ds.ReadAsArray()
        class_num, rows, cols=probarray.shape
        
        falsepredf=file.replace('prob','falsepred')
        fpds=gdal.Open(falsepredf)
        fparr =fpds.ReadAsArray()
        
        #transform to 512*512*6
        bands=np.transpose(probarray,(1,2,0))
        
        tmp = np.zeros((rows,cols,class_num))
        for x in range(cols):
            for y in range(rows):
                if (fparr[x][y]==1 and  np.random.rand()<0.25) or fparr[x][y]==0:
                    data = set(bands[x][y])
                    maxv = max(data)
                    tmp[x][y]=maxv-bands[x][y]
                    feat.append(tmp[x][y])
                    train_y.append(fparr[x][y])
    feat=np.array(feat)
    train_x = feat.reshape(-1, 6)

    training_X = pd.DataFrame(train_x)
    training_y = pd.DataFrame(train_y)
    clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=9, random_state=0)
    clf.fit(training_X, training_y)
    
    # pickle.dump(clf, open(model_npy, 'wb'))
    predf = os.path.join(root_dir,'prob_top_potsdam_2_10_RGBIR_462_1386.tif')
    predfds=gdal.Open(predf)
    nparr=predfds.ReadAsArray()
    bands=np.transpose(nparr,(1,2,0))
    feat = []
    for i in range(cols):
        for j in range(rows):
            feat.append(bands[i][j])
    feat = np.array(feat)
    df = pd.DataFrame(feat)
    
    # clf = pickle.load(open(model, 'rb'))
    r = clf.predict(df)
    classified_array = r.reshape(rows, cols)
    array_to_tiff(classified_array, predfds.GetProjection(), predfds.GetGeoTransform(), 1, os.path.join(root_dir,'rf_prob_hardc_462_1386.tif'))
    
if __name__ == '__main__':
    mark_hard_to_classify_by_probarray()
    # probability_falsepred_classifier()
    