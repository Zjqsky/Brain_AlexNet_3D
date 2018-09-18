#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import nibabel as nib
import os
import pandas as pd
import random
from skimage import transform


def loadData(fileDir, groupMap, Xshape):
    def load_nii(fileName):
        data = nib.load(fileName).get_data().astype(int)
        data = transform.resize(data, Xshape)
        return data

    def load_csv(fileName):
        data = pd.read_csv(fileName)
        data['Group'] = data['Group'].map(groupMap)
        data = np.array(data.values)[:,[0,2]]
        m = {}
        for item in data:
            m[item[0]] = [item[1]]
        return m

    fileCount = 1
    dataMap = load_csv(fileDir + '/ADNI1_Annual_2_Yr_3T_7_12_2018.csv')
    for root, dirs, files in os.walk(fileDir):
        for file in files:
            fileSplit = file.split('.')
            if fileSplit[-1] == 'nii':
                print("read nii_file:%d" % fileCount)
                fileCount += 1
                fileFullName = root + '/' + file
                dataMap[int(fileSplit[0].split('_')[-1][1:])].append(load_nii(fileFullName))

    data = []
    for item in dataMap:
        if len(dataMap[item]) == 2:
            data.append(dataMap[item])

    random.shuffle(data)
    dataX = []
    dataY = []
    for item in data:
        dataX.append(item[1])
        dataY.append(item[0])

    dataX = np.array(dataX)
    dataY = np.array(dataY)
    return [dataX, dataY]


if __name__ == '__main__':
    m1 = {}
    m2 = {}
    m3 = {}
    groupMap = {'CN':0, 'MCI':1, 'AD':2}
    Xshape = (111, 111, 111)
    dataX, dataY = loadData('D:/postGraWork/MedicalCompetition/data/ADNI1_Annual_2_Yr_3T', groupMap, Xshape)
    print(dataX.shape, dataY.shape)
    for i in dataX:
        m1[i.shape[0]] = True
        m2[i.shape[1]] = True
        m3[i.shape[2]] = True
    print('m1:')
    for i in m1:
        print(i)
    print('m2:')
    for i in m2:
        print(i)
    print('m3:')
    for i in m3:
        print(i)
