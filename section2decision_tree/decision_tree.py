#!/usr/bin/env python
# -*- coding: utf-8 -*-

from math import log
from numpy import *


def calcShannonEnt(dataSet):
    '''计算样本实例的香农熵'''
    numEntries = len(dataSet)

    labelCounts = {}  # 字典存储每个分类的出现的次数

    for featVec in dataSet:
        currentLabel = featVec[-1]
        # 将样本标签提取出来并计数
        labelCounts[currentLabel] = labelCounts.get(currentLabel, 0) + 1
    Entropy = 0.0

    # 对每一个类别，计算样本中取到该类的概率
    # 最后将概率代入，求出熵
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        Entropy += prob * math.log(prob, 2)
    return (0 - Entropy)


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']
               ]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def splitDataSet(dataSet, axis, value):
    '''
    使用传入的axis和value划分数据集
    axis代表在每个列表中的第X位，value为用来划分的特征值
    '''
    retDataSet = []

    # 利用循环将不符合value的特征值划入另一集合
    # 相当于将value单独提取出来（或者作为叶节点）
    for feaVec in dataSet:
        if feaVec[axis] == value:
            reducedFeatVec = feaVec[:axis]

            reducedFeatVec.extend(feaVec[axis + 1:])
            # extend将VEC中的元素一一纳入feature_split
            retDataSet.append(reducedFeatVec)
            # append则将feature_split作为列表结合进目标集合

    return retDataSet


def chooseBestFeaturetoSplit(dataset):
    '''
    使用熵原则进行数据集合划分
    @信息增熵：info_gain = old - new
    @最优特征：best feature
    @类别集合：uniVal
    '''
    numFeatures = len(dataset[0]) - 1
    baseEntropy = calcShannonEnt(dataset)
    bestInfoGain = 0.0
    bestFeature = -1

    for i in range(numFeatures):
        featureList = [x[i] for x in dataset]
        # 使用set删除重复项，保留该特征对应的不同取值
        uniVal = set(featureList)
        ent_new = 0.0
        for v in uniVal:
            subset = splitDataSet(dataset, i, v)
            prob = float(len(subset)) / float(len(dataset))
            # 使用熵计算函数计算求出划分后的熵值
            ent_new += prob * calcShannonEnt(subset)

        infoGain = baseEntropy - ent_new
        if (infoGain > bestInfoGain):
            baseEntropy = ent_new
            bestFeature = i

    return bestFeature
