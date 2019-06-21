#!/usr/bin/env python
# -*- coding: utf-8 -*-

from math import log
from numpy import *
import operator


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


def majorityCnt(classlist):
    classCount = {}
    for vote in classlist:
        classCount[vote] = classCount.get(vote, 0) + 1
    sortedCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True);
    # 获取每一类出现的节点数（没出现默认为0）并进行排序
    # 返回最大项的KEY对应的类别
    return sortedCount[0][0]


def create_tree(dataset, label):
    classlist = [example[-1] for example in dataset]
    # 类别完全相同则停止划分
    if classlist.count(classlist[0]) == len(classlist):
        return classlist[0]
    if len(dataset[0]) == 1:
        return majorityCnt(classlist)
    bestFeat = chooseBestFeaturetoSplit(dataset)
    bestFeatLabel = label[bestFeat]
    myTree = {bestFeatLabel: {}}

    subLabels = label[:]
    # 删除属性列表中当前分类数据集特征
    del (subLabels[bestFeat])
    # 使用列表推导式生成该特征对应的列
    featValues = [example[bestFeat] for example in dataset]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        # 递归创建子树并返回
        myTree[bestFeatLabel][value] = create_tree(splitDataSet(dataset, bestFeat, value), subLabels)

    return myTree


def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()
    firstStr = list(firstStr)[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


def store_tree(inp_tree, filename):
    import pickle
    with open(filename, 'wb+') as fp:
        pickle.dump(inp_tree, fp)


def grab_tree(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)


def graplensesData():
    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lensesTree = create_tree(lenses,lensesLabels)
    return lensesTree
