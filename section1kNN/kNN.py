# -*- coding: utf-8 -*-
from numpy import *
import operator


def createDataset():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    # 这一行利用tile函数将输入样本实例转化为与训练集同尺寸的矩阵
    # 方便之后的矩阵减法运算

    sqDiffMat = diffMat ** 2
    sqDistance = sqDiffMat.sum(axis=1)
    distance = sqDistance ** 0.5
    sortedDisIndicies = distance.argsort()
    # 对每个训练样本与输入的测试样本求欧几里得距离，即点之间的范数
    # 随后按照距离由小到大进行排序


    classCount = {}
    for i in range(k):
        votIlabel = labels[sortedDisIndicies[i]]
        classCount[votIlabel] = classCount.get(votIlabel, 0) + 1
    # 记录距离最小的前K个类，并存放入列表。key对应标签，value对应样本。

    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
