#!/usr/bin/env python
# -*- coding: utf-8 -*-

from numpy import *


def loadDataSet():
    '''
    打开文本
    '''
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():  # 每行读取
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])  # 每行前两个值分别是X1和X2 X0是为了方便计算
        labelMat.append(int(lineArr[2]))  # 第三个值是数据对应的类别标签
    return dataMat, labelMat


def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))


def gradAscent(dataMatIn, classLabels):
    '''
    梯度上升法
    :param dataMatIn: 2维的numpy矩阵 每列代表每个不同的特征。每行代表每个训练样本 100*3
    :param classLabels: 1*100的行向量
    :return:
    '''
    dataMatrix = mat(dataMatIn)  # 转换numpy矩阵数据类型
    labelMat = mat(classLabels).transpose()  # 为了便于矩阵运算，需要将行向量转换为列向量
    m, n = shape(dataMatrix)  # 100 3
    alpha = 0.001  # 向目标移动的步长
    maxCycles = 500  # maxCycles是迭代次数
    weights = ones((n, 1))  # 设置梯度上升算法的参数
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        # print(dataMatrix * weights)  # 100*3 . 3*1  = 100 * 1
        # exit()
        # TODO 需要搞懂
        error = (labelMat - h)
        # grad(x) = (y - f(x)) * x'
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights


dataArr, labelMat = loadDataSet()
gradAscent(dataArr, labelMat)
