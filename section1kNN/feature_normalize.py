#!/usr/bin/env python
# -*- coding: utf-8 -*-

from numpy import *


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals #  计算极差
    # 下一步将初始化一个与原始数据矩阵同尺寸的的矩阵
    # 利用tile函数实现扩充向量，并进行元素之间的对位运算
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals
