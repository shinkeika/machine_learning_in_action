#!/usr/bin/env python
# -*- coding: utf-8 -*-
from numpy import *


# 解析约会数据文件，并将数据导入一个numpy矩阵
def file2matrix(fielname):
    fr = open(fielname)
    arrayOfLines = fr.readlines()
    numoflines = len(arrayOfLines)
    # 初始化数据的为m行3列（飞行里程，游戏时间，冰淇淋数）
    returnMat = zeros((numoflines, 3))
    # 标签单独创建一个向量保存
    classLabelVector = []
    index = 0
    for line in arrayOfLines:
        line = line.strip()
        listfromLine = line.split('\t')  # 按照换行符分割数据
        # 将文本数据的前三行存入数据矩阵，第四列存入标签向量
        returnMat[index, :] = listfromLine[0:3]
        classLabelVector.append(int(listfromLine[-1]))
        index += 1
    return returnMat, classLabelVector
