# -*- coding: utf-8 -*-
from numpy import *
import operator
from os import listdir


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


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals  # 计算极差
    # 下一步将初始化一个与原始数据矩阵同尺寸的的矩阵
    # 利用tile函数实现扩充向量，并进行元素之间的对位运算
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def datingClassTest():
    hoRadio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRadio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :],
                                     normMat[numTestVecs:m, :],
                                     datingLabels[numTestVecs:m],
                                     3)
        # 参数1表示从测试集（此处约会数据是随机的，因此抽取前10%即可）
        # 参数2，3，4使用90%作为训练数据，为输入的实例进行投票并分类，K=3

        print('the classfier come back with: %d, the real answer is : %d'
              % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print('the total error rate is : %f' % (errorCount / float(numTestVecs)))


def classifyPerson():
    resultList = ['not at all', 'in small does', 'in large does']
    percentsTats = float(input('percentage of time spent playing video games?'))
    ffMiles = float(input('frequent flier miles earned per year?'))
    iceCream = float(input('liter of ice cream consumed per year?'))
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inarray = array([percentsTats, ffMiles, iceCream])
    classiferResult = classify0(((inarray - minVals) / ranges), normMat, datingLabels, 3)
    print('you will probably like this person:', resultList[classiferResult - 1])


def img2vector(filename):
    '''this is to将32*32的图像转化为1*1024的行向量'''
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        linestr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(linestr[j])
    # returnVec 按照32进位，j代表每位有32个元素
    return returnVect


def handwritingClassTest(traindir, testdir):
    hwLabels = []

    # 将目录内的文件按照名字放入列表，使用函数解析为数字
    trainFileList = listdir(traindir)
    m = len(trainFileList)
    trainingMat = zeros((m, 1024))

    for i in range(m):
        fileNameStr = trainFileList[i]
        filestr = fileNameStr.split('.')[0]
        classNumStr = int(filestr.split('_')[0])
        # 比如'digits/testDigits/0_13.txt'，被拆分为0,13,txt
        # 此处0即为标签数字
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector("%s/%s" % (traindir, fileNameStr))
    # labels is label_vec，同之前的KNN代码相同，存储标签

    testFilelist = listdir(testdir)
    errorCount = 0.0
    mTest = len(testFilelist)
    for i in range(mTest):
        fileNameStr = testFilelist[i]
        filestr = fileNameStr.split('.')[0]
        classNumStr = int(filestr.split('_')[0])
        vectorUnderTest = img2vector("%s/%s" % (testdir, fileNameStr))
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print('the classifier come back with : %d,the real answer is : %d'
              % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1
    print("the total number of errors is: %d" % errorCount)
    print("the total error rate is : %f" % (errorCount/float(mTest)))
