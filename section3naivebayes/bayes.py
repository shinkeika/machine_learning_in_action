#!/usr/bin/env python
# -*- coding: utf-8 -*-
from numpy import *


def loadDataSet():
    '''
        postingList: 进行词条切分后的文档集合
        classVec:类别标签
        使用伯努利模型的贝叶斯分类器只考虑单词出现与否（0，1）
    '''
    postingList = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'hie', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]
    classVec = [0, 1, 0, 1, 0, 1]  # 1代表侮辱性文字，0代表正常言论
    return postingList, classVec


def createVocabList(dataset):
    vocabSet = set([])
    for document in dataset:
        vocabSet = vocabSet | set(document)  # 通过对两个集合取并，找出所有非重复的单词
    return list(vocabSet)


def setOfWord2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    # 创建与词汇表等长的列表向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the world: %s is not in my Vocabulary" % word)
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    '''
        利用NumPy数组计算p(wi|c1)
        词条 属于类1的概率Prob_positive = p(c1)
        因为是二分类所以属于类0概率 =1-p(c1)
    '''
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    # 获取输入文档（句子）数以及向量的长度
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = ones(numWords)  # 创建一个长度为词条向量等长的列表
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num / p1Denom)
    p0Vevt = log(p0Num / p0Denom)
    return p0Vevt, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    '''
    :param vec2Classify: 待测试的词条分类向量
    :param p0Vec: 类别0在所有文档中各个词条出现的频数(P(Wi|c0))
    :param p1Vec: 类别1在所有文档中各个词条出现的频数(P(Wi|c1))
    :param pClass1:类别为1的文档占文档总数比例
    '''
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWord2Vec(myVocabList, postinDoc))

    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))

    testEntry = ['love', 'my', 'dalmation']

    thisDoc = array(setOfWord2Vec(myVocabList, testEntry))

    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))

    testEntry = ['stupid', 'garabe']

    thisDoc = array(setOfWord2Vec(myVocabList, testEntry))

    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))


def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


def textParse(bigString):
    import re
    listOfTokens = re.split(br'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spamTest():
    '''
    处理数据长字符串
    对长字符串进行分割，分割符为除单词和数字之外的任意符号串
    # 将分割后的字符串中所有的大些字母变成小写lower(),并且只
    # 保留单词长度大于3的单词
    '''
    # 新建三个列表
    docList = []
    classList = []
    fullText = []

    for i in range(1, 26):
        # 打开并读取指定目录下的本文中的长字符串，并进行处理返回
        wordList = textParse(open('email/spam/%d.txt' % i, 'rb').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i, 'rb').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    # 将所有邮件中出现的字符串构建成字符串列表
    vocabList = createVocabList(docList)
    # 构建一个大小为50的整数列表和一个空列表
    trainingSet = list(range(50))
    testSet = []
    # 随机选取1~50中的10个数，作为索引，构建测试集
    for i in range(10):
        # 随机选取1~50中的10个数，作为索引，构建测试集
        randIndex = int(random.uniform(0, len(trainingSet)))
        # 将选出的数的列表索引值添加到testSet列表中
        testSet.append(trainingSet[randIndex])
        # 从整数列表中删除选出的数，防止下次再次选出
        # 同时将剩下的作为训练集
        del(trainingSet[randIndex])
    # 新建两个列表
    trainMat = []
    trainClasses = []
    # 遍历训练集中的每个字符串列表

    for docIndex in trainingSet:
        # 将字符串列表转为词条向量，然后添加到训练矩阵中
        trainMat.append(setOfWord2Vec(vocabList, docList[docIndex]))
        # 将该邮件的类标签存入训练类标签列表中
        trainClasses.append(classList[docIndex])
    # 计算贝叶斯函数需要的概率值并返回
    p0v, p1v, pSpam = trainNB0(array(trainMat), array(trainClasses))

    errcount = 0

    # 遍历测试集中的字符串列表
    for docIndex in testSet:
        # 同样将测试集中的字符串列表转为词条向量
        wordVector = setOfWord2Vec(vocabList, docList[docIndex])
        # print(wordVector)
        # 对测试集中字符串向量进行预测分类，分类结果不等于实际结果
        if classifyNB(array(wordVector), p0v, p1v, pSpam) != classList[docIndex]:
            errcount += 1

    print('the error rate is:', float(errcount) / len(testSet))
