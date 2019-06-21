#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def plotNode(nodeTxt, centerpt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt,
                            xy=parentPt,
                            xycoords='axes fraction',
                            xytext=centerpt,
                            textcoords='axes fraction',
                            va="center",
                            ha="center",
                            bbox=nodeType,
                            arrowprops=arrow_args)


def createPlot(inTree):
    # 类似于Matlab的figure，定义一个画布，背景为白色
    fig = plt.figure(1, facecolor='white')
    fig.clf()  # 把画布清空
    axprops = dict(xticks=[], yticks=[])
    # createPlot.ax1为全局变量，绘制图像的句柄，subplot为定义了一个绘图，
    # 111表示figure中的图有1行1列，即1个，最后的1代表第一个图
    # frameon表示是否绘制坐标轴矩形
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()


def getNumLeafs(myTree):
    '''计算此树的叶子节点数目'''
    numLeafs = 0
    firstStr = myTree.keys()
    firstStr = list(firstStr)[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(mytree):
    '''计算此树的总深度'''
    maxDepth = 0
    firstStr = mytree.keys()
    firstStr = list(firstStr)[0]
    secondDict = mytree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        # 对于树，每次判断value是否为字典，若为字典则进行递归，否则计数器+1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth


def retriveTree(i):
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                   {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
                   ]
    return listOfTrees[i]


def plotMidText(cntrPt, parentPt, txtString):
    '''
    作用是计算tree的中间位置
    cntrpt起始位置,parentpt终止位置,txtstring：文本标签信息
    '''
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)


def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getNumLeafs(myTree)
    firstStr = myTree.keys()
    firstStr = list(firstStr)[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    # 计算子节点的坐标
    plotMidText(cntrPt, parentPt, nodeTxt)  # 绘制线上文字
    plotNode(firstStr, cntrPt, parentPt, decisionNode)  # 绘制节点
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    # 每绘制一次图，将y的坐标减少1.0/plottree.totald，间接保证y坐标上深度的
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD
