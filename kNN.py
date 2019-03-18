from numpy import *
import operator

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

def classify0(inX,dataSet,labels,k):
    #距离计算
    dataSetSize = dataSet.shape[0]    #numpy库的shanpe[0],获取矩阵的第一维的长度,dataSetSize=4
    diffMat = tile(inX,(dataSetSize,1)) - dataSet  #tile([0,0],(4,1)),列方向上重复一次，行方向重复4次
    sqDiffMat = diffMat ** 2    #2次方
    sqDistances = sqDiffMat.sum(axis=1)
    distances =sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount={}
    #选择距离最小的k个点
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    #排序
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]