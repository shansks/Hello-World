from numpy import *
import operator

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

#inX为输入的目标，dataset是样本的矩阵，label是标签，k是需要取得个数
def classify0(inX,dataSet,labels,k):
    #距离计算,使用欧式公式
    dataSetSize = dataSet.shape[0]    #numpy库的shanpe[0],获取矩阵的第一维的长度,dataSetSize=4，读取矩阵的行数，也就是样本数量
    print('dataSetSize:',dataSetSize)
    diffMat = tile(inX,(dataSetSize,1)) - dataSet  #tile([0,0],(4,1)),列方向上重复一次，行方向重复4次
    print("diffMat:",diffMat)
    sqDiffMat = diffMat ** 2    #2次方
    print("sqDiffMat:",sqDiffMat)
    sqDistances = sqDiffMat.sum(axis=1)   #按照欧式公式进行相加,axis=0位按列求和，axis为按行求和
    print("sqDistances:",sqDistances)
    distances =sqDistances ** 0.5    #按欧式公式进行求根[1.4866... 1.414....  0    0.1]，计算距离
    print("distances:",distances)
    sortedDistIndicies = distances.argsort()  #argsort()对原队列排序，并提取对应的index[2 3 1 0],按大小逆序排序
    print("sortedDistIndicies:",sortedDistIndicies)
    classCount={}
    #选择距离最小的k个点
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    print("classCount:",classCount)
    #排序,items返回可遍历的（键，值）的元组数组,operator.itemgetter(1)根据第二个域进行排序，reverse=True表示降序，False表示正序
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    print("sortedClassCount:",sortedClassCount)
    return sortedClassCount[0][0]