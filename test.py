import kNN
from numpy import *
import operator
import time

group,labels = kNN.createDataSet()
print(group)
print(labels)
testdata = tile([0,0],(4,1)) - group
print(group)
print(group ** 2)
testdata2 = group ** 2
testdata3 = sum(testdata2,axis=1)
print(testdata3)
testdata4 =testdata3 ** 0.5
print(testdata4)
testdata5 = testdata4.argsort()
print(testdata5)
classCount = {}
print('***********')
for i in range(3):
    print(i)
    print(testdata5[i])
    voteIlabel = labels[testdata5[i]]
    print(voteIlabel)
    classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    print(classCount[voteIlabel])
    print('*********')
print(classCount)
print('--------')
print(classCount.items())
print(operator.itemgetter(1))
sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
print('--------')
print(sortedClassCount)
print(sortedClassCount[0][0])