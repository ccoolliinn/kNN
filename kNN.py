from numpy import *
import operator
def creatDataset():
    group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels
def classify0(inX,dataset,labels,k):
    datasetsize=dataset.shape[0]
    diffmat=tile(inX,(datasetsize,1))-dataset
    sqDiffmat=diffmat**2
    sqDistances=sqDiffmat.sum(axis=1)
    distances=sqDistances**0.5
    sortedDistIndices=distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDistIndices[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]
def file2matrix(filename):  #读取文件数据
    fr=open(filename)
    arrayOLines=fr.readlines()
    numberOLines=len(arrayOLines)
    returnMat=zeros((numberOLines,3))
    classLabelVector=[]
    index=0
    for line in arrayOLines:
        line=line.strip()
        listFromLine=line.split('\t')
        returnMat[index,:]=listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index+=1
    return returnMat,classLabelVector
def autoNorm(dataset):#归一化数据
    minVals=dataset.min(0)
    maxVals=dataset.max(0)
    ranges=maxVals-minVals
    normDataset=zeros(shape(dataset))
    m=dataset.shape[0]
    normDataset=dataset-tile(minVals,(m,1))
    normDataset=normDataset/tile(ranges,(m,1))
    return normDataset,ranges,minVals
def datingClassTest():#测试算法
    hoRatio=0.10
    datingmat,datinglabels=file2matrix('datingTestSet2.txt')
    normmat,ranges,minVals=autoNorm(datingmat)
    m=normmat.shape[0]
    numTestVecs=int(m*hoRatio)
    errorCount=0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normmat[i,:],normmat[numTestVecs:m,:],datinglabels[numTestVecs:m],3)
        print("the classifierResult came back with: %d,the real answer is %d" % (classifierResult, datinglabels[i]))
        if(classifierResult!=datinglabels[i]): errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))

