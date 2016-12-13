from numpy import *
from os import listdir
import operator


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(
        classCount.items(),
        key=operator.itemgetter(1),
        reverse=True)
    return sortedClassCount[0][0]


def file2matrix(filename):
    fr = open(filename)
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOfLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def datingClassTest():
    hoRatio = 0.1
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(hoRatio * m)
    errorCnt = 0.0
    for i in range(numTestVecs):
        res = classify0(
            normMat[i, :],
            normMat[numTestVecs:m, :],
            datingLabels[numTestVecs:m],
            3)
        print(
            "the classifier came back with: %d, the real answer is: %d"
            % (res, datingLabels[i]))
        if res != datingLabels[i]:
            errorCnt += 1.0
    print("The total error rate is: %f" % (errorCnt / float(numTestVecs)))


def img2vector(filename):
    returnVector = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVector[0, 32 * i + j] = int(lineStr[j])
    return returnVector


def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNameStr = int(fileStr.split('_')[0])
        hwLabels.append(classNameStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    mTest = len(testFileList)
    errorCnt = 0.0
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNameStr = int(fileStr.split('_')[0])
        testVector = img2vector('testDigits/%s' % fileNameStr)
        res = classify0(testVector, trainingMat, hwLabels, 10)
        print(
            "the classifier came back with: %d, the real answer is: %d"
            % (res, classNameStr))
        if res != classNameStr:
            errorCnt += 1.0
    print('the total number of errors is: %d' % errorCnt)
    print('the total error rate is: %f' % (errorCnt / float(mTest)))


if __name__ == "__main__":
    # fig = plt.figure()
    # datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    # mpl.rcParams['font.sans-serif'] = ['SimHei']
    # mpl.rcParams['axes.unicode_minus'] = False
    # plt.figure(1)
    # plt.subplot(211)
    # plt.scatter(
    #     datingDataMat[:, 1],
    #     datingDataMat[:, 2],
    #     15.0 * array(datingLabels),
    #     15.0 * array(datingLabels))
    # plt.xlabel('玩视频游戏所耗费时间百分比')
    # plt.ylabel('每周消费的冰淇淋公斤数')
    # plt.subplot(212)
    # plt.scatter(
    #     datingDataMat[:, 0],
    #     datingDataMat[:, 1],
    #     15.0 * array(datingLabels),
    #     15.0 * array(datingLabels))
    # plt.xlabel('每年获得的飞行里程数')
    # plt.ylabel('玩视频游戏所耗费时间百分比')
    # plt.show()
    handwritingClassTest()
