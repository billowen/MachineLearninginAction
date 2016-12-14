from math import log
import operator


def calcShannonEnt(dataSet):
    labelCnt = {}
    for item in dataSet:
        curLabel = item[-1]
        if curLabel not in labelCnt.keys():
            labelCnt[curLabel] = 0
        labelCnt[curLabel] += 1
    num = len(dataSet)
    shannonEnt = 0.0
    for key, value in labelCnt.items():
        prob = float(value) / num
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def createDataSet():
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for item in dataSet:
        if item[axis] == value:
            tmpvec = item[:axis]
            tmpvec.extend(item[axis + 1:])
            retDataSet.append(tmpvec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVal = set(featList)
        newEntropy = 0.0
        for value in uniqueVal:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    classCnt = {}
    for vote in classList:
        if vote not in classCnt.keys():
            classCnt[vote] = 0
        classCnt[vote] += 1
    sortedClassCnt = sorted(
        classCnt.items(),
        key=operator.itemgetter(1),
        reverse=True)
    return sortedClassCnt[0][0]


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeature = chooseBestFeatureToSplit(dataSet)
    bestFeatureLabel = labels[bestFeature]
    myTree = {bestFeatureLabel: {}}
    featVals = [example[bestFeature] for example in dataSet]
    uniqueVal = set(featVals)
    for value in uniqueVal:
        subLabels = labels[:bestFeature]
        subLabels.extend(labels[bestFeature+1:])
        myTree[bestFeatureLabel][value] = createTree(
            splitDataSet(dataSet, bestFeature, value), subLabels)
    return myTree


if __name__ == "__main__":
    myData, labels = createDataSet()
    myTree = createTree(myData, labels)
    print(myTree)
