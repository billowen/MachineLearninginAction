from numpy import *


def loadDataSet():
    postingList = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec


def createVocabList(dataSet):
    vocabSet = set([])
    for doc in dataSet:
        vocabSet = vocabSet | set(doc)
    return list(vocabSet)


def setOfWords2Vec(vocabSet, inputSet):
    returnVec = [0] * len(vocabSet)
    for word in inputSet:
        if word in vocabSet:
            returnVec[vocabSet.index(word)] = 1
        else:
            print('the word: %s is not in my vocabulary!' % word)
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    numOfTrainDocs = len(trainMatrix)
    numOfWord = len(trainMatrix[0])
    pAbusiv = sum(trainCategory) / float(numOfTrainDocs)
    p0num = ones(numOfWord)
    p1num = ones(numOfWord)
    p0denom = 2.0
    p1denom = 2.0
    for i in range(numOfTrainDocs):
        if trainCategory[i] == 1:
            p1num += trainMatrix[i]
            p1denom += sum(trainMatrix[i])
        else:
            p0num += trainMatrix[i]
            p0denom += sum(trainMatrix[i])
    p1vect = log(p1num / p1denom)
    p0vect = log(p0num / p0denom)
    return p0vect, p1vect, pAbusiv


def classifyNB(vec2Classify, p0vec, p1vec, pclass1):
    p1 = sum(vec2Classify * p1vec) + log(pclass1)
    p0 = sum(vec2Classify * p0vec) + log(1.0 - pclass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testNB():
    listOfPost, listClass = loadDataSet()
    vocabSet = createVocabList(listOfPost)
    trainMat = []
    for doc in listOfPost:
        trainMat.append(setOfWords2Vec(vocabSet, doc))
    p0vec, p1vec, pAbusiv = trainNB0(trainMat, listClass)
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(vocabSet, testEntry))
    print(
        testEntry,
        'classified as',
        classifyNB(thisDoc, p0vec, p1vec, pAbusiv))


if __name__ == '__main__':
    testNB()
