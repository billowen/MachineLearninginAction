import matplotlib.pyplot as plt
import trees


decisionNode = dict(boxstyle="square", fc="0.9")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def plotNode(nodeText, centerPt, parentPt, nodeType):
    plt.subplot(111, frameon=False, xticks=[], yticks=[])
    plt.annotate(
        nodeText, xy=parentPt, xycoords='axes fraction',
        xytext=centerPt, textcoords='axes fraction',
        va="center", ha="center", bbox=nodeType,
        arrowprops=arrow_args)


def getTreeDepth(myTree):
    maxDepth = 0
    firststr = list(myTree)[0]
    subDict = myTree[firststr]
    for key in subDict.keys():
        if type(subDict[key]) is dict:
            thisDepth = 1 + getTreeDepth(subDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


def getNumLeafs(myTree):
    numLeaf = 0
    firstStr = list(myTree)[0]
    subDict = myTree[firstStr]
    for key, value in subDict.items():
        if type(value) is dict:
            numLeaf += getNumLeafs(value)
        else:
            numLeaf += 1
    return numLeaf


leafStep = 0.0
yStep = 0.0
xOffset = 0.2
yOffset = 1.0


def plotMidText(centerPt, parentPt, msg):
    xMid = (centerPt[0] + parentPt[0]) / 2.0
    yMid = (centerPt[1] + parentPt[1]) / 2.0
    plt.subplot(111, frameon=False, xticks=[], yticks=[])
    plt.text(xMid, yMid, msg)


def plotTree(myTree, parentPt, msg):
    global yOffset
    global xOffset
    numLeaf = getNumLeafs(myTree)
    centerPt = (xOffset + leafStep * (numLeaf - 1) / 2.0, yOffset)
    plotMidText(centerPt, parentPt, msg)
    plotNode(list(myTree)[0], centerPt, parentPt, decisionNode)
    subDict = myTree[list(myTree)[0]]
    yOffset -= yStep
    for key, value in subDict.items():
        if type(value) is dict:
            plotTree(value, centerPt, str(key))
        else:
            plotMidText((xOffset, yOffset), centerPt, str(key))
            plotNode(value, (xOffset, yOffset), centerPt, leafNode)
            xOffset += leafStep
    yOffset = yStep + yOffset


def createPlot(myTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    global leafStep
    global yStep
    global xOffset
    global yOffset
    leafStep = 0.6 / float(getNumLeafs(myTree) - 1)
    yStep = 1.0 / float(getTreeDepth(myTree))
    xOffset = 0.2
    yOffset = 1.0
    plotTree(myTree, (0.5, 1.0), '')
    plt.show()


def classify(inputTree, featLabels, testVec):
    headStr = list(inputTree)[0]
    subDict = inputTree[headStr]
    print(featLabels)
    featIndex = featLabels.index(headStr)
    for key, value in subDict.items():
        if testVec[featIndex] == key:
            if type(value) is dict:
                classLabel = classify(value, featLabels, testVec)
            else:
                classLabel = value
    return classLabel


if __name__ == "__main__":
    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    print(lensesLabels)
    lensesTree = trees.createTree(lenses, lensesLabels)
    print(lensesLabels)
    createPlot(lensesTree)
    print(classify(lensesTree, lensesLabels, lenses[0]))
