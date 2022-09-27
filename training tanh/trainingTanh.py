import pandas as pd
from createNodes import *
import random
import numpy as np
import math

# I HAVEMNT CHANGE THE OUTPUT NODE CHANGE IN WEIGHT ATTRIBUTE YET
# fucntion to generate hidden weight and biases
# and biases for output
def generate_hidden_weight_bias(noHiddenNodes, x_trainNumpy, y_trainNumpy, outPutrow=0):
    # noColumns is noColumns in training data
    noColumns = np.shape(x_trainNumpy)[1]
    hiddenLayer = []
    # hiddenLayerDict = {}
    # hidNodeNum = 0
    for i in range(noHiddenNodes):
        weight = random.uniform((-2 / noColumns), (2 / noColumns))
        bias = random.uniform((-2 / noColumns), (2 / noColumns))
        hidNode = hiddenNode(weight, bias, 0, 0,0)
        hiddenLayer.append(hidNode)
        # np.append(hiddenLayer,hidNode)

    # generate input nodes for each column
    inputNodes = []
    for i in range(noColumns):
        weights2Hidden = []

        for j in range(0, noHiddenNodes):
            weights2Hidden.append(random.uniform((-2 / noColumns), (2 / noColumns)))
            # weights2HiddenDict[j] = random.uniform((-2/noColumns),(2/noColumns))
        # inNode = inputNode(weights2HiddenDict,0,0,x_trainNumpy[0][i])
        inNode = inputNode(weights2Hidden, 0, 0, x_trainNumpy[0][i],0)
        inputNodes.append(inNode)

    # generate output node for this hidden layer
    outNode = outputNode(0, random.uniform((-2 / noColumns), (2 / noColumns)), 0, 0, y_trainNumpy[outPutrow])

    # return hiddenLayerDict,outNode,np.asarray(inputNodes)
    return np.asarray(hiddenLayer), outNode, np.asarray(inputNodes)


def forwardPass(hidLay, outNode, inputNodes):
    # calculating sJ (weighted sum)
    # for hidNode in hidlay:
    # iterates for lenght of number of hidden layers
    for i in range(len(hidLay)):
        hiddenWeightedSum = hidLay[i].bias
        for j in range(len(inputNodes)):
            # sum of inputs at THIS hidden node
            hiddenWeightedSum = hiddenWeightedSum + (float(inputNodes[j].weight[i]) * float(inputNodes[j].value))

        # tanh activation function
        #activation = 1 / (1 + math.exp(-hiddenWeightedSum))
        tanhActivation = (math.exp(hiddenWeightedSum) - math.exp(-hiddenWeightedSum)) / (math.exp(hiddenWeightedSum) + math.exp(-hiddenWeightedSum))
        hidLay[i].activation = tanhActivation

    # calculate output weighted sum

    outputWeightedSum = outNode.bias
    for i in range(len(hidLay)):
        outputWeightedSum = outputWeightedSum + (float(hidLay[i].weight) * float(hidLay[i].activation))

    # calculate activation for outNode

    outNode.activation = (math.exp(outputWeightedSum) - math.exp(-outputWeightedSum)) / (math.exp(outputWeightedSum) + math.exp(-outputWeightedSum))
    #outNode.activation = 1 / (1 + np.exp(-outputWeightedSum))


def backwardsPass(hidLay, outNode, inputNodes, stepSize):
    # calculate delta and bias for output nodes.
    outNode.delta = (outNode.value - outNode.activation) * (1-(outNode.activation**2))
    outNode.bias = outNode.bias + (stepSize * outNode.delta)

    # update and set the delta, bias and weight for hidden nodes using the formulas given
    for i in range(len(hidLay)):
        hidLay[i].delta = (float(hidLay[i].weight) * outNode.delta) * (1 - (hidLay[i].activation**2))

        hidLay[i].bias = hidLay[i].bias + (stepSize * hidLay[i].delta)
        hidLay[i].weight = hidLay[i].weight + (stepSize * outNode.delta * hidLay[i].activation)

        # update weights of input nodes
        for j in range(len(inputNodes)):
            weight = inputNodes[j].weight[i] + (
                    stepSize * float(hidLay[i].delta) * float(inputNodes[j].value))
            inputNodes[j].weight[i] = weight


# each node in MLP is a column of data,
# so i need to change the input nodes to the values in the next row of the
# corresponding column
def updateNodes(inputNodes, outNode, x_trainNumpy, y_trainNumpy, rowNumber):
    noColumns = np.shape(x_trainNumpy)[1]
    # for i in range(len(inputNodes)):
    #    inputNodes[i].value =

    for i in range(noColumns):
        # i represents a  column value
        inputNodes[i].value = x_trainNumpy[rowNumber][i]

    outNode.value = y_trainNumpy[rowNumber]


def testingNodes(x_testNumpy, y_testNumpy, outNode, hidLay, inputNodes, outputRow=0):
    noColumns = np.shape(x_testNumpy)[1]
    testInputNodes = []

    # creating training input Nodes
    # loop through columns.
    for i in range(noColumns):
        testing_weights_to_hidden = []
        for j in range(0,len(hidLay)):

            testing_weights_to_hidden.append(inputNodes[i].weight[j])

        testing_input = inputNode(testing_weights_to_hidden, 0, 0, x_testNumpy[0][i],0)
        testInputNodes.append(testing_input)

    # outnode = (weight,bias,activation,delta,value)
    # creating trainign output nodes
    # weights = weights from hidden layer to output
    # bias = bias from the last trainingData output
    # activation = what we're working out
    # delta = don't need
    testOutNode = outputNode(0, outNode.bias, 0, 0, y_testNumpy[0])

    return testOutNode, testInputNodes




def trainingData(inputNodes,hidLay,outNode,learningParameter,epochs,x_trainNumpy, y_trainNumpy):

    for j in range(0,epochs):
        updateNodes(inputNodes,outputNode,x_trainNumpy, y_trainNumpy,0)
        for i in range(0,len(x_trainNumpy)):

            forwardPass(hidLay, outNode, inputNodes)

            backwardsPass(hidLay, outNode, inputNodes,learningParameter)

            if i == 353:
                break
            else:
                updateNodes(inputNodes, outNode, x_trainNumpy, y_trainNumpy, i+1)

def testing(inputNodes,hidLay,outNode,x_testNumpy,y_testNumpy):
    error = []
    for i in range(0, len(x_testNumpy)):
        forwardPass(hidLay, outNode, inputNodes)
        error.append((outNode.value - outNode.activation) ** 2)
        if i == 117:
            break
        else:
            updateNodes(inputNodes, outNode, x_testNumpy, y_testNumpy, i+1)

    mse = sum(error) / len(error)
    return mse

def backPropAlgorithm(inputNodes,hidLay,outNode,learningParameter,epochs,x_trainNumpy, y_trainNumpy,x_testNumpy, y_testNumpy):
    mseList = []
    interval = int(epochs / (epochs / 10))
    noMse = int(epochs / 10)
    for i in range(0,noMse):

        trainingData(inputNodes, hidLay, outNode, learningParameter, interval, x_trainNumpy, y_trainNumpy)

        testOutNode, testInputNodes = testingNodes(x_testNumpy, y_testNumpy, outNode, hidLay, inputNodes, outputRow=0)

        mse = testing(testInputNodes,hidLay,testOutNode,x_testNumpy,y_testNumpy)

        mseList.append(mse)

    return mseList


def main():
    # TRAINING DATA
    read_xtrain_data = pd.read_csv("../x_train.csv")
    # read_train_data.to_numpy()
    x_trainData = read_xtrain_data.iloc[:, 1:]
    x_trainNumpy = pd.DataFrame.to_numpy(x_trainData)

    read_ytrain_data = pd.read_csv("../y_train.csv")
    # read_test_data.to_numpy()
    y_trainData = read_ytrain_data.iloc[:, 1:]
    y_trainNumpy = pd.DataFrame.to_numpy(y_trainData)

    # TESTING DATA
    read_xtest_data = pd.read_csv("../X_test.csv")
    x_testData = read_xtest_data.iloc[:, 1:]
    x_testNumpy = pd.DataFrame.to_numpy(x_testData)

    read_ytest_data = pd.read_csv("../y_test.csv")
    # read_test_data.to_numpy()
    y_testData = read_ytest_data.iloc[:, 1:]
    y_testNumpy = pd.DataFrame.to_numpy(y_testData)


    noHiddenNodes = 6
    hidLay, outNode, inputNodes = generate_hidden_weight_bias(noHiddenNodes, x_trainNumpy, y_trainNumpy, outPutrow=0)
    learningParameter = 0.1
    epochs = 6000

    mseList = backPropAlgorithm(inputNodes,hidLay,outNode,learningParameter,epochs,x_trainNumpy, y_trainNumpy,x_testNumpy, y_testNumpy)

    mseListNp = np.array(mseList)
    mse_df = pd.DataFrame(mseListNp)

    mse_df.to_excel("baseTanh_6_hid_4k_epo.xlsx")






main()
