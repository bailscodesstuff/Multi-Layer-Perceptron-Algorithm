import pandas as pd
from createNodes import *
import random
import numpy as np
import math


# function to intialise hidden nodes and input nodes and generate random weights and biases for them.
def generate_hidden_weight_bias(noHiddenNodes, x_trainNumpy, y_trainNumpy, outPutrow=0):
    # noColumns is number of columns in training data
    noColumns = np.shape(x_trainNumpy)[1]
    hiddenLayer = []

    # generates the hidden nodes
    # this includes generating the weights and biases
    for i in range(noHiddenNodes):
        # weights and biases were both randomly generated using a uniform distribution
        # between values [-2/n,2/n] where n is the number of columns.
        weight = random.uniform((-2 / noColumns), (2 / noColumns))
        bias = random.uniform((-2 / noColumns), (2 / noColumns))
        # instatiate a hidden node object
        hidNode = hiddenNode(weight, bias, 0, 0, 0)
        # append each object to an array
        hiddenLayer.append(hidNode)

    # generate input nodes for each column
    inputNodes = []
    for i in range(noColumns):
        weights2Hidden = []
        # generates the weights from input nodes to hidden node#
        # weights and biases were both randomly generated using a uniform distribution
        # between values [-2/n,2/n] where n is the number of columns.
        for j in range(0, noHiddenNodes):
            weights2Hidden.append(random.uniform((-2 / noColumns), (2 / noColumns)))
        # instantiate an inputNode object using the weights generated above
        inNode = inputNode(weights2Hidden, 0, 0, x_trainNumpy[0][i], [0] * noHiddenNodes)
        # append each object to an array
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

        activation = 1 / (1 + math.exp(-hiddenWeightedSum))
        hidLay[i].activation = activation

    # calculate output weighted sum

    outputWeightedSum = outNode.bias
    for i in range(len(hidLay)):
        outputWeightedSum = outputWeightedSum + (float(hidLay[i].weight) * float(hidLay[i].activation))

    # calculate activation for outNode

    outNode.activation = 1 / (1 + np.exp(-outputWeightedSum))


def backwardsPass(hidLay, outNode, inputNodes, stepSize):
    # calculate delta and bias
    outNode.delta = (outNode.value - outNode.activation) * (1 - (outNode.activation ** 2))
    outNode.bias = outNode.bias + (stepSize * outNode.delta)
    # update the weights for input nodes
    # for momentum, I need to keep track of the change in weights.
    for i in range(0, len(inputNodes)):
        # get the weight and change in weight attributes of each input node
        # these variables are used to make the code easier to read
        input_weights = inputNodes[i].weight
        all_input_weight_changes = inputNodes[i].changeInWeights
        for j in range(0, len(hidLay)):
            # calculate the new input weight using the formula provided
            new_input_weight = inputNodes[i].weight[j] + (stepSize * hidLay[j].delta * inputNodes[i].value) + (
                    0.9 * all_input_weight_changes[j])
            # find the change in weights and set it to position j in the "changeInWeights" attribute (which is set as a list)
            input_weight_change = input_weights[j] - new_input_weight
            all_input_weight_changes[j] = input_weight_change
            # update this specific weight of this input node to hidden layer with the new weight
            input_weights[j] = new_input_weight

        # set the new weights of the input node to the input weights
        inputNodes[i].weight = input_weights
        # set the change in weights of the input node to the all_input_weight_change list.
        inputNodes[i].changeInWeights = all_input_weight_changes

    # update the weights, delta, and bias of the hidden layer nodes
    for i in range(0, len(hidLay)):
        hidLay[i].delta = (hidLay[i].weight * outNode.delta) * (hidLay[i].activation * (1 - hidLay[i].activation))
        hidLay[i].bias = hidLay[i].bias + (stepSize * hidLay[i].delta)

        # calculate the new weight
        new_hid_weight = hidLay[i].weight + (stepSize * outNode.delta * hidLay[i].activation) + (
                0.9 * hidLay[i].changeInWeights)
        # find the chnage in weight after one pass
        hid_weight_change = hidLay[i].weight - new_hid_weight
        # update the change in weights and weight.
        hidLay[i].changeInWeights = hid_weight_change
        hidLay[i].weight = new_hid_weight




# called in the training data fucntion.
# allows me to update the model with a new row from the training data
def updateNodes(inputNodes, outNode, x_trainNumpy, y_trainNumpy, rowNumber):
    noColumns = np.shape(x_trainNumpy)[1]
    # each inputNode corresponds to a column, so I use this to set the value.
    for i in range(noColumns):
        # i represents a  column value
        inputNodes[i].value = x_trainNumpy[rowNumber][i]

    # each outnode value corresponds to the same row in x_train
    outNode.value = y_trainNumpy[rowNumber]


# function to generate inputNodes when testing
def testingNodes(x_testNumpy, y_testNumpy, outNode, hidLay, inputNodes):
    noColumns = np.shape(x_testNumpy)[1]
    testInputNodes = []

    # the weights of new input nodes when training
    # I loop through the columns first, (column number corresponds to respective inputNode)
    for i in range(noColumns):
        testing_weights_to_hidden = []
        for j in range(0, len(hidLay)):
            # using hidLay allows me to retrieve the weights generated from the previous row
            # these weights correspond to the initial weights of the new row's inputNode
            # i append these weights to an array
            testing_weights_to_hidden.append(inputNodes[i].weight[j])
        # instatiate an inputNode object
        testing_input = inputNode(testing_weights_to_hidden, 0, 0, x_testNumpy[0][i], [0] * len(hidLay))
        # append new inputNode object to an array
        testInputNodes.append(testing_input)
    # instatiates an outputNode based on bias of previous outnode, and value from the testing data.
    testOutNode = outputNode(0, outNode.bias, 0, 0, y_testNumpy[0])

    return testOutNode, testInputNodes


# function to train my data on a given number of epochs
def trainingData(inputNodes,hidLay,outNode,learningParameter,interval,x_trainNumpy, y_trainNumpy,epochs):

    # loops for a given interval number (i.e. 10)
    for j in range(0,interval):
        # reset data to top of data set after completed interval
        updateNodes(inputNodes,outputNode,x_trainNumpy, y_trainNumpy,0)
        #loop through each row in training data
        for i in range(0,len(x_trainNumpy)):
            # forwards and backwards pass on each row of training data
            forwardPass(hidLay, outNode, inputNodes)
            backwardsPass(hidLay, outNode, inputNodes,learningParameter)
            # 353 is the max number of rows in training data
            if i == 353:
                break
            else:
                # update to the next row of training data
                updateNodes(inputNodes, outNode, x_trainNumpy, y_trainNumpy, i+1)



# function does one forward pass of testing data and calulates error
def testingData(testInputNodes,hidLay,testOutNode,x_testNumpy,y_testNumpy):
    error = []

    for i in range(0, len(x_testNumpy)):
        # does one forward pass of the current testing output and input nodes
        # hidden layer stays the same
        forwardPass(hidLay, testOutNode, testInputNodes)
        # calculation for the error of each row of testing data
        error.append((testOutNode.value - testOutNode.activation) ** 2)

        # 117 is the no. rows in the x_testNumpy file
        if i == 117:
            break
        else:
            # creates a new row of data / objects of input nodes and out put nodes
            updateNodes(testInputNodes, testOutNode, x_testNumpy, y_testNumpy, i+1)
    # calculate the mse
    mse = sum(error) / len(error)
    return mse


def backPropAlgorithm(inputNodes,hidLay,outNode,learningParameter,epochs,x_trainNumpy, y_trainNumpy,x_testNumpy, y_testNumpy):
    # a list of mse generated from the testing set within the intervals
    mseList = []
    # interval = 10
    interval = int(epochs/(epochs/10))
    # number of mse generated, i.e. if epochs = 2000, I will generate 200 Mse values
    noMse = int(epochs/10)
    for i in range(0,noMse):

        # trains the data
        trainingData(inputNodes, hidLay, outNode, learningParameter, interval, x_trainNumpy, y_trainNumpy,epochs)

        # generates new input and output nodes for testing
        testOutNode, testInputNodes = testingNodes(x_testNumpy, y_testNumpy, outNode, hidLay, inputNodes, outputRow=0)

        # mse is returned from this function and appended to a list
        mse = testingData(testInputNodes,hidLay,testOutNode,x_testNumpy,y_testNumpy)
        mseList.append(mse)

    return mseList



def main():
    # TRAINING DATA
    read_xtrain_data = pd.read_csv("../../x_train.csv")
    x_trainData = read_xtrain_data.iloc[:, 1:]
    x_trainNumpy = pd.DataFrame.to_numpy(x_trainData)

    read_ytrain_data = pd.read_csv("../../y_train.csv")
    y_trainData = read_ytrain_data.iloc[:, 1:]
    y_trainNumpy = pd.DataFrame.to_numpy(y_trainData)

    # TESTING DATA
    read_xtest_data = pd.read_csv("../../X_test.csv")
    x_testData = read_xtest_data.iloc[:, 1:]
    x_testNumpy = pd.DataFrame.to_numpy(x_testData)

    read_ytest_data = pd.read_csv("../../y_test.csv")
    y_testData = read_ytest_data.iloc[:, 1:]
    y_testNumpy = pd.DataFrame.to_numpy(y_testData)


    noHiddenNodes = 4
    hidLay, outNode, inputNodes = generate_hidden_weight_bias(noHiddenNodes, x_trainNumpy, y_trainNumpy, outPutrow=0)
    stepSize = 0.1
    epochs = 2000

    mseList = backPropAlgorithm(inputNodes, hidLay, outNode, stepSize, epochs, x_trainNumpy, y_trainNumpy,
                                x_testNumpy, y_testNumpy)

    mseListNp = np.array(mseList)
    mse_df = pd.DataFrame(mseListNp)

    mse_df.to_excel("new_sig_momentum" + str(noHiddenNodes) + "_" + str(epochs) + "_" + ".xlsx")


main()
