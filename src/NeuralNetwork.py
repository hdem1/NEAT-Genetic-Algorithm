import numpy as np
import copy

class Node:
    def __init__(self, actFunction, bias, layerLevel, enabled, historyValue):
        self.actFunction = actFunction
        self.bias = bias
        self.layerLevel = layerLevel
        self.enabled = enabled
        self.historyValue = historyValue

class Edge:
    def __init__(self, origin, destination, weight, enabled, historyValue):
        self.origin = origin
        self.dest = destination
        self.weight = weight
        self.enabled = enabled
        self.historyValue = historyValue

class NeuralNetwork:

    #DEFINING CLASS VARIABLES:
    #Activation Functions:
    identityActFunctionID = 0
    leakyReluActFunctionID = 1
    sigmoidActFunctionID = 2

    #INITIALIZERS:
    def __init__(self):
        #Tracing nodes/edges:
        self.nodes = [] 
        self.edges = [] 
        #Monitoring connections:
        self.connectionsTo = [] # each element is a list of edges running to that node
        self.connectionsFrom = []
        #Specifying the neural network
        self.species = 0
        self.inputs = 0
        self.outputs = 0
        # For dynamic programming:
        self.values = []
        self.visited = []

    @classmethod
    def randomBaseNetwork(cls, inputs, outputs):
        newNN = cls()
        #Setting constants:
        newNN.inputs = inputs
        newNN.outputs = outputs

        #Making arrays:
        newNN.values = np.zeros(inputs+outputs)
        newNN.visited = np.zeros(inputs+outputs)
        newNN.nodes = [ [] for _ in range(inputs+outputs)]
        newNN.edges = [ [] for _ in range(inputs*outputs)]
        newNN.connectionsTo = [ [] for _ in range(inputs+outputs)]
        newNN.connectionsFrom = [ [] for _ in range(inputs+outputs)]

        #Setting input nodes:
        for i in range(inputs):
            newNode = np.zeros(newNN.nodeListLength)
            newNode[newNN.actFunctionIndex] = newNN.identityActFunctionID
            newNode[newNN.biasIndex] = 0
            newNode[newNN.scaleIndex] = 1
            newNode[newNN.layerLevelIndex] = 0
            newNode[newNN.nodeEnabledIndex] = 1
            newNode[newNN.nodeHistoryIndex] = i
            newNN[i] = newNode

        #Setting output nodes:
        for i in range(outputs):
            newNode = np.zeros(newNN.nodeListLength)
            newNode[newNN.actFunctionIndex] = newNN.sigmoidActFunctionID
            newNode[newNN.biasIndex] = 0
            newNode[newNN.scaleIndex] = 1
            newNode[newNN.layerLevelIndex] = 1
            newNode[newNN.nodeEnabledIndex] = 1
            newNode[newNN.nodeHistoryIndex] = inputs + i
            newNN[inputs + i] = newNode

        #Creating edges:
        for origin in range(inputs):
            for dest in range(outputs):
                newEdge = np.zeros(newNN.edgeListLength)
                newEdge[newNN.originIndex] = origin
                newEdge[newNN.destIndex] = dest
                newEdge[newNN.weightIndex] = (1- 2* np.random.random()) * 3
                newEdge[newNN.]

        return newNN

    @classmethod
    def childFromParents(cls, parent1, parent2):
        newNN = cls()

        return newNN

    @classmethod
    def networkFromString(cls, str):
        newNN = cls()

        return newNN

    #---------------------------------------------------
    #MODIFIERS:
    def insertNode(self, node1, node2, lastEdgeID, lastNodeID):


    def disableNode(self, node):

    def insertConnection(self, node1, node2):

    def disableConnection(self, node):

    #---------------------------------------------------
    #ACCESSORS:
    def getNetworkString(self):
        str = ""

        return str

    def getNodes(self):
        return copy.deepcopy(self.nodes)

    def getEdges(self):
        return copy.deepcopy(self.edges)

    #---------------------------------------------------
    #OTHER METHODS:
    def getOutput(self, inputs):
        # Dynamic programming:
        self.values = np.zeros(len(self.nodes))
        self.visited = np.zeros(len(self.nodes))

        #Loading in the inputs:
        for i in range(self.inputs):
            self.values[i] = inputs[i]
            self.visited[i] = 1

        #Making the output
        output = np.zeros(self.outputs)
        for i in range(self.outputs):
            output[i] = getNodeValue(self.inputs + i)
        
        return output

    def getNodeValue(self, nodeIndex):
        #Dynamic programming:
        if self.visited[nodeIndex] == 1:
            return self.values[nodeIndex]

        #Variable initialization
        value = 0

        #Looping through connections
        for edgeIndex in self.connectionsTo[nodeIndex]:
            edge = self.edges[edgeIndex]
            if edge[self.edgeEnabledIndex] == 1: #if enabled
                value += edge[self.weightIndex] * self.getNodeValue(edge[self.originIndex])

        #Activation function
        value = self.activationFunction(value, node)
        
        #Scaling
        value *= self.nodes[node][self.scaleIndex]

        #Saving value:
        self.visited[node] = 1
        self.values[node] = value
        return value

    def activationFunction(self, value, nodeIndex):
        func = self.nodes[nodeIndex][self.actFunctionIndex]
        if func == self.identityActFunctionID:
            output = value
        elif func == self.leakyReluActFunctionID:
            alpha = 0.05
            output = max(alpha * value, value)
        elif func == self.sigmoidActFunctionID:
            output = 1/(1+np.exp(-value))
        return output

    def getDistance(self, otherNetwork):

