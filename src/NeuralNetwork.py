from xmlrpc.client import MININT
import numpy as np
import copy

class Node:
    def __init__(self, actFunction, bias, layerLevel, enabled, historyValue, scale = 1):
        self.actFunction = actFunction
        self.bias = bias
        self.layerLevel = layerLevel
        self.enabled = enabled
        self.historyValue = historyValue
        self.scale = scale

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
    #Mutation magnitudes:
    weightMagnitude = 3
    biasMagnitude = 3
    distribution = "rand" #rand or norm
    randomOnCreation = True

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
        # For performance monitoring:
        self.fitness = MININT

    @classmethod
    def randomBaseNetwork(cls, inputs, outputs):
        newNN = cls()
        #Setting constants:
        newNN.inputs = inputs
        newNN.outputs = outputs

        #Making arrays:
        newNN.values = np.zeros(inputs+outputs)
        newNN.visited = np.zeros(inputs+outputs)
        newNN.connectionsTo = [ [] for _ in range(inputs+outputs)]
        newNN.connectionsFrom = [ [] for _ in range(inputs+outputs)]

        #Setting input nodes:
        newNN.nodes = [Node(newNN.identityActFunctionID, 0, 0,1, i) for i in range(inputs)]
        
        #Setting output nodes:
        newNN.nodes = newNN.nodes + [Node(newNN.sigmoidActFunctionID,0,1,1,inputs+i) for i in range(outputs)]

        #Creating edges:
        for origin in range(inputs):
            for dest in range(inputs, inputs + outputs):
                newNN.edges.append(Edge(origin, dest, (1- 2* np.random.random()) * 3, 1, origin * outputs + dest))
                newNN.connectionsTo[dest].append(origin)
                newNN.connectionsFrom[origin].append(dest)

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
    def insertNode(self, edgeIndex, firstEdgeID, newNodeID):
        edge = self.edges[edgeIndex]
        originNode = self.nodes[edge.origin]
        destNode = self.nodes[edge.dest]

        #disable the current edge:
        self.edges[edgeIndex].enabled = False

        #Insert a new node:
        newNode = Node(self.leakyReluActFunctionID, 0, (originNode.layerLevel + destNode.layerLevel) / 2, True, newNodeID)
        self.nodes.append(newNode)

        #Insert new connections:
        self.connectionsFrom.append([len(self.edges)+1])
        self.connectionsTo.append([len(self.edges)])

        #Insert new edges:
        newEdge1 = Edge(edge.origin,len(self.nodes)-1, edge.weight,True, firstEdgeID)
        newEdge2 = Edge(len(self.nodes)-1, edge.dest, 1, True, firstEdgeID+1)
        self.edges.append(newEdge1)
        self.edges.append(newEdge2)

        if self.randomOnCreation:
            if self.distribution == "rand":
                self.nodes[-1].bias = (1-2*np.random.random()) * self.biasMagnitude
                self.edges[-2].weight = (1-2*np.random.random()) * self.weightMagnitude
            if self.distribution == "norm":
                self.nodes[-1].bias = (1-2*np.random.normal()) * self.biasMagnitude
                self.edges[-2].weight = (1-2*np.random.normal()) * self.weightMagnitude

        # if self.nodes[node1].layerLevel > self.nodes[node2].layerLevel:
        #     firstNode = node1
        #     secondNode = node2
        # elif self.nodes[node1].layerLevel < self.nodes[node2].layerLevel:
        #     firstNode = node2
        #     secondNode = node1

    def disableNode(self, nodeIndex):
        self.nodes[nodeIndex].enabled = False
        for outgoing in self.connectionsFrom[nodeIndex]:
            self.edges[outgoing].enabled = False

    def insertConnection(self, node1, node2, newEdgeID):
        newEdge1 = Edge(node1, node2, 1,True, newEdgeID)
        if self.randomOnCreation:
            if self.distribution == "rand":
                newEdge1.weight = (1-2*np.random.random()) * self.weightMagnitude
            if self.distribution == "norm":
                newEdge1.weight = (1-2*np.random.normal()) * self.weightMagnitude
        self.edges.append(newEdge1)

    def disableConnection(self, edgeIndex):
        self.edges[edgeIndex].enabled = False

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
            if edge.enabled == 1: #if enabled
                value += edge.weight * self.getNodeValue(edge.origin)

        #Activation function
        value = self.activationFunction(value, node)
        
        #Scaling
        value *= self.nodes[node].scale

        #Saving value:
        self.visited[node] = 1
        self.values[node] = value
        return value

    def activationFunction(self, value, nodeIndex):
        func = self.nodes[nodeIndex].actFunction
        if func == self.identityActFunctionID:
            output = value
        elif func == self.leakyReluActFunctionID:
            alpha = 0.05
            output = max(alpha * value, value)
        elif func == self.sigmoidActFunctionID:
            output = 1/(1+np.exp(-value))
        return output

    def getDistance(self, otherNetwork):

