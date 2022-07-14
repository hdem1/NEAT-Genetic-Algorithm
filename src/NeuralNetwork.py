from xmlrpc.client import MININT
import numpy as np
import copy

class Node:
    def __init__(self, *args):
        if len(args) == 0:
            self.actFunction = 0
            self.bias = 0
            self.layerLevel = 0
            self.enabled = False
            self.historyValue = 0
            self.scale = 0
        if len(args) == 1:
            strings = args[0].split(",")
            self.actFunction = int(strings[0])
            self.bias = float(strings[1])
            self.layerLevel = float(strings[2])
            self.enabled = bool(strings[3])
            self.historyValue = int(strings[4])
            self.scale = float(strings[5])
        if len(args) > 1:
            self.actFunction = args[0]
            self.bias = args[1]
            self.layerLevel = args[2]
            self.enabled = args[3]
            self.historyValue = args[4]
            self.scale = 1
        if len(args) == 6:
            self.scale = args[5]
    
    def getString(self):
        output = ""
        output = output + str(self.actFunction) + ","
        output = output + str(self.bias) + ","
        output = output + str(self.layerLevel) + ","
        output = output + str(self.enabled) + ","
        output = output + str(self.historyValue) + ","
        output = output + str(self.scale) + "\n"
        return output

class Edge:
    def __init__(self, *args):
        if len(args) == 1:
            strings = args[0].split(",")
            self.origin = int(strings[0])
            self.dest = int(strings[1])
            self.weight = float(strings[2])
            self.enabled = bool(strings[3])
            self.historyValue = int(strings[4])
        if len(args) == 5:
            self.origin = int(args[0])
            self.dest = int(args[1])
            self.weight = float(args[2])
            self.enabled = bool(args[3])
            self.historyValue = int(args[4])
    
    def getString(self):
        output = ""
        output = output + str(self.origin) + ","
        output = output + str(self.dest) + ","
        output = output + str(self.weight) + ","
        output = output + str(self.enabled) + ","
        output = output + str(self.historyValue) +"\n"
        return output

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
        edgeCount = 0
        for origin in range(inputs):
            for dest in range(inputs, inputs + outputs):
                newNN.edges.append(Edge(origin, dest, (1- 2* np.random.random()) * 3, 1, origin * outputs + dest))
                newNN.connectionsTo[dest].append(edgeCount)
                newNN.connectionsFrom[origin].append(edgeCount)
                edgeCount +=1

        return newNN

    @classmethod
    def childFromParents(cls, parent1, parent2):
        newNN = cls()

        #Combining nodes
        index1 = 0
        index2 = 0
        while index1 < len(parent1.nodes) and index2 < len(parent2.nodes):
            history1 = parent1.nodes[index1].historyValue
            history2 = parent2.nodes[index2].historyValue
            if history1 < history2:
                newNN.nodes.append(Node(parent1.nodes[index1].getString()))
                index1 += 1
            elif history2 < history1:
                newNN.nodes.append(Node(parent2.nodes[index2].getString()))
                index2 += 1
            else:
                threshold = 0.5 + (parent2.fitness - parent1.fitness) / 1000
                if np.random.random() < threshold:
                    newNN.nodes.append(Node(parent1.nodes[index1].getString()))
                else:
                    newNN.nodes.append(Node(parent2.nodes[index2].getString()))
                index1 += 1
                index2 += 1
        
        #Creating connection matrices:
        newNN.connectionsTo = [ [] for _ in range(len(newNN.nodes))]
        newNN.connectionsFrom = [ [] for _ in range(len(newNN.nodes))]
                
        #Combining edges:
        index1 = 0
        index2 = 0
        edgeCount = 0
        while index1 < len(parent1.edges) and index2 < len(parent2.edges):
            history1 = parent1.edges[index1].historyValue
            history2 = parent2.edges[index2].historyValue
            if history1 < history2:
                newNN.edges.append(Edge(parent1.edges[index1].getString()))
                index1 += 1
            elif history2 < history1:
                newNN.edges.append(Edge(parent2.edges[index2].getString()))
                index2 += 1
            else:
                threshold = 0.5 + (parent2.fitness - parent1.fitness) / 1000
                if np.random.random() < threshold:
                    newNN.edges.append(Edge(parent1.edges[index1].getString()))
                else:
                    newNN.edges.append(Edge(parent2.edges[index2].getString()))
                index1 += 1
                index2 += 1
            newNN.connectionsTo[newNN.edges[-1].dest].append(edgeCount)
            newNN.connectionsFrom[newNN.edges[-1].origin].append(edgeCount)
            edgeCount += 1

        return newNN

    @classmethod
    def networkFromString(cls, strings):
        newNN = cls()
        firstLine = strings[0].split(",")
        numNodes = int(firstLine[0])
        numEdges = int(firstLine[1])
        newNN.inputs = int(firstLine[2])
        newNN.outputs = int(firstLine[3])
        newNN.species = int(firstLine[4])
        #Define Nodes:
        for n in range(numNodes):
            newNN.nodes.append(Node(strings[n+1]))
        #Define connection arrays:
        newNN.connectionsTo = [ [] for _ in range(numNodes)]
        newNN.connectionsFrom = [ [] for _ in range(numNodes)]

        #Define edges:
        for e in range(numEdges):
            newNN.edges.append(Edge(strings[numNodes+1+e]))
            newNN.connectionsFrom[newNN.edges[-1].origin].append(e)
            newNN.connectionsTo[newNN.edges[-1].dest].append(e)

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
        output = ""
        #Line 1 - numNodes,numEdges
        output = output + str(len(self.nodes)) + "," 
        output = output + str(len(self.edges)) + ","
        output = output + str(self.inputs) + ","
        output = output + str(self.outputs) + ","
        output = output + str(self.species) + ","
        output = output + str(self.fitness) + "\n"

        #Nodes:
        for node in self.nodes:
            output = output + node.getString()

        #Edges:
        for edge in self.edges:
            output = output + edge.getString()
        return output

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
            output[i] = self.getNodeValue(self.inputs + i)
        
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
        value = self.activationFunction(value, nodeIndex)
        
        #Scaling
        value *= self.nodes[nodeIndex].scale

        #Saving value:
        self.visited[nodeIndex] = 1
        self.values[nodeIndex] = value
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
        #Getting the larger number of genes
        numGenes1 = len(self.edges) + len(self.nodes)
        numGenes2 = len(otherNetwork.edges) + len(otherNetwork.nodes)
        N = max(numGenes1,numGenes2)

        #Setting up variables:
        disjoint = 0
        excess = 0
        avgWeightDif = 0
        avgBiasDif = 0
        edgeCount = 0
        nodeCount = 0

        #Compare node genes:
        index1 = 0
        index2 = 0
        while index1 < len(self.nodes) and index2  < len(otherNetwork.nodes):
            nodeCount += 1
            history1 = self.nodes[index1].historyValue
            history2 = otherNetwork.nodes[index2].historyValue
            if history1 < history2:
                disjoint += 1
                index1 += 1
            elif history2 < history1:
                disjoint += 1
                index2 += 1
            else:
                avgBiasDif += abs(self.nodes[index2].bias - otherNetwork.nodes[index2].bias)
                index1 += 1
                index2 += 1
        excess += max(len(self.nodes) - index1, len(otherNetwork.nodes) - index2)
        avgBiasDif /= nodeCount

        #Compare edge genes:
        index1 = 0
        index2 = 0
        while index1 < len(self.edges) and index2  < len(otherNetwork.edges):
            edgeCount += 1
            history1 = self.edges[index1].historyValue
            history2 = otherNetwork.edges[index2].historyValue
            if history1 < history2:
                disjoint += 1
                index1 += 1
            elif history2 < history1:
                disjoint += 1
                index2 += 1
            else:
                avgWeightDif += abs(self.edges[index2].weight - otherNetwork.edges[index2].weight)
                index1 += 1
                index2 += 1
        excess += max(len(self.edges) - index1, len(otherNetwork.edges) - index2)
        avgWeightDif /= edgeCount

        #normalize for genome size:
        disjoint /= (edgeCount + nodeCount)
        excess /= (edgeCount + nodeCount)

        c1 = 1
        c2 = 1
        c3 = 1
        c4 = 1
        output = c1*disjoint + c2*excess + c3*avgWeightDif + c4*avgBiasDif
        return output
