from multiprocessing.dummy.connection import families
import random
from reprlib import aRepr
from xmlrpc.client import MAXINT, MININT
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
        self.activeConnections = 0
        self.activeNodes = 0
        self.testsRun = 0

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
        newNN.nodes = [Node(newNN.identityActFunctionID, 0, 0, True, i) for i in range(inputs)]
        
        #Setting output nodes:
        newNN.nodes = newNN.nodes + [Node(newNN.sigmoidActFunctionID,0,1,True,inputs+i) for i in range(outputs)]

        #Creating edges:
        edgeCount = 0
        for origin in range(inputs):
            for dest in range(inputs, inputs + outputs):
                newNN.edges.append(Edge(origin, dest, (1 - 2* np.random.random()) * 3, True, origin * outputs + (dest-inputs)))
                newNN.connectionsTo[dest].append(edgeCount)
                newNN.connectionsFrom[origin].append(edgeCount)
                edgeCount +=1
        newNN.activeConnections = len(newNN.edges)
        newNN.activeNodes = len(newNN.nodes)
        return newNN

    @classmethod
    def childFromParents(cls, parent1, parent2):
        #parent1.printErrors("Parent 1 Error 1")
        #parent2.printErrors("Parent 2 Error 1")
        newNN = cls()

        newNN.inputs = parent1.inputs
        newNN.outputs = parent1.outputs

        #Combining nodes
        index1 = 0
        index2 = 0
        #parent1.printErrors("Parent 1 Error 2")
        #parent2.printErrors("Parent 2 Error 2")
        while index1 < len(parent1.nodes) or index2 < len(parent2.nodes):
            if index1 < len(parent1.nodes):
                history1 = parent1.nodes[index1].historyValue
            else:
                history1 = MAXINT
            if index2 < len(parent2.nodes):
                history2 = parent2.nodes[index2].historyValue
            else:
                history2 = MAXINT

            if history1 < history2:
                newNN.nodes.append(Node(parent1.nodes[index1].getString()))
                index1 += 1
            elif history2 < history1:
                newNN.nodes.append(Node(parent2.nodes[index2].getString()))
                index2 += 1
            else:
                if parent1.fitness > parent2.fitness or (parent1.fitness == parent2.fitness and random.random() < 0.5):
                    newNN.nodes.append(Node(parent1.nodes[index1].getString()))
                else:
                    newNN.nodes.append(Node(parent2.nodes[index2].getString()))
                index1 += 1
                index2 += 1
            if newNN.nodes[-1].enabled:
                newNN.activeNodes += 1

        #parent1.printErrors("Parent 1 Error 3")
        #parent2.printErrors("Parent 2 Error 3")
        #Creating connection matrices:
        #print("Num Nodes: " + str(len(parent1.nodes)) + "+" + str(len(parent2.nodes)) + " --> " + str(len(newNN.nodes)))
        newNN.connectionsTo = [ [] for _ in range(len(newNN.nodes))]
        newNN.connectionsFrom = [ [] for _ in range(len(newNN.nodes))]
                
        #Combining edges:
        index1 = 0
        index2 = 0
        edgeCount = 0
        #parent1.printErrors("Parent 1 Error 4")
        #parent2.printErrors("Parent 2 Error 4")
        while index1 < len(parent1.edges) or index2 < len(parent2.edges):
            #parent1.printErrors("Parent 1 Error 4.1")
            #parent2.printErrors("Parent 2 Error 4.1")
            if index1 < len(parent1.edges):
                history1 = parent1.edges[index1].historyValue
            else:
                history1 = MAXINT
            if index2 < len(parent2.edges):
                history2 = parent2.edges[index2].historyValue
            else:
                history2 = MAXINT

            #parent1.printErrors("Parent 1 Error 4.2")
            #parent2.printErrors("Parent 2 Error 4.2")

            if history1 < history2:
                newEdge = parent1.edges[index1]
                if newEdge.dest >= len(parent1.nodes):
                    print(parent1.getNetworkString())
                    print(str(newEdge.origin) + "," + str(newEdge.dest) + "," + str(newEdge.historyValue) + " out of " + str(parent1.getNumNodes()))
                originHistory = parent1.nodes[newEdge.origin].historyValue
                destHistory = parent1.nodes[newEdge.dest].historyValue
                index1 += 1
            elif history2 < history1:
                newEdge = parent2.edges[index2]
                if newEdge.dest >= len(parent2.nodes):
                    print(parent2.getNetworkString())
                    print(str(newEdge.origin) + "," + str(newEdge.dest) + "," + str(newEdge.historyValue) + " out of " + str(parent2.getNumNodes()))
                originHistory = parent2.nodes[newEdge.origin].historyValue
                destHistory = parent2.nodes[newEdge.dest].historyValue
                index2 += 1
            else:
                if parent1.fitness > parent2.fitness or (parent1.fitness == parent2.fitness and random.random() < 0.5):
                    newEdge = parent1.edges[index1]
                    if newEdge.dest >= len(parent1.nodes):
                        print(parent1.getNetworkString())
                        print(str(newEdge.origin) + "," + str(newEdge.dest) + "," + str(newEdge.historyValue) + " out of " + str(parent1.getNumNodes()))
                    originHistory = parent1.nodes[newEdge.origin].historyValue
                    destHistory = parent1.nodes[newEdge.dest].historyValue
                else:
                    newEdge = parent2.edges[index2]
                    if newEdge.dest >= len(parent2.nodes):
                        print(parent2.getNetworkString())
                        print(str(newEdge.origin) + "," + str(newEdge.dest) + "," + str(newEdge.historyValue) + " out of " + str(parent2.getNumNodes()))
                    originHistory = parent2.nodes[newEdge.origin].historyValue
                    destHistory = parent2.nodes[newEdge.dest].historyValue
                index1 += 1
                index2 += 1
            #parent1.printErrors("Parent 1 Error 4.2")
            #parent2.printErrors("Parent 2 Error 4.2")

            newOriginIndex = 0
            while originHistory != newNN.nodes[newOriginIndex].historyValue:
                newOriginIndex += 1
                if newOriginIndex >= len(newNN.nodes):
                    print("Origin ERROR")
            newDestIndex = 0
            while destHistory != newNN.nodes[newDestIndex].historyValue:
                newDestIndex += 1
                if newDestIndex >= len(newNN.nodes):
                    print("DEST ERROR")
            #parent1.printErrors("Parent 1 Error 4.3")
            #parent2.printErrors("Parent 2 Error 4.3")

            #print(str(newEdge.origin) + "," + str(newEdge.dest) + " --> " + str(newOriginIndex) + "," + str(newDestIndex))
            if newNN.areConnected(newOriginIndex, newDestIndex) == 0:
                newEdge = Edge(newEdge.getString())
                newEdge.origin = newOriginIndex
                newEdge.dest = newDestIndex
                newNN.edges.append(newEdge)
                newNN.connectionsTo[newDestIndex].append(edgeCount)
                newNN.connectionsFrom[newOriginIndex].append(edgeCount)
                if newEdge.enabled:
                    newNN.activeConnections += 1
                edgeCount += 1
            #parent1.printErrors("Parent 1 Error 4.4")
            #parent2.printErrors("Parent 2 Error 4.4")

        #parent1.printErrors("Parent 1 Error 5")
        #parent2.printErrors("Parent 2 Error 5")
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
        newNN.testsRun = int(firstLine[6])
        #Define Nodes:
        for n in range(numNodes):
            newNN.nodes.append(Node(strings[n+1]))
            if newNN.nodes[-1].enabled:
                newNN.activeNodes += 1
        #Define connection arrays:
        newNN.connectionsTo = [ [] for _ in range(numNodes)]
        newNN.connectionsFrom = [ [] for _ in range(numNodes)]

        #Define edges:
        for e in range(numEdges):
            newNN.edges.append(Edge(strings[numNodes+1+e]))
            newNN.connectionsFrom[newNN.edges[-1].origin].append(e)
            newNN.connectionsTo[newNN.edges[-1].dest].append(e)
            if newNN.edges[-1].enabled:
                newNN.activeConnections += 1


        return newNN

    #---------------------------------------------------
    #MODIFIERS:
    def changeFitness(self,fitness,tests):
        self.fitness = (self.fitness * self.testsRun + fitness * tests) / (self.testsRun + tests)
        self.testsRun += tests

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
        #print("MAKING NEW EDGE " + str(firstEdgeID) + " = " + str(newEdge1.origin) + " --> " + str(newEdge1.dest) + ", old origin index = " + str(self.edges[edgeIndex].origin))
        #print("MAKING NEW NODE " + str(newNodeID) + " - numNodes = " + str(len(self.nodes)) + ", middle node index = " + str(newEdge1.dest))
        #print("MAKING NEW EDGE " + str(firstEdgeID+1) + " = " + str(newEdge2.origin) + " --> " + str(newEdge2.dest) + ", old destination = " + str(self.edges[edgeIndex].dest))
        self.edges.append(newEdge1)
        self.edges.append(newEdge2)

        if self.randomOnCreation:
            if self.distribution == "rand":
                self.nodes[-1].bias = (1-2*np.random.random()) * self.biasMagnitude
                self.edges[-2].weight = (1-2*np.random.random()) * self.weightMagnitude
            if self.distribution == "norm":
                self.nodes[-1].bias = (1-2*np.random.normal()) * self.biasMagnitude
                self.edges[-2].weight = (1-2*np.random.normal()) * self.weightMagnitude
        self.activeNodes += 1

    def disableNode(self, nodeIndex):
        self.nodes[nodeIndex].enabled = False
        for outgoing in self.connectionsFrom[nodeIndex]:
            self.edges[outgoing].enabled = False
            self.activeConnections -= 1
        self.activeNodes -= 1
        
    def enableNode(self, nodeIndex):
        self.nodes[nodeIndex].enabled = True
        for outgoing in self.connectionsFrom[nodeIndex]:
            self.edges[outgoing].enabled = True
            self.activeConnections += 1
            #ISSUE = WILL REACTIVATE EDGES THAT WERE PREVIOUSLY DEACTIVATED
        self.activeNodes += 1

    def insertConnection(self, nodeIndex1, nodeIndex2, newEdgeID): #Returns -1 if doesn't work, 0 if enabled previous connection, 1 if inserted new connection
        self.printErrors("ADDING CONNECTION START")
        if nodeIndex1 == nodeIndex2:
            return -1
        if nodeIndex1 < 0 or nodeIndex1 >= len(self.nodes) or nodeIndex2 < 0 or nodeIndex2 >= len(self.nodes):
            print("NODE OUT OF BOUNDS")
            return -1
        if nodeIndex1 < self.inputs and nodeIndex2 < self.inputs:
            return -1
        if nodeIndex1 >= self.inputs and nodeIndex1 < self.inputs + self.outputs and nodeIndex2 >= self.inputs and nodeIndex2 < self.inputs + self.outputs:
            return -1
        c = self.areConnected(nodeIndex1, nodeIndex2)
        if c == 1:
            #print("ALREADY CONNECTED")
            return -1
        elif c == -1:
            self.enableConnection(self.getEdgeFromNodes(nodeIndex1, nodeIndex2))
            return 0

        if self.nodes[nodeIndex2].layerLevel > self.nodes[nodeIndex1].layerLevel:
            firstIndex = nodeIndex1
            secondIndex = nodeIndex2
        elif self.nodes[nodeIndex2].layerLevel < self.nodes[nodeIndex1].layerLevel:
            firstIndex = nodeIndex2
            secondIndex = nodeIndex1
        else:
            firstIndex = nodeIndex1
            secondIndex = nodeIndex2 
            #Fixing layer levels:
            maxIncoming = 0
            for edgeIndex in self.connectionsTo[nodeIndex1]:
                if self.nodes[self.edges[edgeIndex].origin].layerLevel > maxIncoming:
                    maxIncoming = self.nodes[self.edges[edgeIndex].origin].layerLevel
            minOutgoing = 1
            for edgeIndex in self.connectionsFrom[nodeIndex2]:
                if self.nodes[self.edges[edgeIndex].dest].layerLevel < minOutgoing:
                    minOutgoing = self.nodes[self.edges[edgeIndex].dest].layerLevel
            self.nodes[nodeIndex1].layerLevel = maxIncoming + (minOutgoing-maxIncoming) / 3
            self.nodes[nodeIndex2].layerLevel = maxIncoming + 2 * (minOutgoing-maxIncoming) / 3
        newEdge1 = Edge(firstIndex, secondIndex, 1,True, newEdgeID)
        if self.randomOnCreation:
            if self.distribution == "rand":
                newEdge1.weight = (1-2*np.random.random()) * self.weightMagnitude
            if self.distribution == "norm":
                newEdge1.weight = (1-2*np.random.normal()) * self.weightMagnitude
        self.edges.append(newEdge1)
        self.printErrors("ADDING CONNECTION END")
        return 1

    def disableConnection(self, edgeIndex):
        self.edges[edgeIndex].enabled = False
    
    def enableConnection(self, edgeIndex):
        self.edges[edgeIndex].enabled = True
    
    def mutateValues(self, mutationOdds, biasMagnitude = 2, weightMagnitude = 3, scaleMagnitude = -1):
        for i in range(len(self.nodes)):
            if random.random() < mutationOdds and biasMagnitude != -1:
                self.nodes[i].bias = (1-2*random.random()) * biasMagnitude
            if random.random() < mutationOdds and scaleMagnitude != -1:
                self.nodes[i].scale = (1-2*random.random()) * scaleMagnitude
        for i in range(len(self.edges)):
            if random.random() < mutationOdds and weightMagnitude != -1:
                self.edges[i].weight = (1-2*random.random()) * weightMagnitude

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
        output = output + str(self.fitness) + ","
        output = output + str(self.testsRun) + "\n"

        #Nodes:
        for node in self.nodes:
            output = output + node.getString()

        #Edges:
        for edge in self.edges:
            output = output + edge.getString()
        return output
    
    def checkFullness(self):
        self.full = True
        for i in range(len(self.nodes)):
            activeConnections = 0
            for e in self.connectionsFrom[i]:
                if e.enabled:
                    activeConnections += 1  
            for e in self.connectionsTo[i]:
                if e.enabled:
                    activeConnections += 1
            if activeConnections != len(self.nodes) - 1:
                self.full = False
                break
        
        return self.full
    
    def getEdgeFromNodes(self, nodeIndex1, nodeIndex2):
        if self.nodes[nodeIndex1].layerLevel < self.nodes[nodeIndex2].layerLevel:
            firstIndex = nodeIndex1
            secondIndex = nodeIndex2
        else:
            firstIndex = nodeIndex2
            secondIndex = nodeIndex1
        for edge1 in self.connectionsFrom[firstIndex]:
            for edge2 in self.connectionsTo[secondIndex]:
                if edge1 == edge2:
                    return edge1
        return -1

    def areConnected(self, nodeIndex1, nodeIndex2): #1 = active conection, 0 = no connection, -1 = disabled connection
        e = self.getEdgeFromNodes(nodeIndex1, nodeIndex2)
        if e == -1:
            return 0
        elif self.edges[e].enabled == False:
            return -1
        return 1

    def getNodes(self):
        return copy.deepcopy(self.nodes)
    
    def getNode(self, index):
        return self.nodes[index]

    def getEdges(self):
        return copy.deepcopy(self.edges)
    
    def getEdge(self, index):
        return self.edges[index]
    
    def getNumEdges(self):
        return len(self.edges)

    def getNumNodes(self):
        return len(self.nodes)
    
    def getTestsRun(self):
        return self.testsRun
    
    def edgeBoundaryError(self):
        for edge in self.edges:
            if edge.origin < 0 or edge.origin >= self.getNumNodes() or edge.dest < 0 or edge.dest >= self.getNumNodes():
                return True
        return False
    
    def selfReferenceError(self):
        for edge in self.edges:
            if edge.origin == edge.dest:
                return True
        return False
    
    def directCycleError(self):
        for i in range(len(self.nodes)):
            for edgeIndex in self.connectionsTo[i]:
                if self.edges[edgeIndex].origin == i:
                    return True
        return False
    
    def edgeDirectionError(self):
        for edge in self.edges:
            if self.nodes[edge.origin].layerLevel > self.nodes[edge.dest].layerLevel:
                return True
        return False
    
    def arrayFailureError(self):
        for i in range(len(self.nodes)):
            for edgeIndex in self.connectionsTo[i]:
                if self.edges[edgeIndex].dest != i:
                    return True
            for edgeIndex in self.connectionsFrom[i]:
                if self.edges[edgeIndex].origin != i:
                    return True
        return False
    
    def printErrors(self, label):
        error = False
        if self.edgeBoundaryError():
            if error == False:
                print(label)
                error = True
            print("EDGE ORIGIN/DEST INDEX ERROR")
        if self.selfReferenceError():
            if error == False:
                print(label)
                error = True
            print("EDGE SELF REFERENCE ERROR")
        if self.directCycleError():
            if error == False:
                print(label)
                error = True
            print("NODE DIRECT SELF REFERENCE ERROR")
        if self.arrayFailureError():
            if error == False:
                print(label)
                error = True
            print("ARRAY FALIURE ERROR")
        if self.edgeDirectionError():
            if error == False:
                print(label)
                error = True
            print("EDGE DIRECTION ERROR")
        if error:
            print(self.getNetworkString())
            exit()


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
    
    def resetFitness(self):
        self.fitness = MININT

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
                avgBiasDif += abs(self.nodes[index1].bias - otherNetwork.nodes[index2].bias)
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
                avgWeightDif += abs(self.edges[index1].weight - otherNetwork.edges[index2].weight)
                index1 += 1
                index2 += 1
        excess += max(len(self.edges) - index1, len(otherNetwork.edges) - index2)
        avgWeightDif /= edgeCount

        #normalize for genome size:
        disjoint /= (edgeCount + nodeCount)
        excess /= (edgeCount + nodeCount)

        c1 = 0.75
        c2 = 0.75
        c3 = 0.1
        c4 = 0.1
        output = c1*disjoint + c2*excess + c3*avgWeightDif + c4*avgBiasDif
        return output
    
    #Sorting:
    def __lt__(self, obj):
        return ((self.fitness) < (obj.fitness))
  
    def __gt__(self, obj):
        return ((self.fitness) > (obj.fitness))
  
    def __le__(self, obj):
        return ((self.fitness) <= (obj.fitness))
  
    def __ge__(self, obj):
        return ((self.fitness) >= (obj.fitness))
  
    def __eq__(self, obj):
        return (self.fitness == obj.fitness)
