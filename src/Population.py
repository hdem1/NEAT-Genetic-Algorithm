from math import ceil, floor
from xmlrpc.client import MININT
import numpy as np
from NeuralNetwork import NeuralNetwork
import random
from os.path import exists, expanduser, isdir
from os import mkdir

class Population:
    #CONSTRUCTION:
    def __init__(self, envName, inputs, outputs, size, survivalRate = 0.1):
        self.NNs = []
        self.envName = envName
        self.size = size
        self.survivalRate = survivalRate
        self.addNodeRate = 0.1
        self.disableNodeRate = 0.01
        self.addConnectionRate = 0.5
        self.disableConnectionRate = 0.05
        self.inputs = inputs
        self.outputs = outputs
        self.latestNodeID = inputs + outputs - 1
        self.latestConnectionID = inputs * outputs - 1
    
    def makeInitialPopulation(self):
        for i in range(self.size):
            self.NNs.append(NeuralNetwork.randomBaseNetwork(self.envName, self.inputs, self.outputs))
    
    #EVOLUTION:
    def evolveGeneration(self):
        #print("Error Check 1 = ", self.edgeErrors(self.NNs))
        #self.printPopulationErrors(self.NNs, "Error Check 1")
        speciesLists = self.makeSpeciesLists(self.NNs)
        avgFitness = self.getPopulationAverageFitness()
        newPop = []
        speciesNum = -1
        for species in speciesLists:
            speciesNum += 1
            #self.printPopulationErrors(species, "Error Check 2")
            #Get the new number of NNs in that species:
            popTotalFitness = 0
            for NN in species:
                popTotalFitness += NN.getFitness()
                NN.species = speciesNum
            newNum = round(popTotalFitness/avgFitness)
            
            #Sort species into descending order:
            species = sorted(species, reverse=True)

            #Get the range of NNs from which to mutate:
            maxIndex = int(ceil(len(species)*self.survivalRate))

            #Get the New generation from the species:
            #self.printPopulationErrors(species, "Error Check 2.5")
            for i in range(newNum):
                parent1 = species[floor(random.random() * maxIndex)]
                parent2 = species[floor(random.random() * maxIndex)]
                
                #print(parent1.getNetworkString())
                #print(parent2.getNetworkString())
                #Make NN:
                #print("Error Check 3 = ", self.edgeErrors([parent1, parent2]))
                #self.printPopulationErrors([parent1, parent2], "Error Check 3")
                newNN = NeuralNetwork.childFromParents(self.envName, parent1, parent2)
                #self.printPopulationErrors([parent1, parent2], "Error Check 4")
                #Mutate NN:
                #To-DO:
                #   Could add functionality to catch double additions and make them the same history values
                #   Could add functionality to turn on disabled nodes/edges
                #Adding nodes:
                while random.random() < self.addNodeRate:
                    edgeIndex = floor(random.random() * newNN.getNumEdges())
                    newNN.insertNode(edgeIndex, self.latestConnectionID+1, self.latestNodeID+1)
                    #print(self.latestConnectionID, ", ", self.latestNodeID)
                    self.latestConnectionID += 2
                    self.latestNodeID += 1
                #Disabling nodes:
                while random.random() < self.disableNodeRate:
                    nodeIndex = floor(random.random() * newNN.getNumNodes())
                    newNN.disableNode(nodeIndex)
                #Adding Connections:
                while random.random() < self.addConnectionRate:
                    node1 = floor(random.random() * newNN.getNumNodes())
                    node2 = floor(random.random() * newNN.getNumNodes())
                    searchedOptions = 0
                    val = newNN.insertConnection(node1,node2, self.latestConnectionID+1)
                    while (val == -1 and searchedOptions < 100):
                        node1 = floor(random.random() * newNN.getNumNodes())
                        node2 = floor(random.random() * newNN.getNumNodes())
                        searchedOptions += 1
                        val = newNN.insertConnection(node1,node2, self.latestConnectionID+1)
                    if val == 1:
                        self.latestConnectionID += 1
                #Disabling Connections:
                while random.random() < self.disableConnectionRate:
                    edgeIndex = floor(random.random() * newNN.getNumEdges())
                    newNN.disableConnection(edgeIndex)
                #Value Mutations:
                newNN.mutateValues(0.1)
                newPop.append(newNN)
                #self.printPopulationErrors([parent1, parent2], "Error Check 5")
        self.NNs = newPop
        self.size = len(self.NNs)
        #self.printPopulationErrors(self.NNs, "Error Check 6")

    def edgeErrors(self, NNs):
        for NN in self.NNs:
            if NN.edgeBoundaryError():
                return True
        return False
    
    def printPopulationErrors(self, NNs, label):
        for NN in NNs:
            NN.printErrors(label)

    def makeSpeciesLists(self, NNs, difLimit = 0.5):
        speciesLists = []
        for NN in NNs:
            foundSpecies = False
            for i in range(len(speciesLists)):
                randomNN = random.choice(speciesLists[i])
                if NN.getDistance(randomNN) < difLimit:
                    speciesLists[i].append(NN)
                    foundSpecies = True
                    break
            if not foundSpecies:
                speciesLists.append([NN])
        return speciesLists

    #ACCESSORS:
    def getBestModel(self):
        bestNN = self.NNs[0]
        index = 0
        for i in range(len(self.NNs)):
            if self.NNs[i].getFitness() > bestNN.getFitness():
                bestNN = self.NNs[i]
                index = i
        return bestNN, index
    
    def getSize(self):
        return len(self.NNs)
    
    def getNeuralNetwork(self, index):
        return self.NNs[index]

    def getPopulationAverageFitness(self):
        total = 0
        for NN in self.NNs:
            total += NN.getFitness()
        return total/len(self.NNs)
    
    def getBestModels(self, portion):
        num = round(portion * len(self.NNs))
        indices = []
        output =[]
        for i in range(num):
            bestIndex = 0
            bestFitness = self.NNs[0].getFitness()
            for j in range(self.size):
                if j not in indices and self.NNs[j].getFitness() > bestFitness:
                    bestIndex = j
                    bestFitness = self.NNs[j].getFitness()
            output.append(self.NNs[bestIndex])
            indices.append(bestIndex)
        return output, indices
    
    #MODIFIERS:
    '''def setFitness(self,index,fitness, tests):
        self.NNs[index].changeFitness(fitness,tests)'''

    #FILE READING/WRITING:
    def getString(self):
        output = self.envName + ","
        output = output + str(len(self.NNs)) + ","
        output = output + str(self.survivalRate) + ","
        output = output + str(self.addNodeRate) + ","
        output = output + str(self.disableNodeRate) + ","
        output = output + str(self.addConnectionRate) + ","
        output = output + str(self.disableConnectionRate) + ","
        output = output + str(self.inputs) + ","
        output = output + str(self.outputs) + ","
        output = output + str(self.latestNodeID) + ","
        output = output + str(self.latestConnectionID) + "\n"
        for NN in self.NNs:
            output = output + NN.getNetworkString()
        return output
    
    @classmethod
    def loadPopulation(cls, lines):
        firstLine = lines[0].split(",") 
        newPop = cls(firstLine[0], int(firstLine[7]), int(firstLine[8]), int(firstLine[1]), float(firstLine[2]))
        newPop.addNodeRate = float(firstLine[3])
        newPop.disableNodeRate = float(firstLine[4])
        newPop.addConnectionRate = float(firstLine[5])
        newPop.disableConnectionRate = float(firstLine[6])
        newPop.latestNodeID = int(firstLine[9])
        newPop.latestNodeID = int(firstLine[10])
        currLine = 1
        for i in range(newPop.size):
            infoLine = lines[currLine].split(",")
            numLines = 1 + int(infoLine[0]) + int(infoLine[1])
            newPop.NNs.append(NeuralNetwork.networkFromString(newPop.envName, lines[currLine:(currLine+numLines)]))
            currLine += numLines
        return newPop
    
    def getBestModelString(self):
        bestModel, _ = self.getBestModel()
        return bestModel.getNetworkString()

