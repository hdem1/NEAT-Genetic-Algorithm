from math import ceil
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
        self.disableNodeRate = 0.02
        self.addConnectionRate = 0.5
        self.disableConnectionRate = 0.05
        self.inputs = inputs
        self.outputs = outputs
        self.latestNodeID = inputs + outputs - 1
        self.latestConnectionID = inputs * outputs - 1
    
    def makeInitialPopulation(self):
        for i in range(self.size):
            self.NNs.append(NeuralNetwork.randomBaseNetwork(self.inputs, self.outputs))
    
    #EVOLUTION:
    def evolveGeneration(self):
        speciesLists = self.makeSpeciesLists(self.NNs, 1)
        avgFitness = self.getPopulationAverageFitness()
        self.NNs = []
        for species in speciesLists:
            #Get the new number of NNs in that species:
            speciesSize = len(species)
            popTotalFitness = 0
            for NN in species:
                popTotalFitness += NN.fitness
            newNum = round(popTotalFitness/avgFitness)
            
            #Sort species into descending order:
            species = sorted(species, reverse=True)

            #Get the range of NNs from which to mutate:
            maxIndex = int(ceil(len(species)*self.survivalRate))

            #Get the New generation from the species:
            for i in range(newNum):
                parent1 = species[round(random.random() * maxIndex)]
                parent2 = species[round(random.random() * maxIndex)]
                #Make NN:
                newNN = NeuralNetwork.childFromParents(species[parent1], species[parent2])
                #Mutate NN:
                #To-DO:
                #   Could add functionality to catch double additions and make them the same history values
                #   Could add functionality to turn on disabled nodes/edges
                #Adding nodes:
                while random.random() < self.addNodeRate:
                    edgeIndex = round(random.random() * newNN.getNumEdges())
                    newNN.insertNode(edgeIndex, self.latestConnectionID+1, self.latestNodeID+1)
                    self.latestConnectionID += 2
                    self.latestNodeID += 1
                #Disabling nodes:
                while random.random() < self.disableNodeRate:
                    nodeIndex = round(random.random() * newNN.getNumNodes())
                    newNN.disableNode(edgeIndex, nodeIndex)
                #Adding Connections:
                while random.random() < self.addConnectionRate:
                    node1 = round(random.random() * newNN.getNumNodes())
                    node2 = round(random.random() * newNN.getNumNodes())
                    while (node1 != node2 and newNN.areConnected(node1,node2) != 1):
                        node2 = round(random.random() * newNN.getNumNodes())
                    if newNN.areConnected(node1,node2) == 0:
                        newNN.insertConnection(node1, node2, self.latestConnectionID+1)
                    else:
                        newNN.enableConnection(newNN.getEdge(node1,node2))
                    self.latestConnectionID += 1
                #Disabling Connections:
                while random.random() < self.disableConnectionRate:
                    edgeIndex = round(random.random() * newNN.getNumEdges())
                    newNN.disableNode(edgeIndex, edgeIndex)
                self.NNs.append(newNN)

    def makeSpeciesLists(self, NNs, difLimit):
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
        for NN in self.NNs:
            if NN.fitness > bestNN.fitness:
                bestNN = NN
        return bestNN
    
    def getNeuralNetwork(self, index):
        return self.NNs[index]

    def getPopulationAverageFitness(self):
        total = 0
        for NN in self.NNs:
            total += NN.fitness
        return total/len(self.NNs)
    
    #MODIFIERS:
    def setFitness(self,index,fitness):
        self.NNs[index].fitness = fitness

    #FILE READING/WRITING:
    def savePopulation(self):
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
            newPop.NNs.append(NeuralNetwork.networkFromString(lines[currLine:(currLine+numLines)]))
            currLine += numLines
        return newPop
    
    def getBestModelString(self):
        return self.getBestModel().getNetworkString()

