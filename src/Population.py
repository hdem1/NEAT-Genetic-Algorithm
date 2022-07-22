from math import ceil
from xmlrpc.client import MININT
import numpy as np
from NeuralNetwork import NeuralNetwork
import random
from os.path import exists, expanduser, isdir
from os import mkdir

class Population:
    #CONSTRUCTION:
    def __init__(self, envName, size):
        self.NNs = []
        self.envName = envName
        self.size = size
        self.combinationRatio = 0.3
        self.randomRatio = 0.2
        self.mutationRatio = 1 - self.combinationRatio - self.randomRatio
    
    def makeInitialPopulation(self):
    
    #EVOLUTION:
    def evolveGeneration(self):
        speciesLists = makeSpeciesLists(self.NNs, 1)
        avgFitness = getPopulationAverageFitness()
        self.NNs = []
        for species in speciesLists:
            #Get the new number of NNs in that species:
            speciesSize = len(species)
            popTotalFitness = 0
            for NN in species:
                popTotalFitness += NN.fitness
            newNum = round(popTotalFitness/avgFitness)

            #Preserve the best one:
            bestNN = species[0]
            for NN in species:
                if NN.fitness > bestNN.fitness:
                    bestNN = NN
            self.NNs.append()
            

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

    #FILE READING/WRITING:
    def savePopulation(self):
    
    def loadPopulation(self, filename):
    
    def saveBestModel(self):


