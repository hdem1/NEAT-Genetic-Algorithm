from math import ceil
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
    
    def getNeuralNetwork(self, index):

    #FILE READING/WRITING:
    def savePopulation(self):
    
    def loadPopulation(self, filename):
    
    def saveBestModel(self):


