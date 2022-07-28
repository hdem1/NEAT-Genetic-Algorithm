from math import ceil
from tokenize import String
from xmlrpc.client import MAXINT, MININT
import numpy as np
from EnvironmentHandler import EnvironmentHandler
from Population import Population
from NeuralNetwork import NeuralNetwork
import random
from os.path import exists, expanduser, isdir
from os import mkdir

class AlgorithmManager:

    def __init__(self, env, numChildren, numTestsPerChild = 5, survivalRate = 0.1, id = -1):
        self.envHandler = EnvironmentHandler(env)
        self.actionRanges = self.envHandler.getActionRanges()
        self.obsRanges = self.envHandler.getObservationRanges()
        self.population = Population(env, len(self.obsRanges), len(self.actionRanges), numChildren, survivalRate=survivalRate)
        self.numChildren = numChildren
        self.numTestsPerChild = numTestsPerChild
        self.survivalRate = survivalRate
        self.numGenerationsDone = 0
        self.modelSaved = False
        self.id = id
        self.rootFolder = expanduser("~/Documents/Random Coding Projects/MachineLearningExperiments/NEAT-Genetic-Algorithm/")
        self.bestModelFolder = "data/bestModels/" + env + "/"
        self.bestGenerationModelFolder = "data/bestGenerationModels/" + env + "/"
        self.trainingLogFolder = "data/trainingLogs/" + env + "/"
        if not isdir(self.rootFolder + self.trainingLogFolder):
            mkdir(self.rootFolder+self.trainingLogFolder)
        if not isdir(self.rootFolder + self.bestGenerationModelFolder):
            mkdir(self.rootFolder+self.bestGenerationModelFolder)
        if not isdir(self.rootFolder + self.bestModelFolder):
            mkdir(self.rootFolder+self.bestModelFolder)
        if self.id == -1:
            self.id = 0
            while isdir(self.rootFolder + self.trainingLogFolder + env + "_" + str(self.id) + "/"):
                self.id+=1
            self.trainingLogFolder = self.trainingLogFolder + env + "_" + str(self.id) + "/"
            self.bestGenerationModelFolder = self.bestGenerationModelFolder + env + "_" + str(self.id) + "/"
            mkdir(self.rootFolder + self.trainingLogFolder)
            mkdir(self.rootFolder + self.bestGenerationModelFolder)
        elif isdir(self.rootFolder + self.trainingLogFolder + env + "_" + str(self.id) + "/"):
            self.trainingLogFolder = self.trainingLogFolder + env + "_" + str(self.id) + "/"
            self.bestGenerationModelFolder = self.bestGenerationModelFolder + env + "_" + str(self.id) + "/"
            self.loadGeneration(-1)
        else:
            self.trainingLogFolder = self.trainingLogFolder + env + "_" + str(self.id) + "/"
            mkdir(self.rootFolder + self.trainingLogFolder)
        self.bestModelFilename = env + "_" + str(self.id) + "_0.txt"
    
    def simulateGeneration(self, printProgress = True, modifyReward=False):
        if printProgress:
            print("Progress: [", end ="", flush = True)
            lastPrint = 0
        if self.numGenerationsDone == 0:
            self.population.makeInitialPopulation()
        else:
            self.population.evolveGeneration()
        for i in range(self.population.size):
            self.population.printPopulationErrors(self.population.NNs, "Error Check 7")
            if printProgress and i - lastPrint >= 0.1 * self.numChildren:
                lastPrint = i
                print("*",end = "", flush = True)
            reward, iterations, fitness = self.envHandler.runMultipleSimulations(self.numTestsPerChild, self.population.getNeuralNetwork(i), modifyReward=modifyReward)#, displaying = True))
            #self.population.setFitness(i,fitness, self.numTestsPerChild)
            #print(rewards[i])
            self.population.printPopulationErrors(self.population.NNs, "Error Check 8")
        if printProgress:
            print("]")
        self.numGenerationsDone += 1
        if printProgress:
            print("Average Fitness (0 to 1000) =", self.population.getPopulationAverageFitness())
            bestModel, _ = self.population.getBestModel()
            print("Best Fitness (0 to 1000) =", bestModel.getFitness())
            print("Number of Species = ", len(self.population.makeSpeciesLists(self.population.NNs)))
        
    def train(self, numGenerations, printProgress = True, displayBest = True, numDisplayIterations = 2, saveOldModel = True, saveBestModelPerGen = True, saveEachGeneration = True, endTests = 100, modifyReward = False):
        if saveOldModel == True and self.modelSaved:
            self.modelSaved = False
            self.filename = self.makeNewModelFileName()
        printing = printProgress
        for i in range(numGenerations):
            if printing:
                print("\nGeneration ",(i+1),":", sep ="")
            self.simulateGeneration(printProgress = printing, modifyReward = modifyReward)
            if saveBestModelPerGen:
                self.saveBestModel(generation = True)
            if saveEachGeneration:
                self.saveGeneration()
            if displayBest:
                bestModel, _ = self.population.getBestModel()
                self.envHandler.runMultipleSimulations(numDisplayIterations, bestModel, displaying=True)
        #Resorting the final set with more data:
        print("\nSorting Best Networks...")
        #fitnesses = []
        bestSet, indices = self.population.getBestModels(0.1)
        for i in range(len(bestSet)):
            reward, iterations, fitness = self.envHandler.runMultipleSimulations(endTests, bestSet[i])
            #self.population.setFitness(indices[i], fitness, endTests)
            #fitnesses.append(self.population)
        # maxIndices = []
        # for _ in range(len(bestSet)):
        #     maxFitness = MININT
        #     maxIndex = 0
        #     for j in range(len(bestSet)):
        #         if rewards[j] > maxReward and maxIndex not in maxIndices:
        #             maxReward = rewards[j]
        #             maxIndex = j
        #     maxIndices.append(maxIndex)
        #     self.population.setFitness(indices[maxIndex],maxReward)
        print("\nSorting Completed")

    def testBest(self, numTests, saving = True):
        avg_reward = 0
        print("\nTesting for", numTests, "iterations...")
        bestModel, index = self.population.getBestModel()
        avg_reward, avg_iterations, avg_fitness = self.envHandler.runMultipleSimulations(numTests, bestModel)
        #self.population.setFitness(index, avg_fitness, numTests)
        print("\nTest Results:")
        indent = "   "
        print(indent, "- Average Reward = ", avg_reward)
        if saving:
            self.saveBestModel()
            self.savePerformance(avg_reward, avg_iterations, printInfo = False)

    def displayBest(self, iterations = -1, printRewards = False):
        i = 0
        while i < iterations or iterations == -1:
            bestModel, _ = self.population.getBestModel()
            avg_reward, avg_iterations, avg_fitness = self.envHandler.runSimulation(bestModel,displaying = True)
            if printRewards:
                print("Reward = ", avg_reward, "; Iterations = ", avg_iterations, sep = "")
            i += 1

    def makeNewModelFileName(self):
        value = 0
        filename = self.envHandler.environmentName + "_" + str(self.id)
        while (exists(self.rootFolder + self.bestModelFolder + filename + "_" + str(value) + ".txt")):
            value+=1
        filename = filename + "_" + str(value)
        self.bestModelFilename = filename +".txt"
    
    def saveBestModel(self, generation = False):
        if generation:
            file = open(self.rootFolder + self.bestGenerationModelFolder + "generation_" + str(self.numGenerationsDone) + "_bestModel.txt", "w")
        if not generation:
            self.makeNewModelFileName()
            file = open(self.rootFolder+self.bestModelFolder+self.bestModelFilename, "w")

        #Writing data:
        file.write(self.population.getBestModelString())

        file.close()
    
    def startNewFile(self):
        self.bestModelFilename = self.makeNewModelFileName()
        self.modelSaved = False
    
    def loadGeneration(self, genNum):  
        if genNum == -1: #get last generation:
            genNum = 0
            while exists(self.rootFolder + self.trainingLogFolder + "generation_" + str(genNum+1) + ".txt"):
                genNum += 1
        file = open(self.rootFolder + self.trainingLogFolder + "generation_" + str(genNum) + ".txt", "r")
        self.population = Population.loadPopulation(file.readlines())
        self.numGenerationsDone = genNum
        file.close()

    def saveGeneration(self):
        file = open(self.rootFolder + self.trainingLogFolder + "generation_" + str(self.numGenerationsDone) + ".txt", "w")
        file.write(self.population.getString())
        file.close()

    
    def loadModel(self, filename):
        self.modelSaved = False
        self.filename = filename
        file = open(self.folder + filename, "r")
        lines = file.readlines()
        NN = NeuralNetwork(self.envHandler.environmentName)
        NN.makeModelFromStrings(lines)
        self.population = Population(self.envHandler.environmentName, len(self.obsRanges), len(self.actionRanges), self.population.getSize(), survivalRate=self.survivalRate)
        self.population.NNs[0] = NN
    
    def savePerformance(self, reward, iterations, printInfo = True):
        #All following lines = training performances
        if exists(self.rootFolder + self.bestModelFolder + self.bestModelFilename):
            if printInfo:   
                print("Saving performance statistics...")
            file = open(self.rootFolder + self.bestModelFolder + self.bestModelFilename, "a")
            lastline = []
            lastline.append(str(reward)+",")
            lastline.append(str(iterations)+"\n")
            file.writelines(lastline)
            file.close()


    def close(self):
        self.envHandler.closeEnvironment()