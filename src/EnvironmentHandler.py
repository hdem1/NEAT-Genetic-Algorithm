from os import environ
import gym
import numpy as np
from NeuralNetwork import NeuralNetwork
from FitnessCalculator import FitnessCalculator

class EnvironmentHandler:

    def __init__(self, environment):
        self.env = gym.make(environment)
        self.environmentName = environment
        self.observationSpace = self.env.observation_space
        self.actionSpace = self.env.action_space
    
    def closeEnvironment(self):
        self.env.close()

    def getActionRanges(self):
        index = 0
        output = []
        if type(self.actionSpace) == gym.spaces.box.Box:
            for i in range(self.actionSpace.shape[0]):
                output.append([self.actionSpace.low[index], self.actionSpace.high[index]])
        if type(self.actionSpace) == gym.spaces.discrete.Discrete:
            return [[0,1] for _ in range(self.actionSpace.n)]#[self.actionSpace.start, self.actionSpace.start + self.actionSpace.n - 1]]
        return output
    
    def getObservationRanges(self):
        index = 0
        output = []
        for i in range(self.observationSpace.shape[0]):
            output.append([self.observationSpace.low[index], self.observationSpace.high[index]])
        return output

    def runSimulation(self, neuralNetwork:NeuralNetwork, maxIterations = -1, displaying = False, modifyReward = True):
        done = False
        obs = self.env.reset()
        
        obsArray = np.array(obs, dtype = object).reshape(len(obs),1)
        iterations = 0
        totalReward = 0
        neuralNetwork.printErrors("Runtime check 1")
        while not done and (maxIterations == -1 or iterations <= maxIterations):
            output = neuralNetwork.getOutput(obsArray)
            action = []
            actionRanges = self.getActionRanges()
            for i in range(len(output)):
                action.append(output[i] * (actionRanges[i][1]-actionRanges[i][0]) + actionRanges[i][0])
            if type(self.actionSpace) == gym.spaces.discrete.Discrete:
                choice = 0
                maxVal = 0
                for i in range(len(action)):
                    if action[i] > maxVal:
                        maxVal = action[i]
                        choice = i
                action = choice
            obs, reward, done, info = self.env.step(action)
            obsArray = np.array(obs, dtype = object).reshape(len(obs), 1)
            totalReward += reward 
            neuralNetwork.updateFitness(reward, obsArray)
            if displaying:
                self.env.render()
            iterations += 1
        neuralNetwork.episodeDone()
        return totalReward, iterations, neuralNetwork.getFitness()
    
    def runMultipleSimulations(self, num_tests:int, neuralNetwork:NeuralNetwork, maxIterations = -1, displaying = False, successChecking = False, successRewardThreshold = 0, modifyReward = True, successIterationThreshold = -1):
        successRate = 0
        avg_reward = 0
        avg_iterations = 0
        avg_fitness = 0
        for t in range(num_tests):
            reward, iterations, fitness = self.runSimulation(neuralNetwork, maxIterations = maxIterations, displaying = displaying, modifyReward = modifyReward)
            avg_reward += reward
            avg_iterations += iterations
            avg_fitness += fitness
            if successChecking:
                if successIterationThreshold >= 0 and iterations >= successIterationThreshold:
                    successRate +=1
                elif successIterationThreshold == -1 and reward >= successRewardThreshold:
                    successRate +=1
        avg_reward /= num_tests
        avg_iterations /= num_tests
        avg_fitness /= num_tests
        if successChecking:
            successRate /= num_tests
            return avg_reward, avg_iterations, neuralNetwork.getFitness(), successRate
        return avg_reward, avg_iterations, neuralNetwork.getFitness()
            
    def getEnvironmentName(self):
        return self.environmentName