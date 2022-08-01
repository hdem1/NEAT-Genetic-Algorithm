from xmlrpc.client import MININT
import numpy as np

class FitnessCalculator:
    def __init__(self, env):
        self.fitnessMin = 1
        self.fitnessMax = 1000
        self.successBonus = 100
        self.consistencyBonus = 100
        self.env = env
        self.fitness = self.fitnessMin 
        self.results = [] 
        self.reward = 0
        if self.env == "CartPole-v1" or self.env == "MountainCar-v0":
            self.frames = 0
        if self.env == "MountainCarContinuous-v0":
            self.consistencyBonus = 25
            self.successBonus = 0
            self.maxXValue = -1.2

    def update(self, reward, observation):
        if self.env == "CartPole-v1":
            self.frames += 1
        elif self.env == "MountainCar-v0":
            self.frames += 1
        elif self.env == "MountainCarContinuous-v0":
            self.reward += reward
            self.maxXValue = max(self.maxXValue, observation[0])
        else:
            self.reward += reward
        '''elif self.env == "Pendulum-v1":
            totalReward += ((1 - abs(obsArray[2][0])/8)**2) * obsArray[0][0] * 10
        elif self.env == "LunarLander-v2":
            totalReward += 3 - (obsArray[1][0] + 1.5)'''
        
    def getSuccessBonus(self):
        if self.env == "CartPole-v1":
            if self.frames >= 500:
                return self.successBonus
            else:
                return 0
        elif self.env == "MountainCar-v0":
            if self.frames <= 110:
                return self.successBonus
            else:
                return 0
        elif self.env == "LunarLander-v2":
            if self.reward >= 200:
                return self.successBonus
            else:
                return 0
        return 0 

    def getConsistencyBonus(self):
        range = np.amax(self.results) - np.amin(self.results)
        if range / np.average(self.results) < 0.25:
            return self.consistencyBonus 
        return 0

    def runComplete(self):
        #add run to values:
        if self.env == "CartPole-v1":
            value = self.fitnessMin
            value += self.frames * (self.fitnessMax-self.successBonus - self.consistencyBonus)/500.0
            value += self.getSuccessBonus()
            self.results.append(value)
        elif self.env == "MountainCar-v0":
            value = (self.fitnessMax - self.successBonus - self.consistencyBonus) 
            value -= (self.frames/200.0)**3 * (self.fitnessMax-self.successBonus - self.consistencyBonus-self.fitnessMin)
            value += self.getSuccessBonus()
            self.results.append(value)
        elif self.env == "MountainCarContinuous-v0":
            maxValue = 100
            minValue = -999
            maxXValueBonus = 0
            value = (self.fitnessMax - self.fitnessMin - self.consistencyBonus - self.successBonus - maxXValueBonus)
            value *= ((self.reward - minValue) / (maxValue - minValue))**4
            value += self.fitnessMin
            value += self.getSuccessBonus()
            value += maxXValueBonus * (self.maxXValue + 1.2) / 1.8
            self.results.append(value)
        else:
            minReward, maxReward = self.getMinMaxReward()
            value = (self.fitnessMax - self.fitnessMin - self.consistencyBonus - self.successBonus)
            value *= ((self.reward - minReward) / (maxReward - minReward))
            value += self.fitnessMin
            value += self.getSuccessBonus()
            self.results.append(value)

        #Reset values:
        if self.env == "CartPole-v1" or self.env == "MountainCar-v0":
            self.frames = 0
        self.reward = 0

    def getFitness(self):
        output = MININT
        if len(self.results) > 0:  
            output = np.average(self.results) + self.getConsistencyBonus()
        #print(self.results, "-->", output)
        return output
    
    def reset(self):
        self.results = []
    
    def getMinMaxReward(self):
        if self.env == "CartPole-v1":
            return 0, 500
        elif self.env == "MountainCar-v0":
            return -200, 0
        elif self.env == "MountainCarContinuous-v0":
            return -100, 100
        elif self.env == "Acrobot-v1":
            return -100, 0
        elif self.env == "Pendulum-v1":
            return 200 * (-16.2736044), 0
        elif self.env == "BipedalWalker-v3":
            return -150, 350
        elif self.env == "CarRacing-v1":
            return -200, 1000
        elif self.env == "LunarLander-v2":
            return -150, 260
        else:
            return 0, 100