from tkinter import N
from NeuralNetwork import NeuralNetwork
from AlgorithmManager import AlgorithmManager


algorithms = ["MCC"]
# C = cartpole
# A = acrobot
# P = pendulum
# MC = mountain Car
# MCC = mountain Car Continuous
# LL = lunar lander
# BW = Bipedal Walker
loading = False
trainingGenerations = 0
displaying = True
id = -1

for algorithm in algorithms:
    algoManager = None
    if algorithm == "C":
        if loading:
            id = 0
        algoManager = AlgorithmManager("CartPole-v1", 1000, id = id)
    elif algorithm == "A":
        if loading:
            id = 0
        algoManager = AlgorithmManager("CartPole-v1", 1000, numTestsPerChild=3, id = id)
    elif algorithm == "P":
        if loading:
            id = 0
        algoManager = AlgorithmManager("Pendulum-v1", 1000, numTestsPerChild=10, id = id)
    elif algorithm == "MC":
        if loading:
            id = 42
        algoManager = AlgorithmManager("MountainCar-v0", 1000, numTestsPerChild=5, id = id)
    elif algorithm == "MCC":
        if loading:
            id = 0
        algoManager = AlgorithmManager("MountainCarContinuous-v0", 1000, numTestsPerChild=10, id = id)
    elif algorithm == "LL":
        if loading:
            id = 0
        algoManager = AlgorithmManager("LunarLander-v2", 1000, numTestsPerChild=10, id = id)
    elif algorithm == "BW":
        if loading:
            id = 0
        algoManager = AlgorithmManager("BipedalWalker-v3", 5000, numTestsPerChild=3, id = id)
            
    if trainingGenerations > 0:
        algoManager.train(trainingGenerations)
        algoManager.testBest(1000)
    if displaying:
        algoManager.displayBest(printRewards = True)
    algoManager.close()