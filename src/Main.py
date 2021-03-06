from NeuralNetwork import NeuralNetwork
from AlgorithmManager import AlgorithmManager


algorithms = ["LL"]
# C = cartpole
# A = acrobot
# P = pendulum
# MC = mountain Car
# MCC = mountain Car Continuous
# T = testing:
# LL = lunar lander
# BW = Bipedal Walker
training = False
loading = False
displaying = True
modifyingReward = False

algoManager = AlgorithmManager("CartPole-v1", 20, 1000)
algoManager.train()

'''for algorithm in algorithms:
    genAlgo = None
    if algorithm == "T":
        NN = NeuralNetwork.randomBaseNetwork(3,3)
    elif algorithm == "C":
        algoManager = AlgorithmManager("CartPole-v1", 10, 1000, hiddenLayerSizes=[3,3], activationFunctions= ["LR","LR"], numTestsPerChild=3)
        if loading:
            algoManager.loadModel("CartPole-v1_gens-10_children-1000_layers-[4, 3, 3, 1]_networkTests-3.txt")
    elif algorithm == "A":
        algoManager = AlgorithmManager("Acrobot-v1", 10, 5000, hiddenLayerSizes=[5,5], activationFunctions= ["LR","LR"], numTestsPerChild=3)
        if loading:
            algoManager.loadModel("Acrobot-v1_gens-10_children-5000_layers-[6, 5, 5, 1]_networkTests-3.txt")
    elif algorithm == "P":
        algoManager = AlgorithmManager("Pendulum-v1", 20, 5000, hiddenLayerSizes=[5,5], activationFunctions= ["LR","LR"], numTestsPerChild=10)
        if loading:
            algoManager.loadModel("Pendulum-v1_gens-20_children-5000_layers-[3, 5, 5, 1]_networkTests-10_1.txt")
    elif algorithm == "MC":
        algoManager = AlgorithmManager("MountainCar-v0", 15, 5000, hiddenLayerSizes=[3,3], activationFunctions= ["LR","LR"], numTestsPerChild=5)
        if loading:
            algoManager.loadModel("MountainCar-v0_gens-15_children-5000_layers-[2, 3, 3, 1]_networkTests-5.txt")
    elif algorithm == "MCC":
        algoManager = AlgorithmManager("MountainCarContinuous-v0", 10, 1000, hiddenLayerSizes=[3,3], activationFunctions= ["LR","LR"], numTestsPerChild=10)
        if loading:
            algoManager.loadModel("MountainCarContinuous-v0_gens-10_children-1000_layers-[2, 3, 3, 1]_networkTests-3.txt")
    elif algorithm == "LL":
        # Trained with 20 generations with a big emphasis on flying time and then back to normal reward
        algoManager = AlgorithmManager("LunarLander-v2", 20, 1000, hiddenLayerSizes=[8,8], activationFunctions= ["LR","LR"], numTestsPerChild=10)
        if loading:
            #genAlgo.loadModel("LunarLander-v2_gens-20_children-10000_layers-[8, 8, 8, 1]_networkTests-3.txt")
            algoManager.loadModel("LunarLander-v2_gens-20_children-1000_layers-[8, 8, 8, 1]_networkTests-10_1.txt")
    elif algorithm == "BW":
        #first training was with iterations < 10 * reward + 500 -> avg reward = 130 (didn't realize 1600 step limit)
        #tested with iterations < 6 * reward + 500 -> avg reward = 
        algoManager = AlgorithmManager("BipedalWalker-v3", 40, 5000, hiddenLayerSizes=[10,10], activationFunctions= ["LR","LR"], numTestsPerChild=3)
        if loading:
            #genAlgo.loadModel("BipedalWalker-v3_gens-20_children-5000_layers-[24, 10, 10, 4]_networkTests-2.txt")
            #genAlgo.loadModel("BipedalWalker-v3_gens-30_children-5000_layers-[24, 10, 10, 4]_networkTests-2.txt")
            #genAlgo.loadModel("BipedalWalker-v3_gens-40_children-5000_layers-[24, 10, 10, 4]_networkTests-3.txt")
            algoManager.loadModel("BipedalWalker-v3_gens-40_children-5000_layers-[24, 10, 10, 4]_networkTests-3_1.txt")
            
    if training:
        algoManager.train(modifyReward=modifyingReward)
        algoManager.testBest(1000)
    if displaying:
        #genAlgo.testBest(1000)
        algoManager.displayBest(printRewards = True)
    genAlgo.close()'''