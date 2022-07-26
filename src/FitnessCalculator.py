from asyncio import SendfileNotAvailableError


class FitnessCalculator:
    def __init__(self, env):
        self.env = env
        self.fitness = 1 #Should end up between 1 and 1000
        if self.env == "CartPole-v1" or self.env == "MountainCar-v0":
            self.frames = 0

    def update(self, values):
        if self.env == "CartPole-v1":
            self.frames += 1
        elif self.env == "MountainCar-v0":
            self.frames += 1
        '''elif self.env == "Pendulum-v1":
            totalReward += ((1 - abs(obsArray[2][0])/8)**2) * obsArray[0][0] * 10
        elif self.env == "LunarLander-v2":
            totalReward += 3 - (obsArray[1][0] + 1.5)'''

    def getFitness(self):
        if self.env == "CartPole-v1":
            return 1 + self.frames * 999.0/500.0
        elif self.env == "MountainCar-v0":
            return 1000 - (self.frames/200.0)**3 * 999.0