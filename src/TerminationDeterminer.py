
class TerminationDeterminer:

    def __init__(self, env):
        self.env = env
    
    def checkTermination(self, done, totalReward, reward, obs):
        if self.env == "LunarLander-v2": 
            if totalReward < 0:
                return True
        return done
