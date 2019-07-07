'''
Env.
'''
import math
import numpy as np
import sys

def check(var):
    print(var)
    i = input()
    if i == 'q':
        sys.exit()

class miniDotaEnv:
    def __init__(self, args, num_agent):
        self.xlimit, self.ylimit = 50, 50
        self.baseX, self.baseY = 30, 30
        self.maxTime = 500
        self.enemyBaseHealthInit = 100
        self.args = args
        self.num_agent = num_agent
        self.reset()
    
    def reset(self):
        self.stateArray = np.zeros((self.num_agent, 3))
            # each row for an agent; each column for an attribute.
            # col-0: posX, col-1: posY, col-2:enemyBaseHealth, 
        self.stateArray[:, 2] = [self.enemyBaseHealthInit] * self.num_agent
        self.done = np.zeros(self.num_agent).astype(bool)
        self.timestamp = 0
        return {'observations':self.stateArray.copy(), 'rewards':np.zeros(self.num_agent).copy(), 
                'local_done':self.done.copy()}
    
    def step(self, action):
        prevDist = self.__dist__() # return a 1-d array.
        
        if self.args.actionType == 'continuous':
            raise NotImplementedError
            attack = (action[0][4] > 3)
            self.posX = self.posX - action[0][0] + action[0][1]
            self.posY = self.posY - action[0][2] + action[0][3]
            self.posX = max(self.posX, 0)
            self.posX = min(self.posX, self.xlimit)
            self.posY = max(self.posY, 0)
            self.posY = min(self.posY, self.ylimit)
            
        elif self.args.actionType == 'discrete':
            left, right, down, up = action[:, 0], action[:, 1], action[:, 2], action[:, 3]
            attack = action[:, 4]
            self.stateArray[:, 0] = np.minimum(np.maximum(self.stateArray[:, 0] + right - left, 0), self.xlimit)
            self.stateArray[:, 1] = np.minimum(np.maximum(self.stateArray[:, 1] + up - down, 0), self.xlimit)
#            if left and not right and self.posX > 0:
#                self.posX -= 1
#            if right and not left and self.posX < self.xlimit:
#                self.posX += 1
#            elif down and not up and self.posY > 0:
#                self.posY -= 1
#            elif up and not down and self.posY < self.ylimit:
#                self.posY += 1
                
        newDist = self.__dist__()
        consumption = np.zeros(self.num_agent)
        
        hitpoint = np.logical_and(np.logical_and(np.less_equal(prevDist, 5), attack), ~self.done).astype(int)
        self.stateArray[:, 2] = self.stateArray[:, 2] - hitpoint # enemy base health.
        self.done = np.less_equal(self.stateArray[:, 2], 0)
        consumption = np.logical_and(np.logical_or(np.greater(prevDist, 5), self.done), attack).astype(int)
#        if attack:
#            if prevDist <= 5:# attack distance.
#                hitpoint = 1
#                self.enemyBaseHealth -= hitpoint
#                if self.enemyBaseHealth <= 0:
#                    defeat = 1
#                    self.done = 1
#            else:
#                consumption += 1
        
        self.timestamp += 1
        
#        if self.timestamp == self.maxTime:
#            self.done = 1
        
        distReward = -newDist
        attackReward = hitpoint
        rewards = 0.1*distReward + 1*attackReward - 0.1*consumption + 100*self.done.astype(int)
        return {'observations':self.stateArray.copy(), 
                'rewards':rewards.copy(), 'local_done':self.done.copy()}
    
    def close(self):
        pass

    def __dist__(self):
         return np.sqrt((self.stateArray[:,0]-self.baseX*np.ones(self.num_agent))**2 + \
                          (self.stateArray[:,1]-self.baseY*np.ones(self.num_agent))**2)
