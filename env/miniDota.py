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
    def __init__(self, args):
        self.xlimit, self.ylimit = 50, 50
        self.baseX, self.baseY = 30, 30
        self.maxTime = 500
        
        self.posX, self.posY = 0, 0#random.randint(0, 50), random.randint(0, 50)
        self.timestamp = 0
        self.enemyBaseHealth = 100
        self.done = 0
        self.shortestDist = self.__dist__(self.posX, self.posY, self.baseX, self.baseY)
        self.args = args
        self.env_info = {'observations':np.array([[self.posX, self.posY, self.enemyBaseHealth]]), 
                         'rewards':[0], 'local_done':[0]}
    
    def reset(self):
        self.posX, self.posY = 0, 0#random.randint(0, 50), random.randint(0, 50)
        self.timestamp = 0
        self.enemyBaseHealth = 100
        self.done = 0
        self.shortestDist = self.__dist__(self.posX, self.posY, self.baseX, self.baseY)
        return {'observations':np.array([[self.posX, self.posY, self.enemyBaseHealth]]), 
                'rewards':[0], 'local_done':[0]}
    
    def step(self, action):
        prevDist = self.__dist__(self.posX, self.posY, self.baseX, self.baseY)
        
        if self.args.actionType == 'continuous':
            attack = (action[0][4] > 3)
            self.posX = self.posX - action[0][0] + action[0][1]
            self.posY = self.posY - action[0][2] + action[0][3]
            self.posX = max(self.posX, 0)
            self.posX = min(self.posX, self.xlimit)
            self.posY = max(self.posY, 0)
            self.posY = min(self.posY, self.ylimit)
            
        elif self.args.actionType == 'discrete':
            attack = action[0][4]
            left, right, down, up = action[0][0], action[0][1], action[0][2], action[0][3]
            if left and not right and self.posX > 0:
                self.posX -= 1
            if right and not left and self.posX < self.xlimit:
                self.posX += 1
            elif down and not up and self.posY > 0:
                self.posY -= 1
            elif up and not down and self.posY < self.ylimit:
                self.posY += 1
                
        newDist = self.__dist__(self.posX, self.posY, self.baseX, self.baseY)
        defeat = 0
        hitpoint = 0
        consumption = 0
        
        if attack:
            if prevDist <= 5:# attack distance.
                hitpoint = 1
                self.enemyBaseHealth -= hitpoint
                if self.enemyBaseHealth <= 0:
                    defeat = 1
                    self.done = 1
            else:
                consumption += 1
        
        self.timestamp += 1
        if self.timestamp == self.maxTime:
            self.done = 1
        
        distReward = -newDist
        attackReward = hitpoint
        rewards = 0.1*distReward + 1*attackReward - 0.1*consumption + 100*defeat
        return {'observations':np.array([[self.posX, self.posY, self.enemyBaseHealth]]), 
                'rewards':[rewards], 'local_done':[self.done]}
    
    def close(self):
        pass

    def __dist__(self, x1, y1, x2, y2):
         return math.sqrt((x2-x1)**2 + (y2-y1)**2)
