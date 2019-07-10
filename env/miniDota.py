'''
miniDota environment.
'''
import numpy as np
import sys
from utils.utils import check, unitEmbed
from random import sample

class miniDotaEnv:
    def __init__(self, args, numAgent):
        self.xlimit, self.ylimit = 200, 200
        self.team1Fountain, self.team2Fountain = (0, 0), (self.xlimit, self.ylimit)
        self.team1Base, self.team2Base = (20, 20), (self.xlimit-20, self.ylimit-20)
        self.maxTime = 2000
        self.baseHealthInit = 1000
        self.agentHealthInit = [600, 800, 1000, 1200, 1400, 1200, 1000, 800, 600, 400]
        self.agentAttack = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
        self.agentRange = [20, 18, 16, 14, 12, 10, 7, 5, 3, 1]
        self.args = args
        self.numAgent = numAgent
        self.embed = {}
        for agent in range(numAgent):
            self.embed[agent] = unitEmbed(agent)
        self.reset()
    
    def reset(self):
        self.state = np.zeros((self.numAgent+2, 4))
            # each row for an agent; each column for an attribute.
            # row idx is the agent type.
            # columns. 0:team, 1:x, 2:y, 3:health
        team2 = sample(range(10))
        for idx in team2:
            self.state[idx, 0] = 1
            self.state[idx, 1] = self.xlimit
            self.state[idx, 2] = self.ylimit
        self.state[11, 0] = 1
        self.state[10:, 3] = [self.baseHealthInit] * 2
        self.state[:10, 3] = self.agentHealthInit
        self.done = False
        self.prevDone = False
        self.timestamp = 0
        observations = {}
        for agent in range(self.numAgent):
            thisObs = [self.embed(agent)]
            for obsAgent in range(10):
                state = np.array([self.state[obsAgent, 0], self.state[obsAgent, 3]/1000, self.agentAttack[obsAgent]/100, 
                    self.agentRange[obsAgent]/10, distance/100, self.state[obsAgent, 1]/100, self.state[obsAgent, 2]/100])
                thisObs.append( np.concatenate([self.embed(obsAgent), state], axis=1) )
            # add base.
            observations[agent] = np.concatenate(thisObs, axis=1)
        return {'observations':observations, 'rewards':np.zeros(self.numAgent).copy(), 
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
            self.state[:, 0] = np.minimum(np.maximum(self.state[:, 0] + right - left, 0), self.xlimit)
            self.state[:, 1] = np.minimum(np.maximum(self.state[:, 1] + up - down, 0), self.xlimit)
#            if left and not right and self.posX > 0:
#                self.posX -= 1
#            if right and not left and self.posX < self.xlimit:
#                self.posX += 1
#            elif down and not up and self.posY > 0:
#                self.posY -= 1
#            elif up and not down and self.posY < self.ylimit:
#                self.posY += 1
                
#        newDist = self.__dist__()
        
        hitpoint = np.logical_and(np.logical_and(np.less_equal(prevDist, 5), attack), ~self.done).astype(int)
        self.state[:, 2] = self.state[:, 2] - hitpoint # enemy base health.
        self.done = np.less_equal(self.state[:, 2], 0)
        self.defeat = np.logical_and(self.done, ~self.prevDone)
        self.prevDone = self.done
#        consumption = np.logical_and(np.logical_or(np.greater(prevDist, 5), self.done), attack).astype(int)
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
        
#        distReward = -newDist
        attackReward = hitpoint
        rewards = 1*attackReward + 100*self.defeat.astype(int)
#        rewards = 0.1*distReward + 10*attackReward - 0.1*consumption + 1000*self.defeat.astype(int)
        return {'observations':self.state.copy(), 
                'rewards':rewards.copy(), 'local_done':self.done.copy()}
                # copy() needed to prevent these value from being changed by processing.
    
    def close(self):
        pass

    def __dist__(self):
         return np.sqrt((self.state[:,0]-self.baseX*np.ones(self.numAgent))**2 + \
                          (self.state[:,1]-self.baseY*np.ones(self.numAgent))**2)
