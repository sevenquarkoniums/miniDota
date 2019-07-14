'''
miniDota environment.
'''
import numpy as np
from utils.utils import check, unitEmbed
from random import sample
from scipy.spatial.distance import cdist

class miniDotaEnv:
    def __init__(self, args, numAgent):
        self.xlimit, self.ylimit = 200, 200
        self.team0Fountain, self.team1Fountain = (0, 0), (self.xlimit, self.ylimit)
        self.team0Base, self.team1Base = (20, 20), (self.xlimit-20, self.ylimit-20)
        self.maxTime = 1000 # min win time: 182.
        self.unitHealthInit = [600, 800, 1000, 1200, 1400, 1200, 1000, 800, 600, 400, 1000, 1000]# last two are bases.
        self.unitAttack = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 0, 0]
        self.unitRange = [19, 17, 15, 13, 11, 9, 7, 5, 3, 1, 0, 0]
        self.teamFactor = 0.2
        self.args = args
        self.numAgent = numAgent
        self.embed = {}
        for unit in range(12):
            self.embed[unit] = unitEmbed(unit)
    
    def reset(self):
        self.state = np.zeros((self.numAgent+2, 4))
            # each row for an agent; each column for an attribute.
            # last 2 rows are bases.
            # row idx is the agent type.
            # columns. 0:team, 1:x, 2:y, 3:health
        self.team1 = sample(range(10), 5)
        self.team0 = [x for x in range(10) if x not in self.team1]
        for idx in self.team1:
            self.state[idx, 0] = 1 # assign team randomly.
            self.state[idx, 1] = self.xlimit
            self.state[idx, 2] = self.ylimit
        self.state[10, 1] = self.team0Base[0]
        self.state[10, 2] = self.team0Base[1]
        self.state[11, 1] = self.team1Base[0]
        self.state[11, 2] = self.team1Base[1]
        self.state[11, 0] = 1
        self.state[:, 3] = self.unitHealthInit
        self.interaction = np.zeros((12,12))
            # 1: row is attacking col.
            # 0: nothing.
            # -1: row is being attacked by col.
        self.done = False
        self.timestamp = 0
        self.distance = cdist(self.state[:, 1:3], self.state[:, 1:3])
        self.genObs()
#        alive = (self.state[:10,3] > 0)
        return {'observations':self.observations, 'rewards':np.zeros(self.numAgent), 
                'local_done':np.array([False]*10)
                }

    def getState(self):
        return self.state.reshape(-1)
    
    def getTeam0(self):
        return self.team0

    def getUnitRange(self):
        return self.unitRange

    def genObs(self):
        '''
        The observation structure is:
        playerEmbed[10] + allyObs[18]*5 + allyBase[18] + enemyObs[18]*5 + enemyBase[18]
        '''
        self.observations = {}

        # construct the observation from the viewpoint of each agent.
        for agent in range(self.numAgent):
            if agent in self.team0:
                order = self.team0 + [10] + self.team1 + [11]
            elif agent in self.team1:
                order = self.team1 + [11] + self.team0 + [10]
            obs = []
            for obsAgent in order:
                obsOne = np.array([0, self.interaction[agent, obsAgent], self.state[obsAgent, 3]/1000, self.unitAttack[obsAgent]/100, 
                    self.unitRange[obsAgent]/10, self.distance[agent, obsAgent]/100, self.state[obsAgent, 1]/100, self.state[obsAgent, 2]/100])
                obs.append( np.concatenate([self.embed[obsAgent], obsOne]) )
            self.observations[agent] = np.concatenate([self.embed[agent]] + obs)

    def step(self, actions):
        self.distance = cdist(self.state[:, 1:3], self.state[:, 1:3])
        harms = []
        self.interaction = np.zeros((12,12)) # attacking and being attacked.

        for agent in range(self.numAgent):
            harm = 0
            if self.state[agent, 3] > 0:# alive.
                # add fountain regen here.
                actionType, target = actions[agent, 0], actions[agent, 3]
                    # target is agentType.
                if actionType == 2:# attack.
                    if agent in self.team0 and target in self.team1+[11] or agent in self.team1 and target in self.team0+[10]:
                        # attackable enemy.
                        # this implementation requires the agents to remember the correlation between agentType and their embeddings.
                        if self.distance[agent, target] <= self.unitRange[agent] and self.state[target, 3] > 0:
                            # within range and alive.
                            harm = min(self.unitAttack[agent], self.state[target, 3])
                            self.state[target, 3] -= harm
                            self.interaction[agent, target] = 1
                            self.interaction[target, agent] = -1
            harms.append(harm)

        # attack is prior to move.
        for agent in range(self.numAgent):
            if self.state[agent, 3] > 0:# alive.
                actionType, offsetX, offsetY = actions[agent,0], actions[agent,1]-1, actions[agent,2]-1
                if actionType == 1:
                    self.state[agent, 1] = min(self.xlimit, max(0, self.state[agent,1] + offsetX))
                    self.state[agent, 2] = min(self.ylimit, max(0, self.state[agent,2] + offsetY))

        win = [0] * 10
        if self.state[10, 3] <= 0 and self.state[11, 3] <= 0:
            win = [1]*10
            self.done = True
        elif self.state[10, 3] <= 0:
            win = [0,0,0,0,0,1,1,1,1,1]
            self.done = True
        elif self.state[11, 3] <= 0:
            win = [1,1,1,1,1,0,0,0,0,0]
            self.done = True

        self.timestamp += 1
        if self.timestamp == self.maxTime:
            self.done = True

        self.genObs()
        personalRewards = []
        for agent in range(self.numAgent):
            targetBase = 11 if agent in self.team0 else 10
            thisReward = 0.01*harms[agent] + 10*win[agent] - 0.000006*self.distance[agent, targetBase]
            personalRewards.append(thisReward)
        personalRewards = np.array(personalRewards)
        team0mean = np.dot(1-self.state[:10, 0], personalRewards) / 5
        team1mean = np.dot(self.state[:10, 0], personalRewards) / 5
        teamVec = (1-self.state[:10, 0])*team0mean + self.state[:10, 0]*team1mean
        oppoMeanVec = (1-self.state[:10, 0])*team1mean + self.state[:10, 0]*team0mean
#        rewards = (1-self.teamFactor) * personalRewards + self.teamFactor * teamVec
        rewards = (1-self.teamFactor) * personalRewards + self.teamFactor * teamVec - oppoMeanVec # make the game zero-sum by minus the opponent's average.
        dead = (self.state[:10,3] <= 0)
        if self.done:
            local_done = np.array([True] * 10)
        else:
            local_done = dead
        return {'observations':self.observations, 
                'rewards':rewards, 'local_done':local_done
                }
                # copy() may be needed to prevent these value from being changed by processing.

