'''
A mini-dota RL experiment.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import time
import math
import random
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
from sys import exit
import os
#if os.name == 'nt':
#    import pygame
random.seed(0)
torch.manual_seed(0)

method = 'dqn' # dqn, ppo.
DEVICE = torch.device('cuda')
if method == 'dqn':
    Transition = namedtuple('Transition', ('state', 'action', 'nextstate', 'reward'))
elif method == 'ppo':
    Transition = namedtuple('Transition', ('state', 'action', 'advantage', 'value', 'net_log_pi'))

def main():
    m = miniDota(saveModel=0, loadModel=0)
    m.train()
#    m.test(randomness=0.1)
    m.behavior()
    del m

class miniDota:
    def __init__(self, saveModel, loadModel):
        self.episodeNum = 600
        self.trainPerEpisode = 1
        self.batchSize = 32
        self.learningRate = 0.01
        self.gamma = 0.999
        self.memorySize = 100000 #may clear memory for ppo.
        self.thresStart, self.thresEnd, self.thresDecay = 1, 0.05, 200
        self.showPeriod = 10
        self.targetUpdatePeriod = 4
        
        self.saveModel = saveModel
        self.loadModel = loadModel

    def __del__(self):
        torch.cuda.empty_cache()

    def train(self):
        if method == 'dqn':
            self.net = RLNet().to(DEVICE)
            self.targetNet = RLNet().to(DEVICE)
            self.targetNet.load_state_dict(self.net.state_dict())
            self.targetNet.eval()
            if self.loadModel:
                self.net.load_state_dict(torch.load('miniDota.pt'))
        elif method == 'ppo':
            self.net = PPONet().to(DEVICE)
        
#        self.optimizer = optim.SGD(self.net.parameters(), lr=self.learningRate, momentum=0.9, weight_decay=1e-5)
#        self.optimizer = optim.RMSprop(self.net.parameters(), lr=self.learningRate, alpha=0.95, eps=1e-8, weight_decay=1e-5, momentum=0.95, centered=False)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learningRate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5, amsgrad=False)
        self.memory = replayMemory(self.memorySize)
        start = time.time()
        losses = []
        lastTen = []
        for episode in range(self.episodeNum):
            self.actionThres = self.thresEnd + (self.thresStart - self.thresEnd) * math.exp(-episode/self.thresDecay)
            if episode % self.showPeriod == 0:
                result, record, baseX, baseY = self.playGame(recording=1)
                self.draw(record, baseX, baseY)
                print(result)
            else:
                self.playGame(recording=0)
            loss = self.optimize(iterate=self.trainPerEpisode, clip_range=0.1)
            if loss is not None:
                losses.append(loss)
                lastTen.append(loss)
            if (episode+1) % 10 == 0:
                print('Episode %d' % (episode+1))
            if len(lastTen) == 10:
                print('10-episode avg Loss: %.6f' % (sum(lastTen)/len(lastTen)))
                lastTen = []
            if self.saveModel and (episode+1) % 100 == 0:
                torch.save(self.net.state_dict(), 'miniDota.pt')
            if (episode+1) % self.targetUpdatePeriod == 0:
                self.targetNet.load_state_dict(self.net.state_dict())
        end = time.time()
        print('Time: %.f s' % (end-start))
        
        if os.name == 'nt':
            fig, ax = plt.subplots(figsize=(8,6))
            ax.plot(losses, '-')
            ax.set_xlabel('Optimize Step')
            ax.set_ylabel('Loss')

    def playGame(self, recording):
        episodeEnd = False
        game = gameEngine()
        record = []
        while not episodeEnd:
            state = game.getState()
            self.state = state
            
            sample = random.random()
            if sample > self.actionThres: # use network to select move.
                with torch.no_grad():
                    if method == 'dqn':
                        values = self.net.forward(state)
                        action = torch.argmax(values, dim=1)
                    elif method == 'ppo':
                        pi, value = self.net.forward(state)
            else: # random move.
                action = torch.randint(0, 10, (1, ), device=DEVICE)
            
            reward, result = game.update(action.item())
            nextstate = game.getState()
            
            if recording: # not pushing to training set.
                attack = torch.tensor([[1]], device=DEVICE, dtype=torch.float) if action.item() % 2 == 1 else torch.tensor([[0]], device=DEVICE, dtype=torch.float)
                record.append(torch.cat([state, attack], dim=1))
            else:
                if method == 'dqn':
                    self.memory.push(state, action.unsqueeze(1), nextstate, reward)
                elif method == 'ppo':
                    advantage = self._calc_advantages(result, reward, value)
                    neg_log_pi = -pi.log_prob(pi.sample())
                    self.memory.push(state, action, advantage, value, neg_log_pi)

            if result in ['win', 'draw']:
                episodeEnd = True
        
        if recording:
            record = torch.cat(record)
        return result, record, game.baseX, game.baseY

    def optimize(self, iterate, clip_range):
        for i in range(iterate):
            if len(self.memory) < self.batchSize:
                return
            transitions = self.memory.sample(self.batchSize)
            batch = Transition(*zip(*transitions))
            
            if method == 'dqn':
                stateBatch = torch.cat(batch.state)
                actionBatch = torch.cat(batch.action)
                nextstateBatch = torch.cat(batch.nextstate)
                rewardBatch = torch.cat(batch.reward)
                values = self.net(stateBatch).gather(1, actionBatch)
                nextstateValues = self.targetNet(nextstateBatch).max(1)[0]
                expectedValues = (nextstateValues * self.gamma) + rewardBatch
    #            loss = F.smooth_l1_loss(values, expectedValues.unsqueeze(1))
                loss = nn.MSELoss(size_average=None, reduce=None, reduction='mean')(values, expectedValues.unsqueeze(1))
            elif method == 'ppo':
                sampled_obs = torch.cat(batch.state)
                sampled_action = torch.cat(batch.action)
                sampled_advantage = torch.cat(batch.advantage)
                sampled_value = torch.cat(batch.value)
                sampled_normalized_advantage = self._normalize(sampled_advantage)
                sampled_neg_log_pi = torch.cat(batch.net_log_pi)
                sampled_return = sampled_value + sampled_advantage
                pi, value = self.net.forward(sampled_obs)
                neg_log_pi = -pi.log_prob(sampled_action)
                ratio = torch.exp(sampled_neg_log_pi - neg_log_pi)
                clipped_ratio = ratio.clamp(min=1.0 - clip_range, max=1.0 + clip_range)
                policy_reward = torch.min(ratio * sampled_normalized_advantage, clipped_ratio * sampled_normalized_advantage)
                policy_reward = policy_reward.mean()
                clipped_value = sampled_value + (value - sampled_value).clamp(min=-clip_range, max=clip_range)
                vf_loss = torch.max((value - sampled_return) ** 2, (clipped_value - sampled_return) ** 2)
                vf_loss = 0.5 * vf_loss.mean()
                entropy_bonus = pi.entropy()
                entropy_bonus = entropy_bonus.mean()
                loss = - ( policy_reward - 0.5 * vf_loss + 0.01 * entropy_bonus )
            
            self.optimizer.zero_grad()
            loss.backward()
            if method == 'dqn':
                for param in self.net.parameters():
                    param.grad.data.clamp_(-1, 1)
            elif method == 'ppo':
                nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.5)
            self.optimizer.step()
        return loss

    def _calc_advantages(self, dones, rewards, values):
        advantage = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)
        last_advantage = 0
        _, last_value = self.net.forward(self.state)
        last_value = last_value.cpu().data.numpy()

        for t in reversed(range(self.worker_steps)):
            mask = 1.0 - dones[:, t]
            last_value = last_value * mask
            last_advantage = last_advantage * mask
            delta = rewards[:, t] + self.gamma * last_value - values[:, t]
            last_advantage = delta + self.gamma * self.lamda * last_advantage

            advantage[:, t] = last_advantage

            last_value = values[:, t]
        return advantage
    
    def _normalize(array):
        return (array - array.mean()) / (array.std() + 1e-8)
    
    def test(self, randomness=0.05):
        self.net = RLNet().to(DEVICE)
        self.net.load_state_dict(torch.load('miniDota.pt'))
        self.actionThres = randomness
        result, record, baseX, baseY = self.playGame(recording=1)
        self.draw(record, baseX, baseY)
        print(result)
        
    def draw(self, record, baseX, baseY):
        if os.name == 'nt':
            fig, ax = plt.subplots(figsize=(6,5))
            x, y, attack = record[:, 0].tolist(), record[:, 1].tolist(), record[:, 3].tolist()
            ax.plot(x, y, '-')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_xlim(-1, 51)
            ax.set_ylim(-1, 51)
            ax.plot(baseX, baseY, '*r', markersize=10)
            attX, attY = [], []
            for idx, a in enumerate(attack):
                if a == 1:
                    attX.append(x[idx])
                    attY.append(y[idx])
            ax.plot(attX, attY, 'x', color='chocolate')
            plt.show()
            
    def behavior(self):
        self.net = RLNet().to(DEVICE)
        self.net.load_state_dict(torch.load('miniDota.pt'))
        
        for enemyBaseHealth in [100, 50, 0]:
            allinput = []
            for posX in range(51):
                for posY in range(51):
                    allinput.append([posX, posY, enemyBaseHealth])
            with torch.no_grad():
                values = self.net.forward(torch.tensor(allinput, device=DEVICE, dtype=torch.float))
            actions = torch.argmax(values, dim=1).tolist()
            fig, ax = plt.subplots(figsize=(12,12))
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_xlim(-1, 51)
            ax.set_ylim(-1, 51)
            ax.plot(30, 30, '*r', markersize=10)
            for idx, act in enumerate(actions):
                halfsize, headW, headL = 0.3, 0.3, 0.2
                x, y = idx // 51, idx % 51
                if act % 5 == 1:
                    ax.arrow(x+halfsize, y, -2*halfsize, 0, head_width=headW, head_length=headL, fc='g', ec='g')
                elif act % 5 == 2:
                    ax.arrow(x-halfsize, y, 2*halfsize, 0, head_width=headW, head_length=headL, fc='r', ec='r')
                elif act % 5 == 3:
                    ax.arrow(x, y+halfsize, 0, -2*halfsize, head_width=headW, head_length=headL, fc='c', ec='c')
                elif act % 5 == 4:
                    ax.arrow(x, y-halfsize, 0, 2*halfsize, head_width=headW, head_length=headL, fc='b', ec='b')
            plt.title('EnemyBaseHealth = %d' % enemyBaseHealth)
            plt.tight_layout()
            plt.savefig('MoveAtHealth%d.png' % enemyBaseHealth)
            plt.close()

class gameEngine:
    def __init__(self):
        self.xlimit, self.ylimit = 50, 50
        self.baseX, self.baseY = 30, 30
        self.posX, self.posY = 0, 0#random.randint(0, 50), random.randint(0, 50)
        self.timestamp = 0
        self.maxTime = 1000
        self.enemyBaseHealth = 100
        self.shortestDist = self.__dist__(self.posX, self.posY, self.baseX, self.baseY)
    
    def getState(self):
        return torch.tensor([[self.posX, self.posY, self.enemyBaseHealth]], device=DEVICE, dtype=torch.float)

    def update(self, action):
        prevDist = self.__dist__(self.posX, self.posY, self.baseX, self.baseY)
        if action % 5 == 1 and self.posX > 0:
            self.posX -= 1
        elif action % 5 == 2 and self.posX < self.xlimit:
            self.posX += 1
        elif action % 5 == 3 and self.posY > 0:
            self.posY -= 1
        elif action % 5 == 4 and self.posY < self.ylimit:
            self.posY += 1
        newDist = self.__dist__(self.posX, self.posY, self.baseX, self.baseY)
        
        hitpoint = 0
        consumption = 0
        if action % 2 == 1:# attack.
            if prevDist < 10:
                hitpoint = 1
                self.enemyBaseHealth -= hitpoint
                if self.enemyBaseHealth == 0:
                    return torch.tensor([100], device=DEVICE, dtype=torch.float), 'win'# not consistent.
            else:
                consumption += 1
        self.timestamp += 1
        if self.timestamp == self.maxTime:
            return torch.tensor([0], device=DEVICE, dtype=torch.float), 'draw'
        
#        if newDist < self.shortestDist:
#            distReward = self.shortestDist - newDist
#            self.shortestDist = newDist
#        else:
#            distReward = 0
        distReward = -newDist/50
        attackReward = hitpoint
        reward = 0.1 * distReward + 1 * attackReward - 0 * consumption
        return torch.tensor([reward], device=DEVICE, dtype=torch.float), 'on'

    def __dist__(self, x1, y1, x2, y2):
         return math.sqrt((x2-x1)**2 + (y2-y1)**2)

class RLNet(nn.Module):
    def __init__(self):
        super(RLNet, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)
        self.sm = nn.Softmax(1)

    def forward(self, x):
        y = torch.empty(x.size(), device=DEVICE)
        y[:, :2] = x[:, :2] / 25 - 1
        y[:, 2] = x[:, 2] / 50 - 1
        x = F.relu(self.fc1(y))
        x = F.relu(self.fc2(x))
        x = self.sm(self.fc3(x))
        return x

class PPONet(nn.Module):
    def __init__(self):
        super(PPONet, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x):
        y = torch.empty(x.size(), device=DEVICE)
        y[:, :2] = x[:, :2] / 25 - 1
        y[:, 2] = x[:, 2] / 50 - 1
        x = F.relu(self.fc1(y))
        x = F.relu(self.fc2(x))
        pi = Categorical(logits=self.fc3(x))
        value = self.fc4(x)
        return pi, value

class replayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

if __name__ == '__main__':
    main()
