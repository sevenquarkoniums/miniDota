'''
A mini-dota RL experiment.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import math
import random
from collections import namedtuple
import matplotlib.pyplot as plt
from sys import exit
import os
if os.name == 'nt':
    import pygame
#random.seed(0)
#torch.manual_seed(0)

DEVICE = torch.device('cuda')
Transition = namedtuple('Transition', ('state', 'action', 'nextstate', 'reward'))

def main():
    m = miniDota(saveModel=1, loadModel=1)
    m.train()
#    m.test()
    del m

class miniDota:
    def __init__(self, saveModel, loadModel):
        self.episodeNum = 3000
        self.trainPerEpisode = 10
        self.batchSize = 256
        self.learningRate = 0.001
        self.gamma = 0.999
        self.memorySize = 10000
        self.thresStart, self.thresEnd, self.thresDecay = 0.3, 0.05, 1000
        
        self.saveModel = saveModel
        self.loadModel = loadModel

    def __del__(self):
        torch.cuda.empty_cache()

    def train(self):
        self.net = RLNet().to(DEVICE)
        self.targetNet = RLNet().to(DEVICE)
        self.targetNet.load_state_dict(self.net.state_dict())
        self.targetNet.eval()
        if self.loadModel:
            self.net.load_state_dict(torch.load('miniDota.pt'))
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.learningRate, momentum=0.9, weight_decay=1e-5)
        self.memory = replayMemory(self.memorySize)
        start = time.time()
        losses = []
        lastTen = []
        for episode in range(self.episodeNum):
            self.actionThres = self.thresEnd + (self.thresStart - self.thresEnd) * math.exp(-episode/self.thresDecay)
            if episode % 10 == 0:
                result, record = self.playGame(recording=1)
                self.draw(record)
                print(result)
            else:
                self.playGame(recording=0)
            loss = self.optimize(iterate=self.trainPerEpisode)
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
            sample = random.random()
            if sample > self.actionThres: # use network to select move.
                with torch.no_grad():
                    values = self.net.forward(state)
                action = torch.argmax(values, dim=1)
            else: # random move.
                action = torch.randint(0, 10, (1, ), device=DEVICE)
            reward, result = game.update(action.item())
            if result in ['win', 'draw']:
                episodeEnd = True
            nextstate = game.getState()
            if recording:
                attack = torch.tensor([[1]], device=DEVICE, dtype=torch.float) if action.item() % 2 == 1 else torch.tensor([[0]], device=DEVICE, dtype=torch.float)
                record.append(torch.cat([state, attack], dim=1))
            else:
                self.memory.push(state, action.unsqueeze(1), nextstate, reward)
        if recording:
            record = torch.cat(record)
        return result, record

    def optimize(self, iterate=1):
        for i in range(iterate):
            if len(self.memory) < self.batchSize:
                return
            transitions = self.memory.sample(self.batchSize)
            batch = Transition(*zip(*transitions))
            stateBatch = torch.cat(batch.state)
            actionBatch = torch.cat(batch.action)
            nextstateBatch = torch.cat(batch.nextstate)
            rewardBatch = torch.cat(batch.reward)
            values = self.net(stateBatch).gather(1, actionBatch)
            nextstateValues = self.targetNet(nextstateBatch).max(1)[0]
            expectedValues = (nextstateValues * self.gamma) + rewardBatch
            loss = F.smooth_l1_loss(values, expectedValues.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            for param in self.net.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
        return loss

    def test(self):
        self.net = RLNet().to(DEVICE)
        self.net.load_state_dict(torch.load('miniDota.pt'))
        self.actionThres = 0.05
        result, record = self.playGame(recording=1)
        self.draw(record)
        print(result)
        
    def draw(self, record):
        if os.name == 'nt':
            fig, ax = plt.subplots(figsize=(7,6))
            x, y, attack = record[:, 0].tolist(), record[:, 1].tolist(), record[:, 3].tolist()
            ax.plot(x, y, '-')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_xlim(-1, 51)
            ax.set_ylim(-1, 51)
            ax.plot(50, 50, '*r')
            attX, attY = [], []
            for idx, a in enumerate(attack):
                if a == 1:
                    attX.append(x[idx])
                    attY.append(y[idx])
            ax.plot(attX, attY, 'x', color='chocolate')
            plt.show()

class gameEngine:
    def __init__(self):
        self.xlimit, self.ylimit = 50, 50
        self.baseX, self.baseY = 50, 50
        self.posX, self.posY = 0, 0
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
                    return torch.tensor([300], device=DEVICE, dtype=torch.float), 'win'
            else:
                consumption += 0.5
        self.timestamp += 1
        if self.timestamp == self.maxTime:
            return torch.tensor([0], device=DEVICE, dtype=torch.float), 'draw'
        
#        if newDist < self.shortestDist:
#            distReward = self.shortestDist - newDist
#            self.shortestDist = newDist
#        else:
#            distReward = 0
        distReward = prevDist - newDist
        attackReward = hitpoint
        reward = 1 * distReward + 1 * attackReward - 1 * consumption
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
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc2(x))
        x = self.sm(self.fc3(x))
        return x

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
