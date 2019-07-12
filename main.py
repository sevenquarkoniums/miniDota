'''
To do:
    2 accelerate the environment update.
    3 make the numpy calculation into pytorch.
    4 use multiprocessing for game update.
'''

import os
import torch
import argparse
import numpy as np
import torch.optim as optim
from model import ac
from utils.utils import to_tensor, get_action, save_checkpoint
from utils.running_state import ZFilter
from utils.memory import Memory
from agent.ppo import process_memory, train_model
from env.miniDota import miniDotaEnv
import matplotlib.pyplot as plt
from utils.utils import check
import random
import time
import imageio
from scipy.spatial.distance import euclidean
random.seed(1)
torch.manual_seed(1)

parser = argparse.ArgumentParser(description='Setting for agent')
parser.add_argument('--load_model', type=str, default=None)
parser.add_argument('--gamma', type=float, default=0.995, help='discount factor')
parser.add_argument('--lamda', type=float, default=0.95, help='GAE hyper-parameter')
parser.add_argument('--hidden_size', type=int, default=512,
                    help='hidden unit size of actor and critic networks')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--max_iter', type=int, default=1000,
                    help='the number of max iteration')
parser.add_argument('--time_horizon', type=int, default=10000,
                    help='the number of time horizon (step number) T ')
parser.add_argument('--l2_rate', type=float, default=0.001,
                    help='l2 regularizer coefficient')
parser.add_argument('--clip_param', type=float, default=0.1,
                    help='hyper parameter for ppo policy loss and value loss')
parser.add_argument('--randomActionRatio', type=float, default=0.05,
                    help='A minimum number of random action is enforced.')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def draw(record, iteration, unitRange):
    '''
    Draw the action trace of a game.
    '''
    matplotlib.rcParams.update({'font.size': 20})
    for step in range(record.shape[0]):
        states, actions = record[step, :4*12].reshape((12,4)), record[step, 4*12:].reshape((10,4))
        fig, ax = plt.subplots(figsize=(11,10))
        for player in range(10):
            if states[player,0] == 0:
                color = 'green'
            elif states[player,0] == 1:
                color = 'firebrick'
            ax.plot(states[player,1], states[player,2], 'o', markersize=10, color=color)
            if actions[player,0] == 2:
                target = states[player,3]
                if states[target,0] != states[player,0]:
                    playerPos, targetPos = states[player,1:2], states[target,1:2]
                    if euclidean(playerPos, targetPos) <= unitRange[player] and states[target,3] > 0:
                        ax.plot([playerPos[0], targetPos[0]], [playerPos[1], targetPos[1]], '-', color=color)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_xlim(-1, 200)
        ax.set_ylim(-1, 200)
        ax.plot(20, 20, '*', markersize=10, color='green')
        ax.plot(180, 180, '*', markersize=10, color='firebrick')

        plt.title('output/iter%d-step%d' % (iteration, step))
        plt.tight_layout()
        plt.savefig('output/iter%d-step%d.png' % (iteration, step))
        plt.close()

    # make gif.
    images = []
    for istep in range(step+1):
        filename = 'output/iter%d-step%d.png' % (iteration, istep)
        images.append(imageio.imread(filename))
    imageio.mimsave('output/iter%d.gif' % iteration, images)

def main():
    train()
    #behavior()
    
def behavior():
    '''
    Draw the action probability of the agent at different observations.
    '''
    actor = Actor(3, 5, args).to(device)
    critic = Critic(3, args).to(device)
    saved_ckpt_path = os.path.join(os.getcwd(), 'save_model', str(args.load_model))
    ckpt = torch.load(saved_ckpt_path)
    actor.load_state_dict(ckpt['actor'])
    critic.load_state_dict(ckpt['critic'])
    running_state = ZFilter((10, 3), clip=5)
    running_state.rs.n = ckpt['z_filter_n']
    running_state.rs.mean = ckpt['z_filter_m']
    running_state.rs.sum_square = ckpt['z_filter_s']
    actionstr = {0:'left', 1:'right', 2:'down', 3:'up'}

    for enemyBaseHealth in [100,50,1]:
        allinput = []
        for posX in range(51):
            for posY in range(51):
                allinput.append([posX, posY, enemyBaseHealth])
        allinput = np.array(allinput)
        normalized = []
        for i in range(0, 2592, 10):
            normalized.append(running_state(allinput[i:i+10,:]))
        ending = running_state(allinput[2591:2601,:])
        ending2d = np.empty((1, 3))
        ending2d[0,:] = ending[-1,:]
        normalized.append(ending2d)
        allNormalized = np.concatenate(normalized, axis=0)
        with torch.no_grad():
            mu = actor(to_tensor(allNormalized))
        mu = torch.cat([to_tensor(allNormalized), mu], dim=1)
        for action in range(4):
            fig, ax = plt.subplots(figsize=(7,7))
            value = np.empty((51, 51))
            for row in range(51):
                for col in range(51):
                    value[row, col] = mu[51*col + 50 - row, 3+action].item() # (x, y) = (col, 50-row)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.imshow(value, cmap='Greens', interpolation='spline36', vmin=0.05, vmax=0.95)
            plt.colorbar()
            ax.plot(30, 30, '*r', markersize=10)
            plt.title('Health %d Action-%s' % (enemyBaseHealth, actionstr[action]))
            plt.tight_layout()
            plt.savefig('Health%dAction%dat.png' % (enemyBaseHealth, action))
            plt.close()

def train():

    numAgent = 10 # multiple agents are running synchronously.
        # each agent has a different type with different properties.
    numGame = 8 # multiple games running simultaneously.
    print('agent count:', numAgent)
    print('Env num:', numGame)
    
    env = {}
    for game in range(numGame):
        env[game] = miniDotaEnv(args, numAgent)

    # initialize the neural networks.
    # use a single network to share the knowledge.
    net = ac(args).to(device)
    if torch.cuda.is_available():
        net = net.cuda()# useful?

    if args.load_model is not None:
        saved_ckpt_path = os.path.join(os.getcwd(), 'save_model', str(args.load_model))
        ckpt = torch.load(saved_ckpt_path)
        net.load_state_dict(ckpt['net'])

    observations = {}
#    observations, alive = {}, {}
    for game in range(numGame):
        observations[game] = env[game].reset()['observations']
#        observations[game], alive[game] = env[game].reset()['observations']
            # get initial state.

    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    for iteration in range(args.max_iter):
        start = time.time()
        if iteration == 0:
            print('Start iteration 0 ..')
        net.eval()
        memory = []
        for i in range(numGame):
            memory.append( [Memory() for j in range(numAgent)] )
                # memory is cleared at every iter so only the current iteration's sample are used in training.
                # the separation of memory is needed as they need to be processed individually for each episode of game.

        steps = 0
        teamscore = 0 # only for game 0.
        record = []
        gameEnd = np.zeros(numGame).astype(bool)
        
#        lastDones = np.zeros(numAgent).astype(bool)
        while steps <= args.time_horizon: # loop for one round of games.
            if np.all(gameEnd):
                break
            steps += 1
            stateList = []
            for game in range(numGame):
                for agent in range(numAgent):
                    stateList.append(np.expand_dims(observations[game][agent], axis=0))
            stateCombined = np.concatenate(stateList, axis=0) # each env is one row.
            actionDistr = net(to_tensor(stateCombined)) # calculate all envs together.
            actions = get_action(actionDistr)

            for game in range(numGame):
                if not gameEnd[game]:
                    thisGameAction = actions[10*game:10*(game+1), :] # contain actions from all agents.
                    envInfo = env[game].step(thisGameAction) # environment runs one step given the action.
    #                envInfo, nextAlive = env[game].step(thisGameAction) # environment runs one step given the action.
                    nextObs = envInfo['observations'] # get the next state.
                    if game == 0:
                        record.append( np.concatenate([ env[game].getState(), actions[0:10, :].reshape(-1) ]) )
                    rewards = envInfo['rewards']
                    dones = envInfo['local_done']
#                    masks = list(~dones) # cut the return calculation at the done point.
                    masks = [True] * numAgent # no need to mask out the last state-action pair.
    
                    for i in range(numAgent):
    #                    if alive[game][i]:
                        memory[game][i].push(observations[game][i], thisGameAction[i], rewards[i], masks[i])
    
                    if game == 0:
                        teamscore += sum([rewards[x] for x in env[game].getTeam0()])
                    observations[game] = nextObs
    #                alive[game] = nextAlive
    
        #            if (dones[0] and not lastDones[0]) or steps == args.time_horizon:
                    gameEnd[game] = np.all(dones)
                    if gameEnd[game]:
                        assert np.sum(rewards) == 0
                        if game == 0:
                            print('Game 0 score: %f' % teamscore)
                            recordMat = np.stack(record)# stack will expand the dimension before concatenate.
                            draw(recordMat, iteration, env[game].getUnitRange())
                        env[game].reset()
                
        #            lastDones = dones.copy()

        simEnd = time.time()
        net.train() # switch to training mode.

        sts, ats, returns, advants, old_policy, old_value = [], [], [], [], [], []

        for game in range(numGame):
            for i in range(numAgent):
                batch = memory[game][i].sample()
                st, at, rt, adv, old_p, old_v = process_memory(net, batch, args)
                sts.append(st)
                ats.append(at)
                returns.append(rt)
                advants.append(adv)
                old_policy.append(old_p)
                old_value.append(old_v)

        sts = torch.cat(sts)
        ats = torch.cat(ats)
        returns = torch.cat(returns)
        advants = torch.cat(advants)
        old_policy = torch.cat(old_policy)
        old_value = torch.cat(old_value)

        train_model(net, optimizer, sts, ats, returns, advants,
                    old_policy, old_value, args)
            # training is based on the state-action pairs from the current iteration.

        trainEnd = time.time()
        print('Simulation time: %.f' % (simEnd-start))
        print('Training time: %.f' % (trainEnd-simEnd))

        if iteration % 10:
            model_path = os.path.join(os.getcwd(),'save_model')
            if not os.path.isdir(model_path):
                os.makedirs(model_path)

            ckpt_path = os.path.join(model_path, 'ckpt_'+ str(teamscore)+'.pth.tar')

            save_checkpoint({
                'net': net.state_dict(),
                'args': args,
                'score': teamscore
            }, filename=ckpt_path)

if __name__ == "__main__":
    main()
    
    
