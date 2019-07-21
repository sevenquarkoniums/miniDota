'''
To do:
    1 update observation: 
        correct team label in the observation; 
        add alive state; 
        use relative position.
        shuffle the observation unit order.
    2 accelerate the environment update.
    3 may make the numpy calculation into pytorch.
    4 use multiprocessing for game update.
    5 use multi-GPU.
'''

import os
import torch
import argparse
import numpy as np
import torch.optim as optim
from model import ac
from utils.utils import to_tensor, get_action, save_checkpoint
from utils.memory import Memory
from agent.ppo import process_memory, train_model
from env.miniDota import miniDotaEnv
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from utils.utils import check
import random
import time
import imageio
from scipy.spatial.distance import euclidean
random.seed(0)
torch.manual_seed(0)

parser = argparse.ArgumentParser(description='Setting for agent')
parser.add_argument('--load_model', type=str, default=None)
parser.add_argument('--gamma', type=float, default=0.995, help='discount factor')
parser.add_argument('--lamda', type=float, default=0.95, help='Policy gradient hyper-parameter')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--max_iter', type=int, default=10000,
                    help='the number of iteration in the training.')
parser.add_argument('--time_horizon', type=int, default=10000,
                    help='the number of max time step for each game. Not used now.')
parser.add_argument('--l2_rate', type=float, default=0.001,
                    help='l2 regularizer coefficient')
parser.add_argument('--clip_param', type=float, default=0.1,
                    help='hyper parameter for ppo policy loss and value loss')
parser.add_argument('--cpuSimulation', action='store_true', 
                    help='Force using CPU for simulation.')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if args.cpuSimulation:
    print('Using CPU for simulation.')

def main():
#    train()
#    behavior()
    test(interval=1, runs=1)
    
def draw(record, iteration, unitRange, interval):
    '''
    Draw the action trace of a game.
    Draw frame by frame, and concatenate them into gif.
    '''
    matplotlib.rcParams.update({'font.size': 20})
    for step in range(0, record.shape[0], interval):
        states, actions = record[step, :4*12].reshape((12,4)), record[step, 4*12:(4*12+4*10)].reshape((10,4))
        actionDistr = record[step, (4*12+4*10):].reshape((10,21))

        fig = plt.figure(1, figsize=(15,9.5))
        gridspec.GridSpec(4,3)

        # draw the players on the map.
        plt.subplot2grid((4,3), (0,0), rowspan=4, colspan=2)
        for player in range(10):
            if states[player,0] == 0:# team 0.
                color = 'green'
            elif states[player,0] == 1:# team 1.
                color = 'firebrick'
#            alpha = 0.2 * (player % 5 + 1)
            if states[player,3] <= 0:
                color = 'grey'
#                alpha = 1
            plt.plot(states[player,1], states[player,2], 'o', markersize=10, color=color)
            if actions[player,0] == 2 and states[player,3] > 0:
                target = int(actions[player,3])
                if states[target,0] != states[player,0]:
                    playerPos, targetPos = states[player,1:3], states[target,1:3]
                    if euclidean(playerPos, targetPos) <= unitRange[player] and states[target,3] > 0:
                        plt.plot([playerPos[0], targetPos[0]], [playerPos[1], targetPos[1]], '-', color=color)
        plt.plot(20, 20, '*', markersize=15, color='green', alpha=states[10, 3] / 1000)
        plt.plot(180, 180, '*', markersize=15, color='firebrick', alpha=states[11, 3] / 1000)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlim(-1, 200)
        plt.ylim(-1, 200)
        plt.title('miniDota step%d' % (step))

        # draw bar graphs for action probability for all players.
        tsize = 14
        plt.subplot2grid((4,3), (0,2))
        x, y, color = [], [], []
        for player in range(10):
            x = x + [4*player+1, 4*player+2, 4*player+3]
            y = y + list(actionDistr[player, 0:3])
            if states[player,3] > 0:
                color = color + ['c', 'b', 'r']
            else:
                color = color + ['grey']*3
            plt.axvline(x=4*player)
            if states[player,0] == 0:
                textcolor = 'green'
            elif states[player,0] == 1:
                textcolor = 'firebrick'
            plt.text(4*player+2, 1.1, str(player), size=tsize, color=textcolor, va='center', ha='center')
        plt.bar(x, y, width=0.8, color=color)
        plt.xlim(0, 40)
        plt.ylim(0, 1.2)
        plt.xticks([])
        plt.xlabel('Action Probability')

        plt.subplot2grid((4,3), (1,2))
        x, y, color = [], [], []
        for player in range(10):
            x = x + [4*player+1, 4*player+2, 4*player+3]
            y = y + list(actionDistr[player, 3:6])
            if states[player,3] > 0:
                color = color + ['gold', 'c', 'orange']
            else:
                color = color + ['grey']*3
            plt.axvline(x=4*player)
            if states[player,0] == 0:
                textcolor = 'green'
            elif states[player,0] == 1:
                textcolor = 'firebrick'
            plt.text(4*player+2, 1.1, str(player), size=tsize, color=textcolor, va='center', ha='center')
        plt.bar(x, y, width=0.8, color=color)
        plt.xlim(0, 40)
        plt.ylim(0, 1.2)
        plt.xticks([])
        plt.xlabel('Horizontal Move Probability')

        plt.subplot2grid((4,3), (2,2))
        x, y, color = [], [], []
        for player in range(10):
            x = x + [4*player+1, 4*player+2, 4*player+3]
            y = y + list(actionDistr[player, 6:9])
            if states[player,3] > 0:
                color = color + ['gold', 'c', 'orange']
            else:
                color = color + ['grey']*3
            plt.axvline(x=4*player)
            if states[player,0] == 0:
                textcolor = 'green'
            elif states[player,0] == 1:
                textcolor = 'firebrick'
            plt.text(4*player+2, 1.1, str(player), size=tsize, color=textcolor, va='center', ha='center')
        plt.bar(x, y, width=0.8, color=color)
        plt.xlim(0, 40)
        plt.ylim(0, 1.2)
        plt.xticks([])
        plt.xlabel('Vertical Move Probability')

        plt.subplot2grid((4,3), (3,2))
        x, y, color = [], [], []
        for player in range(4):# only some players are drawn.
            x = x + [13*player+x for x in range(1, 13)]
            y = y + list(actionDistr[player, 9:])
            if states[player,3] > 0:
                color = color + ['blue']*10+['green']+['firebrick']
            else:
                color = color + ['grey']*12
            plt.axvline(x=13*player)
            if states[player,0] == 0:
                textcolor = 'green'
            elif states[player,0] == 1:
                textcolor = 'firebrick'
            plt.text(13*player+6, 1.1, str(player), size=tsize, color=textcolor, va='center', ha='center')
        plt.bar(x, y, width=0.8, color=color)
        plt.xlim(0, 52)
        plt.ylim(0, 1.2)
        plt.xticks([])
        plt.xlabel('Target Probability')

        plt.tight_layout()
        plt.savefig('output/step%d.png' % (step))
        plt.close()

    # make gif.
    images = []
    for istep in range(0, step+1, interval):
        filename = 'output/step%d.png' % (istep)
        images.append(imageio.imread(filename))
    imageio.mimsave('output/iter%d.gif' % iteration, images, duration=0.05)

def behavior():
    '''
    Obsolete.
    Draw the action probability of the agent at different observations.
    '''
    actor = Actor(3, 5, args).to(device)
    critic = Critic(3, args).to(device)
    saved_ckpt_path = os.path.join(os.getcwd(), 'save_model', str(args.load_model))
    ckpt = torch.load(saved_ckpt_path)
    actor.load_state_dict(ckpt['actor'])
    critic.load_state_dict(ckpt['critic'])
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
        # Only one network is created, different agent gets their 
        # own behavior according to the embedding input.
    numGame = 20 # multiple games running simultaneously.
    print('agent count:', numAgent)
    print('Env num:', numGame)
    
    env = {}
    for game in range(numGame):
        env[game] = miniDotaEnv(args, numAgent)

    # initialize the neural networks.
    # use a single network to share the knowledge.
    net = ac(args)
    if not args.cpuSimulation:
        net = net.to(device)

    if args.load_model is not None:
        saved_ckpt_path = os.path.join(os.getcwd(), 'save_model', str(args.load_model))
        ckpt = torch.load(saved_ckpt_path)
        net.load_state_dict(ckpt['net'])

    observations, lastDone = {}, {}
    for game in range(numGame):
        observations[game] = env[game].reset(0)['observations'] # get initial state.
        lastDone[game] = [False] * 10 # to record whether game is done at the previous step.

    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    for iteration in range(args.max_iter):# playing-training iteration.
        start = time.time()
        print()
        print('Start iteration %d ..' % iteration)
        if args.cpuSimulation:
            net = net.cpu()
        net.eval()# switch to evaluation mode.
        memory = []
        for i in range(numGame):
            memory.append( [Memory() for j in range(numAgent)] )
                # memory is cleared at every iter so only the current iteration's samples are used in training.
                # the separation of memory according to game is necessary as they 
                # need to be processed separate for each game.

        steps = 0
        teamscore = 0 # only for game 0.
        record = [] # record the states for visualization.
        gameEnd = np.zeros(numGame).astype(bool)

        while steps <= args.time_horizon: # loop for one game.
            if np.all(gameEnd):
                break
            steps += 1
            stateList = []
            for game in range(numGame):
                for agent in range(numAgent):
                    stateList.append(np.expand_dims(observations[game][agent], axis=0))
            stateCombined = np.concatenate(stateList, axis=0)
            # concatenate the states of all games and process them by the network together.
            with torch.no_grad():
                actionDistr = net(to_tensor(stateCombined, args.cpuSimulation))
            actions = get_action(actionDistr)

            for game in range(numGame):
                if not gameEnd[game]:
                    # the following random action cannot work, because random action has too small prob density value,
                    # leading to strange bugs.
#                    sample = random.random()
#                    if sample > args.randomActionRatio * (1 - min(1, iteration/1000) ):
#                        thisGameAction = actions[10*game:10*(game+1), :] # contain actions from all agents.
#                        check(thisGameAction)
#                    else:
#                        actionmove = np.random.randint(0, 3, size=(10,3))
#                        target = np.random.randint(0, 12, size=(10,1))
#                        thisGameAction = np.concatenate([actionmove, target], axis=1)
                    thisGameAction = actions[10*game:10*(game+1), :] # select the actions from all agents of this env.
                    envInfo = env[game].step(thisGameAction) # environment runs one step given the action.
                    nextObs = envInfo['observations'] # get the next state.
                    if game == 0:
                        record.append( np.concatenate([ env[game].getState(), actions[0:10, :].reshape(-1) ]) )
                    rewards = envInfo['rewards']
                    dones = envInfo['local_done']
#                    masks = list(~dones) # cut the return calculation at the done point.
                    masks = [True] * numAgent # no need to mask out the last state-action pair, 
                                              # because the last reward is useful to us.
    
                    for i in range(numAgent):
                        if not lastDone[game][i]:
                            memory[game][i].push(observations[game][i], thisGameAction[i], rewards[i], masks[i])
                    lastDone[game] = dones
                    if game == 0:
                        teamscore += sum([rewards[x] for x in env[game].getTeam0()])
                    observations[game] = nextObs
    
                    gameEnd[game] = np.all(dones)
                    if gameEnd[game]:
                        if game == 0:
                            print('Game 0 score: %f' % teamscore)
#                            recordMat = np.stack(record)# stack will expand the dimension before concatenate.
#                            draw(recordMat, iteration, env[game].getUnitRange(), 10)
                        observations[game] = env[game].reset(iteration+1)['observations']
                        lastDone[game] = [False] * 10
        
        simEnd = time.time()
        print('Simulation time: %.f' % (simEnd-start))

        net.train() # switch to training mode.
        net = net.cuda()

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
            # training is based on the state-action pairs from all games of the current iteration.

        trainEnd = time.time()
        print('Training time: %.f' % (trainEnd-simEnd))

        if iteration % 10 == 0:
            model_path = os.path.join(os.getcwd(),'save_model')
            if not os.path.isdir(model_path):
                os.makedirs(model_path)

            ckpt_path = os.path.join(model_path, 'ckpt_%.3f.pth.tar' % teamscore)

            save_checkpoint({
                'net': net.state_dict(),
                'args': args,
                'score': teamscore
            }, filename=ckpt_path)

def test(interval, runs):
    print('Testing..')
    numAgent = 10
    numGame = 1
    assert numGame==1 # needed.
    env = {0:miniDotaEnv(args, numAgent)}
    net = ac(args)
    if not args.cpuSimulation:
        net = net.to(device)
    saved_ckpt_path = os.path.join(os.getcwd(), 'save_model', str(args.load_model))
    ckpt = torch.load(saved_ckpt_path)
    net.load_state_dict(ckpt['net'])
    net.eval()
    observations = {0:env[0].reset(0)['observations']}

    for iteration in range(runs):
        start = time.time()
        print()
        print('Start iteration %d ..' % iteration)
        if args.cpuSimulation:
            net = net.cpu()
        steps = 0
        teamscore = 0
        gameEnd = np.zeros(numGame).astype(bool)
        record = []
        teamLabel = env[0].getState().reshape((12, 4))[:10, 0]
        
        while steps <= args.time_horizon: # loop for one round of games.
            if np.all(gameEnd):
                break
            steps += 1
            stateList = []
            for game in range(numGame):
                for agent in range(numAgent):
                    stateList.append(np.expand_dims(observations[game][agent], axis=0))
            stateCombined = np.concatenate(stateList, axis=0)
            with torch.no_grad():
                actionDistr = net(to_tensor(stateCombined, args.cpuSimulation)) # calculate all envs together.
            actions = get_action(actionDistr)

            for game in range(numGame):
                if not gameEnd[game]:
                    thisGameAction = actions[10*game:10*(game+1), :] # contain actions from all agents.
#                    for player in range(10):
#                        if teamLabel[player] == 0 and steps < 100:
#                            thisGameAction[player] = [0, 1, 1, 0] # ablation test.
                    envInfo = env[game].step(thisGameAction) # environment runs one step given the action.
                    nextObs = envInfo['observations'] # get the next state.
                    allAction = np.concatenate([actionDistr[x] for x in range(1, 5)], axis=1)
                    record.append( np.concatenate([ env[0].getState(), 
                                                   actions[0:10, :].reshape(-1),
                                                   allAction.reshape(-1)
                                                   ]) )
                    rewards = envInfo['rewards']
                    dones = envInfo['local_done']
                    teamscore += sum([rewards[x] for x in env[0].getTeam0()])
                    observations[game] = nextObs
    
                    gameEnd[game] = np.all(dones)
                    if gameEnd[game]:
                        print('Team 0 score: %f' % teamscore)
                        simEnd = time.time()
                        print('Simulation time: %.f' % (simEnd-start))
                        recordMat = np.stack(record)# stack will expand the dimension before concatenate.
                        draw(recordMat, iteration, env[game].getUnitRange(), interval)
                        observations[game] = env[game].reset(iteration+1)['observations']
        
        drawEnd = time.time()
        print('Drawing time: %.f' % (drawEnd-simEnd))

if __name__ == "__main__":
    main()
    
    
