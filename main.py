'''
To do:
    1 use mask to make an end.
    2 accelerate the environment update.
    3 make the numpy calculation into pytorch.
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
import sys
from utils.utils import check
import random
random.seed(1)
torch.manual_seed(1)

parser = argparse.ArgumentParser(description='Setting for agent')
parser.add_argument('--load_model', type=str, default=None)
parser.add_argument('--gamma', type=float, default=0.995, help='discount factor')
parser.add_argument('--lamda', type=float, default=0.95, help='GAE hyper-parameter')
parser.add_argument('--hidden_size', type=int, default=512,
                    help='hidden unit size of actor and critic networks')
parser.add_argument('--critic_lr', type=float, default=0.0001)
parser.add_argument('--actor_lr', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--max_iter', type=int, default=1000,
                    help='the number of max iteration')
parser.add_argument('--time_horizon', type=int, default=1000,
                    help='the number of time horizon (step number) T ')
parser.add_argument('--l2_rate', type=float, default=0.001,
                    help='l2 regularizer coefficient')
parser.add_argument('--clip_param', type=float, default=0.1,
                    help='hyper parameter for ppo policy loss and value loss')
parser.add_argument('--activation', type=str, default='tanh',
                    help='you can choose between tanh and swish')
parser.add_argument('--actionType', type=str, default='discrete',
                    help='The action is either discrete or continuous.')
parser.add_argument('--randomActionRatio', type=float, default=0.05,
                    help='A minimum number of random action is enforced.')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def draw(record, baseX, baseY, iteration):
    '''
    Draw the action trace of an agent.
    '''
    record = np.stack(record)
    fig, ax = plt.subplots(figsize=(6,5))
    x, y = record[:, 0].tolist(), record[:, 1].tolist()
#    stop, attack = record[:, 3].tolist(), record[:, 4].tolist()
    ax.plot(x, y, '-')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim(-1, 51)
    ax.set_ylim(-1, 51)
    ax.plot(baseX, baseY, '*r', markersize=10)
#    attX, attY = [], []
#    stopX, stopY = [], []
#    for idx, a in enumerate(attack):
#        if a == 1:
#            attX.append(x[idx])
#            attY.append(y[idx])
#        if stop[idx] == 1:
#            stopX.append(x[idx])
#            stopY.append(y[idx])
#    ax.plot(attX, attY, 'x', markersize=3, color='chocolate')
#    ax.plot(stopX, stopY, 'o', markersize=3, color='forestgreen')
    plt.title('iteration-%d' % iteration)
    plt.tight_layout()
    plt.savefig('output/iter-%d.png' % iteration)
    plt.close()

def check(var):
    print(var)
    i = input()
    if i == 'q':
        sys.exit()

def main():
    train()
    #behavior()
    
def behavior():
    '''
    Draw the action probability of the agent at different states.
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

    num_inputs = 10+17*10
    num_actions = 15
    numAgent = 10 # multiple agents are running synchronously.
        # each agent has a different type with different properties.
    numGame = 8 # multiple games running simultaneously.
    print('state size:', num_inputs)
    print('action size:', num_actions)
    print('agent count:', numAgent)
    print('game num:', numGame)
    
    env = {}
    for game in range(numGame):
        env[game] = miniDotaEnv(args, numAgent)

    # initialize the neural networks.
    # use a single network to share the knowledge.
    net = ac(num_inputs, num_actions, args).to(device)
    if torch.cuda.is_available():
        net = net.cuda()

    if args.load_model is not None:
        raise NotImplementedError
        saved_ckpt_path = os.path.join(os.getcwd(), 'save_model', str(args.load_model))
        ckpt = torch.load(saved_ckpt_path)

        actor.load_state_dict(ckpt['actor'])
        critic.load_state_dict(ckpt['critic'])

    states = {}
    for game in range(numGame):
        states[game] = env[game].reset()['observations']
            # get initial state.
            # states: 2d tensor.

    actor_optim = optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=args.critic_lr,
                              weight_decay=args.l2_rate)

    scores = []
    score_avg = 0

    for iter in range(args.max_iter):
        if iter == 0:
            print('Start iterations..')
        actor.eval(), critic.eval()
        memory = [Memory() for _ in range(numAgent)]
            # memory is cleared at every iter so only the current iteration's sample are used in training.
            # the separation of memory into agents seems unnecessary, as they are finally combined.

        steps = 0
        score = 0
        record = []
        
#        lastDones = np.zeros(numAgent).astype(bool)
        while steps <= args.time_horizon: # loop for one round of games.
            steps += 1
            stateList = []
            for game in range(numGame):
                stateList.append(states[game])
            stateCombined = np.concatenate(stateList, axis=0)
            mu = actor(to_tensor(stateCombined))
            actions = get_action(mu, None, args.actionType)

            for game in range(numGame):
                thisGameAction = actions[10*game:10*game+10, :] # contain actions from all agents.
                envInfo = env[game].step(thisGameAction) # environment runs one step given the action.
                nextState = envInfo['observations']
                    # get the next state.
                if game == 0:
                    record.append( np.concatenate([ envInfo['observations'][0], actions[0] ], axis=0) )
                rewards = envInfo['rewards']
    #            dones = envInfo['local_done']

    #            masks = list(~(np.array(dones))) # cut the return calculation at the done point.
                masks = [True] * numAgent

                for i in range(numAgent):
                    memory[i].push(states[game][i], thisGameAction[i], rewards[i], masks[i])

                score += np.sum(rewards)
                states[game] = nextState

    #            if (dones[0] and not lastDones[0]) or steps == args.time_horizon:
                if steps == args.time_horizon:# currently env doesn't end in itself.
                    env[game].reset()
                    scores.append(score)
                    if game == numGame - 1:
                        score = 0 # reset score.
                        print('Avg game and avg agent score for this iteration: %f' % np.mean(scores))
                        #draw(record, 30, 30, iter)
            
    #            lastDones = dones.copy()

        actor.train(), critic.train() # switch to training mode.

        sts, ats, returns, advants, old_policy, old_value = [], [], [], [], [], []

        for i in range(numAgent):
            batch = memory[i].sample()
            st, at, rt, adv, old_p, old_v = process_memory(actor, critic, batch, args)
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

        train_model(actor, critic, actor_optim, critic_optim, sts, ats, returns, advants,
                    old_policy, old_value, args)
            # training is based on the state-action pairs from the current iteration.

        if iter % 20:
            score_avg = int(score_avg)

            model_path = os.path.join(os.getcwd(),'save_model')
            if not os.path.isdir(model_path):
                os.makedirs(model_path)

            ckpt_path = os.path.join(model_path, 'ckpt_'+ str(score_avg)+'.pth.tar')

            save_checkpoint({
                'actor': actor.state_dict(),
                'critic': critic.state_dict(),
                'args': args,
                'score': score_avg
            }, filename=ckpt_path)

if __name__ == "__main__":
    main()
    
    
