'''
To do:
    1 use mask to make an end.
    2 accelerate the environment update.
    3 make the numpy calculation into pytorch.
    4 add another player.
    5 add RNN.
'''

import os
import torch
import argparse
import numpy as np
import torch.optim as optim
from model import Actor, Critic
from utils.utils import to_tensor, get_action, save_checkpoint
from utils.running_state import ZFilter
from utils.memory import Memory
from agent.ppo import process_memory, train_model
from env.miniDota import miniDotaEnv
import matplotlib.pyplot as plt
import sys

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
#    behavior()
    
def behavior():
    '''
    Have problem.
    '''
    actor = Actor(3, 5, args).to(device)
    critic = Critic(3, args).to(device)
    saved_ckpt_path = os.path.join(os.getcwd(), 'save_model', str(args.load_model))
    ckpt = torch.load(saved_ckpt_path)
    actor.load_state_dict(ckpt['actor'])
    critic.load_state_dict(ckpt['critic'])

    for enemyBaseHealth in [100, 50, 1, 0]:
        allinput = []
        for posX in range(51):
            for posY in range(51):
                allinput.append([posX, posY, enemyBaseHealth])
        with torch.no_grad():
            inputs = to_tensor(allinput)
            mu = actor(inputs)
            mu = torch.cat([inputs, mu], dim=1)
        for action in range(4):
            fig, ax = plt.subplots(figsize=(7,7))
            value = np.empty((51, 51))
            for row in range(51):
                for col in range(51):
                    value[row, col] = mu[51*col + 50 - row, 3+action].item() # (x, y) = (col, 50-row)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_xlim(-1, 51)
            ax.set_ylim(-1, 51)
            value[0, 50] = 1
            value[0, 0] = 0
            ax.imshow(value, cmap='hot', interpolation='nearest')
            ax.plot(30, 30, '*r', markersize=10)
            plt.title('Health %d Action-%d' % (enemyBaseHealth, action))
            plt.tight_layout()
            plt.savefig('Health%dAction%dat.png' % (enemyBaseHealth, action))
            plt.close()

def train():
    torch.manual_seed(1)

    num_inputs = 3
    num_actions = 5
    num_agent = 10 # multiple agents are running synchronously.
    print('state size:', num_inputs)
    print('action size:', num_actions)
    print('agent count:', num_agent)
    
    env = miniDotaEnv(args, num_agent)
    env_info = env.reset()

    # ZFilter handles a running average of states.
    running_state = ZFilter((num_agent,num_inputs), clip=5)

    # initialize the neural networks.
    actor = Actor(num_inputs, num_actions, args).to(device)
    critic = Critic(num_inputs, args).to(device)
    if torch.cuda.is_available():
        actor = actor.cuda()
        critic = critic.cuda()

    if args.load_model is not None:
        saved_ckpt_path = os.path.join(os.getcwd(), 'save_model', str(args.load_model))
        ckpt = torch.load(saved_ckpt_path)

        actor.load_state_dict(ckpt['actor'])
        critic.load_state_dict(ckpt['critic'])

        running_state.rs.n = ckpt['z_filter_n']
        running_state.rs.mean = ckpt['z_filter_m']
        running_state.rs.sum_square = ckpt['z_filter_s']

        print("Loaded OK ex. Zfilter N {}".format(running_state.rs.n))

    states = running_state(env_info['observations']) # get initial state.
        # states: 2d tensor.
        # the running_state() will normalize the input.

    actor_optim = optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=args.critic_lr,
                              weight_decay=args.l2_rate)

    scores = []
    score_avg = 0

    for iter in range(args.max_iter):
        if iter == 0:
            print('Start iterations..')
        actor.eval(), critic.eval()
        memory = [Memory() for _ in range(num_agent)]
            # memory is cleared at every iter so only the current iteration's sample are used in training.

        steps = 0
        score = 0
        record = [np.array([0,0,100,0,0])] # record states for drawing.
        
#        lastDones = np.zeros(num_agent).astype(bool)
        while steps <= args.time_horizon: # loop for one round of game.
            steps += 1
            if args.actionType == 'continuous':
                mu, std, _ = actor(to_tensor(states)) # get action probability from the actor network.
                actions = get_action(mu, std, args.actionType) # get certain action from probability.
            elif args.actionType == 'discrete':
                mu = actor(to_tensor(states))
                actions = get_action(mu, None, args.actionType)

            env_info = env.step(actions) # environment runs one step given the action.
            next_states = running_state(env_info['observations']) # get the next state.
            record.append(np.concatenate([env_info['observations'][0], np.array([actions[0, 0]]), np.array([actions[0, 4]])], axis=0))
            rewards = env_info['rewards']
#            dones = env_info['local_done']

#            masks = list(~(np.array(dones))) # cut the return calculation at the done point.
            masks = [True] * num_agent

            for i in range(num_agent):
                memory[i].push(states[i], actions[i], rewards[i], masks[i])

            score += rewards[0] # only for one agent.
            states = next_states

#            if (dones[0] and not lastDones[0]) or steps == args.time_horizon:
            if steps == args.time_horizon:
                scores.append(score)
                score = 0
                episodes = len(scores)
                if episodes % 10 == 0:
                    score_avg = np.mean(scores[-min(10, episodes):])
                    print('{}-th episode : last 10 episode mean score of 1st agent is {:.2f}'.format(
                        episodes, score_avg))
                draw(record, 30, 30, iter)
                env.reset()
            
#            lastDones = dones.copy()

        actor.train(), critic.train() # change to training mode.

        sts, ats, returns, advants, old_policy, old_value = [], [], [], [], [], []

        for i in range(num_agent):
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

        if iter % 100:
            score_avg = int(score_avg)

            model_path = os.path.join(os.getcwd(),'save_model')
            if not os.path.isdir(model_path):
                os.makedirs(model_path)

            ckpt_path = os.path.join(model_path, 'ckpt_'+ str(score_avg)+'.pth.tar')

            save_checkpoint({
                'actor': actor.state_dict(),
                'critic': critic.state_dict(),
                'z_filter_n':running_state.rs.n,
                'z_filter_m': running_state.rs.mean,
                'z_filter_s': running_state.rs.sum_square,
                'args': args,
                'score': score_avg
            }, filename=ckpt_path)

    env.close()

if __name__ == "__main__":
    main()
    
    