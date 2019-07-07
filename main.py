import os
import platform
import torch
import argparse
import numpy as np
import torch.optim as optim
from model import Actor, Critic
from utils.utils import to_tensor, get_action, save_checkpoint
from collections import deque
from utils.running_state import ZFilter
from utils.memory import Memory
from agent.ppo import process_memory, train_model
from env.miniDota import miniDotaEnv
import matplotlib.pyplot as plt
import sys

parser = argparse.ArgumentParser(description='Setting for unity walker agent')
parser.add_argument('--render', default=False, action='store_true',
                    help='if you dont want to render, set this to False')
parser.add_argument('--train', default=False, action='store_true',
                    help='if you dont want to train, set this to False')
parser.add_argument('--load_model', type=str, default=None)
parser.add_argument('--gamma', type=float, default=0.995, help='discount factor')
parser.add_argument('--lamda', type=float, default=0.95, help='GAE hyper-parameter')
parser.add_argument('--hidden_size', type=int, default=512,
                    help='hidden unit size of actor and critic networks')
parser.add_argument('--critic_lr', type=float, default=0.0003)
parser.add_argument('--actor_lr', type=float, default=0.0003)
parser.add_argument('--batch_size', type=int, default=64)
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
parser.add_argument('--logdir', type=str, default='logs',
                    help='tensorboardx logs directory')
parser.add_argument('--env', type=str, default='plane',
                    help='environment, plane or curved')
parser.add_argument('--actionType', type=str, default='discrete',
                    help='The action is either discrete or continuous.')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def draw(record, baseX, baseY):
    if os.name == 'nt':
        record = np.concatenate(record)
        fig, ax = plt.subplots(figsize=(6,5))
        x, y = record[:, 0].tolist(), record[:, 1].tolist()
        stop, attack = record[:, 3].tolist(), record[:, 4].tolist()
        ax.plot(x, y, '-')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_xlim(-1, 51)
        ax.set_ylim(-1, 51)
        ax.plot(baseX, baseY, '*r', markersize=10)
        attX, attY = [], []
        stopX, stopY = [], []
        for idx, a in enumerate(attack):
            if a == 1:
                attX.append(x[idx])
                attY.append(y[idx])
            if stop[idx] == 1:
                stopX.append(x[idx])
                stopY.append(y[idx])
        ax.plot(attX, attY, 'x', color='chocolate')
        ax.plot(stopX, stopY, 'o', markersize=3, color='forestgreen')
        plt.show()

def check(var):
    print(var)
    i = input()
    if i == 'q':
        sys.exit()

if __name__ == "__main__":
    train_mode = args.train
    torch.manual_seed(500)

    env = miniDotaEnv(args)
    env_info = env.reset()

    num_inputs = 3
    num_actions = 5
    num_agent = 1 
        # better use a lot of agents.
        # multiple agents are running synchronously.

    print('state size:', num_inputs)
    print('action size:', num_actions)
    print('agent count:', num_agent)
    
    # running average of state
    running_state = ZFilter((num_agent,num_inputs), clip=5)

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

    states = running_state(env_info['observations'])
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
            # memory is cleared at every iter so only the current iter's sample are used in RL.

        steps = 0
        score = 0

        record = []
        while steps < args.time_horizon: # loop for one round of game.
            steps += 1
            if args.actionType == 'continuous':
                mu, std, _ = actor(to_tensor(states)) # get action probability from the actor network.
                actions = get_action(mu, std, args.actionType) # get certain action from probability.
            elif args.actionType == 'discrete':
                mu = actor(to_tensor(states))
                actions = get_action(mu, None, args.actionType)

            env_info = env.step(actions) # environment runs one step given the action.
            next_states = running_state(env_info['observations']) # get the next state.
            if (len(scores)+1) % 10 == 0:
                actionPart = np.zeros((actions.shape[0], 2))
                actionPart[:, 0] = actions[:, 0]
                actionPart[:, 1] = actions[:, 4]
                record.append(np.concatenate([env_info['observations'], actionPart], axis=1))
            rewards = env_info['rewards']
            dones = env_info['local_done']

            masks = list(~(np.array(dones)))

            for i in range(num_agent):
                memory[i].push(states[i], actions[i], rewards[i], masks[i])

            score += rewards[0]
            states = next_states

            if dones[0]:
                scores.append(score)
                score = 0
                episodes = len(scores)
                if episodes % 10 == 0:
                    score_avg = np.mean(scores[-min(10, episodes):])
                    print('{}-th episode : last 10 episode mean score of 1st agent is {:.2f}'.format(
                        episodes, score_avg))
                    draw(record, 30, 30)
                env.reset()
                break

        actor.train(), critic.train()

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
            # training is based on the state-action pairs from one iteration.

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
