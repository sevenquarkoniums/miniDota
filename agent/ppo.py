import numpy as np
import torch
from utils.utils import to_tensor, to_tensor_long, log_density


def getGA(rewards, masks, values, args):
    # get the returns and advantages for one episode.
    returns = torch.zeros_like(rewards)
    advants = torch.zeros_like(rewards)

    running_returns = 0
    previous_value = 0
    running_advants = 0

    for t in reversed(range(0, len(rewards))):
        running_returns = rewards[t] + args.gamma * running_returns * masks[t]
        running_tderror = rewards[t] + args.gamma * previous_value * masks[t] - values.data[t]
        running_advants = running_tderror + args.gamma * args.lamda * running_advants * masks[t]
            # exactly the formula in the PPO paper.

        returns[t] = running_returns
        previous_value = values.data[t]
        advants[t] = running_advants

    advants = (advants - advants.mean()) / (advants.std() + 1e-15) # this normalization is common practice.
    return returns, advants


def surrogate_loss(net, advants, states, old_policy, actions, index, args):
    # calculate r*A.
    policyDistr = net(states)
    values = policyDistr[0]
    new_policy = log_density(actions, policyDistr)
    old_policy = old_policy[index]

    ratio = torch.exp(new_policy - old_policy)
    surrogate = ratio * advants
    return surrogate, ratio, values


def process_memory(net, batch, args):
    states = to_tensor(batch.state, False)
    actions = to_tensor(batch.action, False)
    rewards = to_tensor(batch.reward, False)
    masks = to_tensor(batch.mask, False)

    netOutput = net(states)# (value, action, moveX, moveY, target)
    values = netOutput[0]

    old_policy = log_density(actions, netOutput)
    old_values = values.clone()
    returns, advants = getGA(rewards, masks, values, args)

    return states, actions, returns, advants, old_policy, old_values


def train_model(net, optimizer, states, actions,
                returns, advants, old_policy, old_values, args):
    criterion = torch.nn.MSELoss()
    n = len(states)
    arr = np.arange(n)
    record = []

    for epoch in range(3):# iterations of training on these data.
        np.random.shuffle(arr)

        for i in range(n // args.batch_size):
            batch_index = arr[args.batch_size * i: args.batch_size * (i + 1)]
            batch_index = to_tensor_long(batch_index, False)
            inputs = states[batch_index]
            returns_samples = returns.unsqueeze(1)[batch_index]
            advants_samples = advants.unsqueeze(1)[batch_index]
            actions_samples = actions[batch_index]
            oldvalue_samples = old_values[batch_index].detach()

            vanillaLoss, ratio, values = surrogate_loss(net, advants_samples, inputs,
                                         old_policy.detach(), actions_samples,
                                         batch_index, args)

            clipped_values = oldvalue_samples + \
                             torch.clamp(values - oldvalue_samples,
                                         -args.clip_param,
                                         +args.clip_param)
            critic_loss1 = criterion(clipped_values, returns_samples)
            critic_loss2 = criterion(values, returns_samples)
            critic_loss = torch.max(critic_loss1, critic_loss2).mean()

            clipped_ratio = torch.clamp(ratio,
                                        1.0 - args.clip_param,
                                        1.0 + args.clip_param)
            clipped_loss = clipped_ratio * advants_samples
            actor_loss = -torch.min(vanillaLoss, clipped_loss).mean()

            loss = actor_loss + 0.5 * critic_loss
                # 0.5 is an adjustable coefficient.
            record.append(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    print('Avg loss: %f' % (sum(record) / len(record)))
