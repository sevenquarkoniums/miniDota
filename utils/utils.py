import torch
import math
import numpy as np
import sys

def check(var):
    print(var)
    try:
        i = input()
        if i == 'q':
            sys.exit()
    except KeyboardInterrupt:
        sys.exit()

def get_one_hot(target, nb_classes):
    res = np.eye(nb_classes)[np.array(target)]
    return res

def unitEmbed(agent):
    mapp = [[ 0.8789365 , -1.46669075, -0.8060484 , -0.00359309,  0.68484395,
         0.60168591,  0.83205078,  0.25865925,  1.71127154, -0.31694766],
       [ 1.28788371,  1.62049922,  1.65614445,  0.8218171 ,  0.23978736,
        -1.53601262,  0.82928828,  0.33234628,  0.05079679,  0.89984695],
       [-0.94398478, -0.63923543,  0.29805187,  0.09317771,  0.99368787,
        -1.05850182, -2.0664041 , -0.33478335,  0.2436032 ,  2.23036506],
       [-0.56912503, -0.02379822, -0.79355072,  0.13847508,  0.6246829 ,
        -1.29298248, -0.72513038,  0.22262808, -2.60193517,  0.39712555],
       [ 0.31502459, -0.00715841, -1.05666031,  0.51339775,  0.10108334,
        -0.18786572,  0.92214912,  1.87581704,  0.44596211,  0.32171026],
       [-1.05037746, -2.12272089,  0.68360326, -0.20849123, -0.56509515,
         1.53543781,  0.9761528 ,  0.86647268,  0.17537769, -0.22335242],
       [-0.74091603, -0.96443297,  2.61791442,  0.69626782, -1.12041336,
         1.53308149, -2.47671792,  0.78121506,  1.47757334,  0.20595855],
       [-0.65935648,  0.17964917, -2.25858313, -1.41300853, -0.91586181,
        -0.23451315,  0.37269398, -1.90074103,  1.00578038,  0.17041562],
       [-1.25371377, -1.16508746, -1.69654309, -1.43783792,  0.63299341,
        -1.662901  , -0.40147154,  0.00694847, -0.40408588,  0.10202011],
       [ 0.4701976 ,  0.77524024,  0.18218596,  1.33997103,  0.00979764,
         1.10949467,  1.21646542,  1.34010374,  1.80792296,  0.87354297]]
        # from np.random.randn(10, 10).
    onehot = get_one_hot(agent, 10)
    embed = np.matmul(onehot, mapp)
    return embed # 10-unit row vector.

def to_tensor_long(numpy_array):
    if torch.cuda.is_available():
        variable = torch.LongTensor(numpy_array).cuda()
    else:
        variable = torch.LongTensor(numpy_array).cpu()
    return variable


def to_tensor(numpy_array):
    if torch.cuda.is_available():
        variable = torch.Tensor(numpy_array).cuda()
    else:
        variable = torch.Tensor(numpy_array).cpu()
    return variable


def get_action(netOutput):
    action = torch.multinomial(netOutput[1], 1).cpu().data.numpy()
    moveX = torch.multinomial(netOutput[2], 1).cpu().data.numpy()
    moveY = torch.multinomial(netOutput[3], 1).cpu().data.numpy()
    target = torch.multinomial(netOutput[4], 1).cpu().data.numpy()
    return np.concatenate([action, moveX, moveY, target], axis=1)


def log_density(actions, policyDistr):
    # returns the log probability of actions ~ multinomial(policyDistr).
    # actions: a 2d tensor with 4 cols.
    # policyDistr: a 4-tuple, each is a 2d tensor.
    action = torch.nn.functional.one_hot(actions[:,0], 3)
    moveX = torch.nn.functional.one_hot(actions[:,1], 3)
    moveY = torch.nn.functional.one_hot(actions[:,2], 3)
    target = torch.nn.functional.one_hot(actions[:,3], 12)
    prob = torch.sum(action * policyDistr[1], dim=1, keepdim=True) * \
           torch.sum(moveX  * policyDistr[2], dim=1, keepdim=True) * \
           torch.sum(moveY  * policyDistr[3], dim=1, keepdim=True) * \
           torch.sum(target * policyDistr[4], dim=1, keepdim=True)
    log_density = torch.log(prob)
    return log_density


def flat_grad(grads):
    grad_flatten = []
    for grad in grads:
        grad_flatten.append(grad.view(-1))
    grad_flatten = torch.cat(grad_flatten)
    return grad_flatten


def flat_hessian(hessians):
    hessians_flatten = []
    for hessian in hessians:
        hessians_flatten.append(hessian.contiguous().view(-1))
    hessians_flatten = torch.cat(hessians_flatten).data
    return hessians_flatten


def flat_params(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))
    params_flatten = torch.cat(params)
    return params_flatten


def update_model(model, new_params):
    index = 0
    for params in model.parameters():
        params_length = len(params.view(-1))
        new_param = new_params[index: index + params_length]
        new_param = new_param.view(params.size())
        params.data.copy_(new_param)
        index += params_length


def kl_divergence(new_actor, old_actor, states):
    # not used.
    mu, std, logstd = new_actor(torch.Tensor(states))
    mu_old, std_old, logstd_old = old_actor(torch.Tensor(states))
    mu_old = mu_old.detach()
    std_old = std_old.detach()
    logstd_old = logstd_old.detach()

    # kl divergence between old policy and new policy : D( pi_old || pi_new )
    # pi_old -> mu0, logstd0, std0 / pi_new -> mu, logstd, std
    # be careful of calculating KL-divergence. It is not symmetric metric
    kl = logstd_old - logstd + (std_old.pow(2) + (mu_old - mu).pow(2)) / \
         (2.0 * std.pow(2)) - 0.5
    return kl.sum(1, keepdim=True)


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


