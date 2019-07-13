import torch
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
    mapp = [[-0.0786036 ,  2.04118482, -0.34092627, -1.1541647 ,  1.4151794 ,
        -1.49332902,  0.64944531,  1.97610011,  1.01910371,  0.70660314,
         0.37868015, -0.5828654 ],
       [-1.61199401,  0.76541484,  0.44583175,  0.75570891, -1.08256865,
        -0.20874901,  1.34263652,  0.24429005, -0.12382564, -1.12410592,
         0.04655836,  0.72922137],
       [ 0.76324432,  1.7580071 , -0.47733487, -0.46430275, -0.45795684,
         0.31646925,  0.45397791, -0.720694  , -0.49957321,  1.07954844,
         1.56330428, -0.13302661],
       [ 0.09243752,  1.09676539, -1.44002539, -0.08148459,  0.33319172,
         0.31405757,  0.36865051, -0.56732647, -1.56073957,  0.51343353,
         1.09089616,  2.25183221],
       [-0.49863823,  1.53535017, -0.77509604,  0.74025913, -0.61070429,
         1.33790085,  0.37283497,  0.56420502, -1.73101124,  0.05505819,
         1.47636104, -1.56723328],
       [ 2.07296143, -0.23785035, -0.70502317, -0.64352587,  0.85735674,
         1.0471403 , -0.61102597,  1.11010261, -1.30341356, -1.81265384,
         0.1636475 , -0.43450309],
       [ 0.43609545,  0.6963067 , -0.09364687, -0.48646577,  0.32345968,
        -0.01456758, -2.15908548,  1.43040459,  1.08257079, -1.50394687,
        -0.78737189, -0.59844352],
       [ 0.25367141,  0.04817729,  0.98881724, -0.44335864, -0.59901669,
         0.5427117 , -0.46790898,  1.1415493 ,  0.19946521, -0.23042303,
         1.06946268, -0.81884672],
       [-0.35494505, -1.05158486,  1.35032517, -0.58202717, -0.28541758,
         0.100301  ,  0.33451816, -1.08056776,  1.35561838, -0.14093178,
         0.39951382,  0.04217748],
       [-0.23618642,  0.64789286, -0.41538226, -0.64577092, -0.78634422,
         1.42113727,  1.42708234, -0.06039009, -0.04511963,  1.66252199,
         0.30943731,  2.20539777],
       [ 1.06666174,  0.36374009,  1.42771073, -1.04206682, -1.11317026,
         0.84700292, -0.95290389,  0.86676491, -1.34649261, -0.49102252,
        -0.30762332, -1.20021593],
       [ 0.69714513,  0.57098163, -0.71465807,  0.27457825,  0.84032202,
        -0.75805944,  0.54412346,  0.95308094, -1.14018971,  0.73420034,
         0.16497484, -1.39858277]]
        # from np.random.randn(12, 12).
    onehot = get_one_hot(agent, 12)
    embed = np.matmul(onehot, mapp)
    return embed # 12-unit row vector.

def to_tensor_long(numpy_array, cpuSimulation):
    if torch.cuda.is_available() and not cpuSimulation:
        variable = torch.LongTensor(numpy_array).cuda()
    else:
        variable = torch.LongTensor(numpy_array).cpu()
    return variable


def to_tensor(numpy_array, cpuSimulation):
    if torch.cuda.is_available() and not cpuSimulation:
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
    action = torch.nn.functional.one_hot(actions[:,0].to(torch.int64), 3).to(torch.float)
    moveX = torch.nn.functional.one_hot(actions[:,1].to(torch.int64), 3).to(torch.float)
    moveY = torch.nn.functional.one_hot(actions[:,2].to(torch.int64), 3).to(torch.float)
    target = torch.nn.functional.one_hot(actions[:,3].to(torch.int64), 12).to(torch.float)
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


