import torch
import math


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


def get_action(mu, std, actionType):
    # get the specific action out of a probability distribution.
    # mu: 2d tensor.
    if actionType == 'continuous':
        action = torch.normal(mu, std)
    elif actionType == 'discrete':
        action = torch.distributions.bernoulli.Bernoulli(mu).sample()
    action = action.cpu().data.numpy()
    return action


def log_density(x, mu, std, logstd, args):
    # x: the actual action; 2d tensor.
    if args.actionType == 'continuous':
        # the probability of x ~ Norm(mu, std).
        var = std.pow(2)
        log_density = -(x - mu).pow(2) / (2 * var) \
                      - 0.5 * math.log(2 * math.pi) - logstd
    elif args.actionType == 'discrete':
        # the probability of x ~ Bernoulli(mu).
        log_density = x * mu + (1-x) * (1-mu)
    return log_density.sum(1, keepdim=True) # why sum?


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


