import torch as th
import numpy as np


def build_td_lambda_targets(rewards, terminated, mask, target_qs, n_agents, gamma, td_lambda):
    # Assumes  <target_qs > in B*T*A and <reward >, <terminated >, <mask > in (at least) B*T-1*1
    # Initialise  last  lambda -return  for  not  terminated  episodes
    ret = target_qs.new_zeros(*target_qs.shape)
    ret[:, -1] = target_qs[:, -1] * (1 - th.sum(terminated, dim=1))
    # Backwards  recursive  update  of the "forward  view"
    for t in range(ret.shape[1] - 2, -1,  -1):
        ret[:, t] = td_lambda * gamma * ret[:, t + 1] + mask[:, t] \
                    * (rewards[:, t] + (1 - td_lambda) * gamma * target_qs[:, t + 1] * (1 - terminated[:, t]))
    # Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
    return ret[:, 0:-1]

def build_bootstrap_targets(rewards, terminated, mask, target_qs, n_agents, gamma):
    ret = target_qs.new_zeros(*target_qs.shape)
    ret[:, -1] = target_qs[:, -1] * (1 - th.sum(terminated, dim=1))
    for t in range(ret.shape[1] - 2, -1, -1):
        ret[:, t] = (gamma * ret[:, t + 1] * (1 - terminated[:, t])) + rewards[:, t] * mask[:, t]
    # Returns bootstrap-return from t=0 to t=T-1
    return ret[:, :-1]

def categorical_entropy(logits):
    a0 = logits - th.max(logits, dim=-1, keepdim=True)[0]
    ea0 = th.exp(a0)
    z0 = th.sum(ea0, dim=-1, keepdim=True)
    p0 = ea0 / z0
    return th.sum(p0 * (th.log(z0) - a0), dim=-1)


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r
        r = r*(1.-done)
        discounted.append(r)
    return discounted[::-1]


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


# def build_td_lambda_targets_v1(rewards, terminated, mask, target_qs, gamma, td_lambda):
#     # Assumes  <target_qs > <reward >, <terminated >, <mask > in (at least) B*T-1*1
#     # Initialise  last  lambda -return  for  not  terminated  episodes
#     ret = target_qs.new_zeros(*target_qs.shape)
#     ret[:, -1] = target_qs[:, -1] * (1 - th.sum(terminated, dim=1))
#     # Backwards  recursive  update  of the "forward  view"
#     for t in range(ret.shape[1] - 2, -1,  -1):
#         ret[:, t] = td_lambda * gamma * ret[:, t + 1] + mask[:, t] \
#                     * (rewards[:, t] + (1 - td_lambda) * gamma * target_qs[:, t + 1] * (1 - terminated[:, t]))
#     # Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*1
#     return ret
#
# def bootstrap_value(rewards, terminated, mask, target_qs, gamma):
#     # Assumes  <target_qs > <reward >, <terminated >, <mask > in (at least) B*T-1*1
#     # Initialise  last  lambda -return  for  not  terminated  episodes
#     ret = target_qs.new_zeros(*target_qs.shape)
#     ret[:, -1] = target_qs[:, -1] * (1 - th.sum(terminated, dim=1))
#     # Backwards  recursive  update  of the "forward  view"
#     for t in range(ret.shape[1] - 2, -1, -1):
#         ret[:, t] = rewards[:, t] + gamma * target_qs[:, t + 1] * (1 - terminated[:, t])
#     # Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*1
#     return ret
def denormalize(x, stats):
    if stats is None:
        return x
    return x * stats.std + stats.mean


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count
