import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


def combined_shape(length, shape=None): # length = size(1e6), shape = obs_dim(17)
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()]) # p.shape = 출력뉴런수, 입력뉴런수


LOG_STD_MAX = 2
LOG_STD_MIN = -20

class SquashedGaussianMLPActor(nn.Module):
    """목적 : obs 받아서 continous action 샘플링하는 함수"""

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation) # (17, 256) ReLU (256, 256) ReLU
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim) # 256, 6
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim) # 256, 6
        self.act_limit = act_limit # 1.0

    def forward(self, obs, deterministic=False, with_logprob=True): #  a, _ = self.pi(obs, deterministic, False)
        # type(obs) : <class 'torch.Tensor'>
        net_out = self.net(obs)
        # type(net_out) : <class 'torch.Tensor'>
        # net_out.shape : torch.Size([256])
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX) # torch.Size([6]) # 로그값이 너무 클 수 있어서
        std = torch.exp(log_std) # 표준편차는 항상 양수여야해서 exponential로 한번 더 감싼다

        pi_distribution = Normal(mu, std) # stochastic policy
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu # mu : action (행동을 했을 때 얻는 리워드가 가장 높은 액션)
        else:
            """
            sample하는 이유가 sac에서는 train때 stochastic policy에서 action을 샘플링하는데
            action을 그냥 정규분포에서 샘플링해서 쓰면, policy 파라미터를 업데이트할때, policy도 바뀌는데 우리는 액션을 샘플링해서 고정해두고, Q(s,a)에 대한 gradient는 반영하지않아서 정확하게 gradient가 계산되지 않아 문제가 된다
            -> policy에서 action을 직접 샘플링하는 대신, 미분가능한 함수로 action을 생성하자
               action을 정규분포에서 샘플링을 하면, mu, sigma로 미분할 수 없다.(즉, backprop이 되지 않음)
               정규분포를 표준정규분포 확률변수로 표현을 바꾸자
               rsample을 통해, 원래 정규분포에서 샘플링한 액션을 표준정규분포에서 샘플링한 입실론과 기존 mu, std를 이용해 다시 표현해서 사용함
            """
            pi_action = pi_distribution.rsample() # reparameterization trick 적용한 샘플링

        if with_logprob:
            """
            log_prob : torch.distributions.Normal 클래스에 정의된 메서드
            logp_pi : policy loss 계산할 때, Q function 타겟 계산 시 필요
            """
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1) # 스칼라
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1) # openai spinningup point 2
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action) # 액션이 정규분포에서 샘플링돼서 범위가 무한대임 -> 액션을 -1~1로 squashing하고 (왜냐, mujoco 환경 액션 범위 -1, 1)
        pi_action = self.act_limit * pi_action # 범위가 (-1, 1)보다 더 넓은 경우를 대비

        return pi_action, logp_pi # torch.Size([6])


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation) # q함수니까 (s,a)입력으로 주고, 마지막은 단일 스칼라값

    def forward(self, obs, act):
        """
        (torch.cat([obs, act], dim=-1) : torch.Size([100, 23])
        """
        q = self.q(torch.cat([obs, act], dim=-1)) # torch.Size([100, 1])
        return torch.squeeze(q, -1) # 마지막 차원이 1일경우 제거 : torch.Size([100])

class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0] # 17
        act_dim = action_space.shape[0] # 6 
        act_limit = action_space.high[0] # 1.0

        self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, deterministic=False):
        with torch.no_grad(): # inference만(gradient계산 no) : train이 아니라 행동 선택용
            a, _ = self.pi(obs, deterministic, False) # torch.Size([6])
            return a.numpy() # policy는 torch.Tensor, gym은 numpy를 쓰기때문에 numpy로 변환
