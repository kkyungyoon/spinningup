import sys
sys.path.append("/root/spinningup/")

from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import spinup.algos.pytorch.sac.core as core
from spinup.utils.logx import EpochLogger


class ReplayBuffer:

    def __init__(self, obs_dim, act_dim, size): # 17, 6, int(1e6)
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size # ptr : transition을 저장할 index를 가리키는 포인터

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size # FIFO 구조 구현
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size) # 랜덤 인덱스 샘플링
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}
        """
        k : 'obs', 'obs2', 'act', 'rew', 'done'
        v : (batch_size, obs_dim), (batch_size, obs_dim), (batch_size, act_dim), (batch_size,), (batch_size,)
        배치를 이용해서, Q function 업데이트, policy 업데이트 할거니까 torch.Tensor로 변환
        """



def sac(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000, 
        update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000, 
        logger_kwargs=dict(), save_freq=1):
    """
    env_fn : 환경 생성하는 함수
    actor_critic : actor-critic 모델 클래스
    ac_kwargs : actor-critic에 전달할 키워드 인자들
    steps_per_epoch : 1 epoch마다 몇 step씩(1 에폭당 얼마나 많은 transition을 수집할지)
    epochs : 총 에폭수
    replay_size : replay buffer 최대 크기
    gamma : Q loss 계산 시, 타겟 계산시 사용되는 discount factor
    polyak : polyak averaging으로 타겟 네트워크를 부드럽게 업데이트할 때 사용되는 polyak averaging coefficient
    lr : optimizer에서 사용되는 학습률. 얼만큼 파라미터를 업데이트할지
    alpha : Q loss 계산 시, 타겟 계산시 policy entropy를 조절하는 temperature parameter
    batch_size : 학습 시 샘플링할 미니배치 크기
    start_steps : 초반에 수행하는 랜덤액션을 언제까지 쓸지(이후에는 policy에서 샘플링해서 씀)
    update_after : 몇 스텝 이후부터 학습 시작할지
    update_every : 몇 step마다 학습 업데이트할지
    num_test_episodes : 각 에폭마다 테스트 에피소드 수
    max_ep_len : 한 에피소드에서 최대 스텝 수
    logger_kwargs : 로그 저장 경로 등 정보
    save_freq : 몇 에폭마다 모델 저장할지
    """

    logger = EpochLogger(**logger_kwargs) # logger_kwargs : {'output_dir': '/root/spinningup/data/sac/sac_s0', 'exp_name': 'sac'}
    logger.save_config(locals()) # locals()를 dict로 넘김(locals().keys())

    torch.manual_seed(seed) # torch seed 고정
    np.random.seed(seed)    # numpy seed 고정

    env, test_env = env_fn(), env_fn() # 환경은 reset(), step()이 호출될 때마다 내부 state가 바뀜 : 독립된 환경이 안정성을 보장해줌(내부 상태 및 시드가 독립적)
    
    # env.observation_space : Box(17,)
    # env.action_space : Box(6,)
    # env.spec : EnvSpec(HalfCheetah-v2)
    # env.action_space.sample() : [ 0.77627426 -0.24766919 -0.55545646 -0.7230827   0.9719488  -0.9437829 ]
    obs_dim = env.observation_space.shape # (17,)
    act_dim = env.action_space.shape[0] # 6

    act_limit = env.action_space.high[0] # 1.0

    # 초기 SAC에서는 value network가 존재하지만, 실제 구현에서는 value network 생략
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac_targ = deepcopy(ac) # Q loss 계산 시 타겟 계산을 위해서

    for p in ac_targ.parameters(): # 레이어마다 weight, bias 출력가능
        p.requires_grad = False # 타겟 계산할 때 사용해야하므로, optimizer에 의해 학습되지않도록
        
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters()) # 동시에 하나의 optimizer로 학습하고 싶을 때, 두 네트워크의 파라미터를 하나로 묶음

    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size) # 17, 6, int(1e6)

    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)

    def compute_loss_q(data): # data = batch
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = ac.q1(o,a) # torch.Size([100])
        q2 = ac.q2(o,a) # torch.Size([100])

        # 타겟 계산 시, gradient가 흐르지 않게 함 (타겟은 업데이트가 되면 안 됨)
        with torch.no_grad():
            a2, logp_a2 = ac.pi(o2) # torch.Size([6]), torch.Size([100])

            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ) # two Q function
            backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2) # openai spinningup point 1 # logp_a2 : maximum entropy term

        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        q_info = dict(Q1Vals=q1.detach().numpy(), # detach없이 numpy하면 에러
                      Q2Vals=q2.detach().numpy())

        return loss_q, q_info

    def compute_loss_pi(data):
        """
        reparameterization trick을 적용해서 action을 구했기 때문에, gradient가 흐르게 된다.
        PyTorch에서는 autograd가 자동으로 처리
        Q. normal distribution에서 샘플링하면 gradient가 안 흐르는 이유
        """
        o = data['obs']
        pi, logp_pi = ac.pi(o)
        q1_pi = ac.q1(o, pi)
        q2_pi = ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi) # two Q function

        loss_pi = (alpha * logp_pi - q_pi).mean() # sac의 objective function # logp_pi : maximum entropy term

        pi_info = dict(LogPi=logp_pi.detach().numpy())

        return loss_pi, pi_info

    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    q_optimizer = Adam(q_params, lr=lr)

    logger.setup_pytorch_saver(ac) # pytorch_saver_elements 변수에 저장

    def update(data):
        """
        data = batch
        Q-network, policy-network를 번갈아가며 업데이트
        """
        q_optimizer.zero_grad() # 기존에 누적된 gradient 초기화
        loss_q, q_info = compute_loss_q(data) # Q-function loss 계산
        loss_q.backward() # gradient 계산
        q_optimizer.step() # 파라미터 업데이트

        logger.store(LossQ=loss_q.item(), **q_info)

        """
        policy를 업데이트할 때도 Q-network의 출력을 사용
        하지만 이때는 Q-network의 파라미터를 학습하려는 것이 아니라,
        policy 네트워크의 출력을 기반으로 logπ와 Q(s,a)를 계산해서 policy의 loss를 계산하고,
        gradient는 policy 네트워크 방향으로만 흐르게 하고싶기 때문에 Q-network 파라미터 freeze
        """
        for p in q_params:
            p.requires_grad = False

        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # 다음 번 Q-step에서 다시 학습 가능하도록 복구
        for p in q_params:
            p.requires_grad = True

        logger.store(LossPi=loss_pi.item(), **pi_info)

        # polyak averaging으로 타겟 네트워크 업데이트(안정적인 학습) # openai spinningup point 3
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # mul_, add_ : 새로운 텐서 만들지 않고 in-place 방식
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, deterministic=False):
        # type(o) : <class 'numpy.ndarray'>
        return ac.act(torch.as_tensor(o, dtype=torch.float32), 
                      deterministic)

    def test_agent():
        """
        train때 stochastic하게 하고, test때는 deterministic하게 하는게 보편적인지
        : 보편적으로 한다.(하지만 가장 좋은건 엔트로피를 갈수록 줄여가는것
        왜냐, stochastic의 mean이 좋은 deterministic policy라는 보장없음)
        """
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                o, r, d, _ = test_env.step(get_action(o, True)) # exploration이 없음(deterministic action)
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    total_steps = steps_per_epoch * epochs # 전체학습동안 에이전트가 환경에서 총 몇번 action을 취할지 = 총 interaction step 수
    start_time = time.time()
    """
    o : env.reset()을 해서 에피소드의 initial state를 가져옴
        o.shape : (17,)
        type(o) : <class 'numpy.ndarray'>
        : 환경(gym)은 numpy를 쓰고,
        : policy는 torch.Tensor를 사용
    ep_ret : 현재 에피소드의 리턴(초기값 0)
    ep_len : 현재 에피소드의 길이(초기값 0) (한 에피소드의 transition의 개수 = step의 개수)
    """
    o, ep_ret, ep_len = env.reset(), 0, 0

    for t in range(total_steps):
        
        if t > start_steps:
            a = get_action(o) # <class 'numpy.ndarray'>, 6
        else:
            a = env.action_space.sample() # start_steps가 되기 전까지는 경험만 수집하고 학습은 안함

        o2, r, d, _ = env.step(a) # 제일 끝에는 info(에이전트가 잘 달렸는지, 컨트롤이 얼마나 부드러웠는지) : {'reward_run': 0.7831539368432663, 'reward_ctrl': -0.2078265905380249})

        ep_ret += r # undiscounted return
        ep_len += 1

        """
        - max_ep_len의 존재이유 : 한 에피소드가 terminal state에 도착하지 않으면 무한 반복되고 길어질 수 있음
        - 환경에서는 ep_len==max_ep_len일 때도 done=True를 리턴함
          SAC같은 off policy 방법은 이전에 끝난 시점의 transition을 꺼내어 done=True면 bootstrapping 하지않음
          하지만 단순히 max 길이에만 도착했을 뿐, 유효한 state이므로 done=False 처리해서 next Q를 계속 써야함
        """ 
        d = False if ep_len==max_ep_len else d

        replay_buffer.store(o, a, r, o2, d)

        o = o2

        """
        새 에피소드로 넘어가기 위해 reset
        - d : terminal state에 도달, 에피소드 종료, Q value 계산에서 next state Q 무시 (진짜 종료된 경우)
        - ep_len == max_ep_len : 에피소드 종료, d=False로 강제수정, next state Q 계속 사용(학습 계속 이어나감) (환경은 에피소드가 끝났다고 판단하기 때문에, 새 에피소드로 넘어가기 위해 끝났다고 생각하고 reset()해줘야함)
        """
        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0

        """
        update_after : 얼마나 많은 transition을 쌓은 뒤부터 gradient descent를 실행할지 결정하는 값 : 1000 (초기에 업데이트하면 버퍼가 비어있어서 학습이 불안정)
        update_every : 얼마나 자주 gradient update 할지 : 50
        """
        if t >= update_after and t % update_every == 0:
            for j in range(update_every): # 총 학습횟수가 부족할까봐 트릭
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch)

        if (t+1) % steps_per_epoch == 0: # 1 에폭이 끝난 시점 = 4000 step 마다
            epoch = (t+1) // steps_per_epoch # 몇 번째 에폭인지

            if (epoch % save_freq == 0) or (epoch == epochs): # save_freq 주기 or 마지막 에폭일때 모델 저장
                logger.save_state({'env': env}, None)

            test_agent() # policy의 현재 성능을 측정하고, 로그로 기록하기 위해(모니터링 목적)

            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)     # 학습 에피소드 리턴값
            logger.log_tabular('TestEpRet', with_min_and_max=True) # 테스트 환경에서의 리턴값(exploration 없음)
            logger.log_tabular('EpLen', average_only=True)         # 에피소드 길이(한 에피소드에서 step 수)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)             # 전체 step 수
            logger.log_tabular('Q1Vals', with_min_and_max=True)    # Q function이 추정한 Q(s,a)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('LogPi', with_min_and_max=True)     # Policy entropy의 간접지표(작아질수록 policy가 deterministic해지고 있다는 뜻)
            logger.log_tabular('LossPi', average_only=True)        # Policy network의 loss
            logger.log_tabular('LossQ', average_only=True)         # Q-network 업데이트 시 MSE Loss
            logger.log_tabular('Time', time.time()-start_time)     # 경과시간
            logger.dump_tabular()

# class DebugEnv(gym.Wrapper):
#     def __init__(self, env, log_path='env_log.txt'):
#         super().__init__(env)
#         self.log_file = open(log_path, 'w')

#     def reset(self, **kwargs):
#         obs = self.env.reset(**kwargs)
#         self.log_file.write(f'reset() → obs: {obs}\n')
#         self.log_file.flush()
#         return obs

#     def step(self, action):
#         obs, reward, done, info = self.env.step(action)
#         self.log_file.write(f'step(action={action}) → obs: {obs}, reward: {reward}, done: {done}\n')
#         self.log_file.flush()
#         return obs, reward, done, info

#     def __del__(self):
#         self.log_file.close()


log_path = 'env_log.txt'

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99) # Q loss 계산 시, 타겟 계산시 사용되는 discount factor
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='sac')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    # logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, data_dir='/root/spinningup/data')

    torch.set_num_threads(torch.get_num_threads()) # 현재 설정된 스레드(프로그램 안에서 실행되는 작업 단위) 수 다시 그대로 설정(병렬 처리 세팅 초기화)

    sac(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic, # 지금 바로 환경을 만들지 않고, 나중에 만들기 위해 넘김
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs)
    
    # sac(lambda : DebugEnv(gym.make(args.env), log_path=log_path),
    #     actor_critic=core.MLPActorCritic,
    #     ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
    #     gamma=args.gamma, seed=args.seed, epochs=args.epochs,
    #     logger_kwargs=logger_kwargs)