from typing import List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from torch.distributions import Independent, Normal

from fsrl.agent import OnpolicyAgent
from fsrl.policy import PPOLagrangian
from fsrl.utils import BaseLogger
from fsrl.utils.exp_util import auto_name, seed_all
from fsrl.utils.net.common import ActorCritic
from .multi_category_distribution import MultiCategoryDistribution
from torch import nn
class DiscretePPOAgent(OnpolicyAgent):
    def __init__(
        self,
        env: gym.Env,
        actor = None,
        logger: BaseLogger = BaseLogger(),
        cost_limit: float = 10,
        device: str = "cpu",
        thread: int = 4,  # if use "cpu" to train
        seed: int = 10,
        lr: float = 5e-4,
        hidden_sizes: Tuple[int, ...] = (128, 128),
        unbounded: bool = False,
        last_layer_scale: bool = False,
        # PPO specific arguments
        target_kl: float = 0.02,
        vf_coef: float = 0.25,
        max_grad_norm: Optional[float] = None,
        gae_lambda: float = 0.95,
        eps_clip: float = 0.2,
        dual_clip: Optional[float] = None,
        value_clip: bool = False,
        advantage_normalization: bool = True,
        recompute_advantage: bool = False,
        # Lagrangian specific arguments
        use_lagrangian: bool = True,
        lagrangian_pid: Tuple = (0.05, 0.0005, 0.1),
        rescaling: bool = True,
        # Base policy common arguments
        gamma: float = 0.99,
        max_batchsize: int = 99999,
        reward_normalization: bool = False,  # can decrease final perf
        deterministic_eval: bool = True,
        action_scaling: bool = True,
        action_bound_method: str = "clip",
        lr_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
        use_safetyloss: bool = True,

    ) -> None:
        super().__init__()
        self.logger = logger
        self.cost_limit = cost_limit

        if np.isscalar(cost_limit):
            cost_dim = 1
        else:
            cost_dim = len(cost_limit)

        # set seed and computing
        seed_all(seed)
        torch.set_num_threads(thread)

        # model
        state_shape = env.observation_space.shape or env.observation_space.n
        # action_shape = env.action_space.shape or env.action_space.n
        # max_action = env.action_space.high[0]

        # net = Net(state_shape, hidden_sizes=hidden_sizes, device=device)
        
            # actor = ActorProb(
            #     net, action_shape, max_action=max_action, unbounded=unbounded, device=device
            # ).to(device) # TODO Fix Here 
        
        critic = [
            Critic(
                Net(state_shape, hidden_sizes=hidden_sizes, device=device),
                device=device
            ).to(device) for _ in range(1 + cost_dim)
        ] 

        # torch.nn.init.constant_(actor.sigma_param, -0.5)
        actor_critic = ActorCritic(actor, critic)
        # orthogonal initialization
        for m in actor_critic.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight)
                torch.nn.init.zeros_(m.bias)
        if last_layer_scale:
            # do last policy layer scaling, this will make initial actions have (close
            # to) 0 mean and std, and will help boost performances, see
            # https://arxiv.org/abs/2006.05990, Fig.24 for details
            for m in actor.mu.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.zeros_(m.bias)
                    m.weight.data.copy_(0.01 * m.weight.data)
        optim = torch.optim.Adam(actor_critic.parameters(), lr=lr)

        # replace DiagGuassian with Independent(Normal) which is equivalent pass *logits
        # to be consistent with policy.forward
        
        # TODO Fix here 
        def dist(*logits):
            return MultiCategoryDistribution(*logits) #Independent(Normal(*logits), 1)

        self.policy = PPOLagrangianCustom(
            actor,
            critic,
            optim,
            dist,
            logger=logger,
            # PPO specific arguments
            target_kl=target_kl,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            gae_lambda=gae_lambda,
            eps_clip=eps_clip,
            dual_clip=dual_clip,
            value_clip=value_clip,
            advantage_normalization=advantage_normalization,
            recompute_advantage=recompute_advantage,
            # Lagrangian specific arguments
            use_lagrangian=use_lagrangian,
            lagrangian_pid=lagrangian_pid,
            cost_limit=cost_limit,
            rescaling=rescaling,
            # Base policy common arguments
            gamma=gamma,
            max_batchsize=max_batchsize,
            reward_normalization=reward_normalization,
            deterministic_eval=deterministic_eval,
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            observation_space=env.observation_space,
            action_space=env.action_space,
            lr_scheduler=lr_scheduler,
            use_safetyloss=use_safetyloss,
        )
        
from tianshou.data import Batch
from typing import Type

class PPOLagrangianCustom(PPOLagrangian):
    def __init__(
        self,
        actor: nn.Module,
        critic: Union[nn.Module, List[nn.Module]],
        optim: torch.optim.Optimizer,
        dist_fn: Type[torch.distributions.Distribution],
        logger: BaseLogger = BaseLogger(),
        # PPO specific argumentsß
        target_kl: float = 0.02,
        vf_coef: float = 0.25,
        max_grad_norm: Optional[float] = None,
        gae_lambda: float = 0.95,
        eps_clip: float = 0.2,
        dual_clip: Optional[float] = None,
        value_clip: bool = False,
        advantage_normalization: bool = True,
        recompute_advantage: bool = False,
        # Lagrangian specific arguments
        use_lagrangian: bool = True,
        lagrangian_pid: Tuple = (0.05, 0.0005, 0.1),
        cost_limit: Union[List, float] = np.inf,
        rescaling: bool = True,
        # Base policy common arguments
        gamma: float = 0.99,
        max_batchsize: int = 99999,
        reward_normalization: bool = False,
        deterministic_eval: bool = True,
        action_scaling: bool = True,
        action_bound_method: str = "clip",
        observation_space: Optional[gym.Space] = None,
        action_space: Optional[gym.Space] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
        use_safetyloss: bool = True
    ) -> None:
        super().__init__(
            actor,
            critic,
            optim,
            dist_fn,
            logger=logger,
            # PPO specific arguments
            target_kl=target_kl,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            gae_lambda=gae_lambda,
            eps_clip=eps_clip,
            dual_clip=dual_clip,
            value_clip=value_clip,
            advantage_normalization=advantage_normalization,
            recompute_advantage=recompute_advantage,
            # Lagrangian specific arguments
            use_lagrangian=use_lagrangian,
            lagrangian_pid=lagrangian_pid,
            cost_limit=cost_limit,
            rescaling=rescaling,
            # Base policy common arguments
            gamma=gamma,
            max_batchsize=max_batchsize,
            reward_normalization=reward_normalization,
            deterministic_eval=deterministic_eval,
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            observation_space=observation_space,
            action_space=action_space,
            lr_scheduler=lr_scheduler,
        )
        self.use_safetyloss = use_safetyloss

    
    def policy_loss(self, batch: Batch, dist: Type[torch.distributions.Distribution]):
        log_p = dist.log_prob(batch.act)
        ratio = (log_p - batch.logp_old).exp().float()
        ratio = ratio.reshape(ratio.size(0), -1).transpose(0, 1)
        if self._norm_adv:
            for i in range(self.critics_num):
                adv = batch.advs[..., i]
                mean, std = adv.mean(), adv.std()
                batch.advs[..., i] = (adv - mean) / std  # per-batch norm

        # compute normal ppo loss
        rew_adv = batch.advs[..., 0]
        surr1 = ratio * rew_adv
        surr2 = ratio.clamp(1.0 - self._eps_clip, 1.0 + self._eps_clip) * rew_adv
        if self._dual_clip:
            clip1 = torch.min(surr1, surr2)
            clip2 = torch.max(clip1, self._dual_clip * rew_adv)
            loss_actor_rew = -torch.where(rew_adv < 0, clip2, clip1).mean()
        else:
            loss_actor_rew = -torch.min(surr1, surr2).mean()

        # compute safety loss
        values = [ratio * batch.advs[..., i]
                    for i in range(1, self.critics_num)] if self.use_lagrangian else []
        loss_actor_safety, stats_actor = self.safety_loss(values)
        rescaling = stats_actor["loss/rescaling"]
        if self.use_safetyloss:
            loss_actor_total = rescaling * (loss_actor_rew + loss_actor_safety)
        else:
            loss_actor_total = rescaling * (loss_actor_rew)
        # compute approx KL for early stop
        approx_kl = (batch.logp_old - log_p).mean().item()
        stats_actor.update(
            {
                "loss/actor_rew": loss_actor_rew.item(),
                "loss/actor_total": loss_actor_total.item(),
                "loss/kl": approx_kl
            }
        )
        return loss_actor_total, stats_actor