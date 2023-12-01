from typing import List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from torch.distributions import Independent, Normal

from fsrl.agent import OnpolicyAgent
# from fsrl.policy import PPOLagrangian
from fsrl.utils import BaseLogger
from fsrl.utils.exp_util import auto_name, seed_all
from fsrl.utils.net.common import ActorCritic
from fsrl.policy import CPO
from .multi_category_distribution import MultiCategoryDistribution
from torch import nn
class DiscreteCPOAgent(OnpolicyAgent):
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

        self.policy = CPO(
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
from torch.distributions import kl_divergence
class CPOCustom(CPO):
    def __init__(
         self,
        actor: nn.Module,
        critics: Union[nn.Module, List[nn.Module]],
        optim: torch.optim.Optimizer,
        dist_fn: Type[torch.distributions.Distribution],
        logger: BaseLogger = BaseLogger(),
        # CPO specific arguments
        target_kl: float = 0.01,
        backtrack_coeff: float = 0.8,
        damping_coeff: float = 0.1,
        max_backtracks: int = 10,
        optim_critic_iters: int = 20,
        l2_reg: float = 0.001,
        gae_lambda: float = 0.95,
        advantage_normalization: bool = True,
        cost_limit: Union[List, float] = np.inf,
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
            critics,
            optim,
            dist_fn,  # type: ignore
            logger=logger,
            # CPO specific arguments
            target_kl=target_kl,
            backtrack_coeff=backtrack_coeff,
            damping_coeff=damping_coeff,
            max_backtracks=max_backtracks,
            optim_critic_iters=optim_critic_iters,
            l2_reg=l2_reg,
            gae_lambda=gae_lambda,
            advantage_normalization=advantage_normalization,
            cost_limit=cost_limit,
            # Base policy common arguments
            gamma=gamma,
            max_batchsize=max_batchsize,
            reward_normalization=reward_normalization,
            deterministic_eval=deterministic_eval,
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            observation_space=observation_space,
            action_space=action_space,
            lr_scheduler=lr_scheduler
        )
        self.use_safetyloss = use_safetyloss

    
    def policy_loss(self, minibatch: Batch) -> Tuple[torch.Tensor, dict]:

        self.actor.train()
        # get objective & KL & cost surrogate
        dist = self.forward(minibatch).dist
        ent = dist.entropy().mean()
        logp = dist.log_prob(minibatch.act)

        dist_old = self.dist_fn(*(minibatch.mean_old, minibatch.std_old))  # type: ignore
        kl = kl_divergence(dist_old, dist).mean()

        objective = self._get_objective(logp, minibatch.logp_old, minibatch.advs[..., 0])
        cost_surrogate = self._get_cost_surrogate(
            logp, minibatch.logp_old, minibatch.advs[..., 1]
        )
        loss_actor_total = objective + cost_surrogate

        # get gradient
        grad_g = self._get_flat_grad(objective, self.actor, retain_graph=True)
        grad_b = self._get_flat_grad(-cost_surrogate, self.actor, retain_graph=True)
        flat_kl_grad = self._get_flat_grad(kl, self.actor, create_graph=True)
        H_inv_g = self._conjugate_gradients(grad_g, flat_kl_grad)
        approx_g = self._MVP(H_inv_g, flat_kl_grad)
        c_value = cost_surrogate - self._cost_limit

        # solve Lagrangian problem
        EPS = 1e-8
        if torch.dot(grad_b, grad_b) <= EPS and c_value < 0:
            H_inv_b, scalar_r, scalar_s, A_value, B_value = [
                torch.zeros(1) for _ in range(5)
            ]
            scalar_q = torch.dot(approx_g, H_inv_g)
            optim_case = 4
        else:
            H_inv_b = self._conjugate_gradients(grad_b, flat_kl_grad)
            approx_b = self._MVP(H_inv_b, flat_kl_grad)
            scalar_q = torch.dot(approx_g, H_inv_g)
            scalar_r = torch.dot(approx_g, H_inv_b)
            scalar_s = torch.dot(approx_b, H_inv_b)

            # should be always positive (Cauchy-Shwarz)
            A_value = scalar_q - scalar_r**2 / scalar_s
            # does safety boundary intersect trust region? (positive = yes)
            B_value = 2 * self._delta - c_value**2 / scalar_s
            if c_value < 0 and B_value < 0:
                optim_case = 3
            elif c_value < 0 and B_value >= 0:
                optim_case = 2
            elif c_value >= 0 and B_value >= 0:
                optim_case = 1
            else:
                optim_case = 0

        if optim_case in [3, 4]:
            lam = torch.sqrt(scalar_q / (2 * self._delta))
            nu = torch.zeros_like(lam)
        elif optim_case in [1, 2]:
            LA, LB = [0, scalar_r / c_value], [scalar_r / c_value, np.inf]
            LA, LB = (LA, LB) if c_value < 0 else (LB, LA)
            proj = lambda x, L: max(L[0], min(L[1], x))
            lam_a = proj(torch.sqrt(A_value / B_value), LA)
            lam_b = proj(torch.sqrt(scalar_q / (2 * self._delta)), LB)
            f_a = lambda lam: -0.5 * (A_value / (lam + EPS) + B_value * lam
                                      ) - scalar_r * c_value / (scalar_s + EPS)
            f_b = lambda lam: -0.5 * (scalar_q / (lam + EPS) + 2 * self._delta * lam)
            lam = lam_a if f_a(lam_a) >= f_b(lam_b) else lam_b
            lam = torch.tensor(lam)
            nu = max(0, (lam * c_value - scalar_r).item()) / (scalar_s + EPS)
        else:
            nu = torch.sqrt(2 * self._delta / (scalar_s + EPS))
            lam = torch.zeros_like(nu)
        # line search
        with torch.no_grad():
            delta_theta = (1. / (lam + EPS)) * (
                H_inv_g + nu * H_inv_b
            ) if optim_case > 0 else nu * H_inv_b
            delta_theta /= torch.norm(delta_theta)
            beta = 1.0
            # sometimes the scalar_q can be negative causing lam to be nan
            if not torch.isnan(lam):
                init_theta = self._get_flat_params(self.actor).clone().detach()
                init_objective = objective.clone().detach()
                init_cost_surrogate = cost_surrogate.clone().detach()
                for _ in range(self._max_backtracks):
                    theta = beta * delta_theta + init_theta
                    self._set_from_flat_params(self.actor, theta)
                    dist = self.forward(minibatch).dist
                    logp = dist.log_prob(minibatch.act)
                    new_kl = kl_divergence(dist_old, dist).mean().item()
                    new_objective = self._get_objective(
                        logp, minibatch.logp_old, minibatch.advs[..., 0]
                    )
                    new_cost_surrogate = self._get_cost_surrogate(
                        logp, minibatch.logp_old, minibatch.advs[..., 1]
                    )
                    if new_kl <= self._delta and \
                        (new_objective > init_objective if optim_case > 1 else True) and \
                        new_cost_surrogate - init_cost_surrogate <= max(-c_value.item(), 0): # noqa
                        break
                    beta *= self._backtrack_coeff

        stats_actor = {
            "loss/kl": kl.item(),
            "loss/entropy": ent.item(),
            "loss/rew_loss": objective.item(),
            "loss/cost_loss": cost_surrogate.item(),
            "loss/optim_A": A_value.item(),
            "loss/optim_B": B_value.item(),
            "loss/optim_C": c_value.item(),
            "loss/optim_Q": scalar_q.item(),
            "loss/optim_R": scalar_r.item(),
            "loss/optim_S": scalar_s.item(),
            "loss/optim_lam": lam.item(),
            "loss/optim_nu": nu.item(),
            "loss/optim_case": optim_case,
            "loss/step_size": beta
        }
        return loss_actor_total, stats_actor