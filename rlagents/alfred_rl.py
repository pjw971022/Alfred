from typing import List, Optional, Tuple, Union, Sequence

import torch
from tianshou.utils.net.common import MLP
import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
SIGMA_MIN = -20
SIGMA_MAX = 2

from typing import Any,Dict
import numpy as np
class AlfredActor(nn.Module):

    def __init__(
        self,
        type_action_shape: Sequence[int], # AlfredSubgoal.get_action_type_space_dim()
        arg_action_shape: Sequence[int], # segdef.get_num_objects() + 1
        hidden_sizes : int = 128, # dmodel
        input_dim : int = 384,
        device: Union[str, int, torch.device] = "cpu",
        joint_prob=False,
    ) -> None:
        super().__init__()
        self.device = device
        # self.preprocess = preprocess_net
        self.num_types = type_action_shape
        self.num_args = arg_action_shape  
        self.joint_prob = joint_prob
        self.linear_a = nn.Linear(input_dim, hidden_sizes)
        self.linear_a1 = nn.Linear(hidden_sizes, hidden_sizes)
        self.linear_a2 = nn.Linear(hidden_sizes * 2, hidden_sizes)

        if self.joint_prob:
            self.linear_b = nn.Linear(hidden_sizes * 3, self.num_types + self.num_args * self.num_types)
        else:
            self.linear_b = nn.Linear(hidden_sizes * 3, self.num_types + self.num_args)
        self.act = nn.LeakyReLU()

    def forward(
        self,
        combined_embedding: Union[np.ndarray, torch.Tensor],
        state: Any = None,
        info=None
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""
        
        # logits, hidden = self.preprocess(obs, state)
        # logits = self.last(logits)
        if self.device is not None:
            combined_embedding = torch.as_tensor(combined_embedding, device=self.device, dtype=torch.float32)
        x1 = self.act(self.linear_a(combined_embedding))
        x2 = self.act(self.linear_a1(x1))
        x12 = torch.cat([x1, x2], dim=1)
        x3 = self.act(self.linear_a2(x12))

        x123 = torch.cat([x1, x2, x3], dim=1)
        x = self.linear_b(x123)
        #x = self.linear_b(self.act(self.linear_a(combined_embedding)))

        act_type_logits = x[:, :self.num_types]
        act_arg_logits = x[:, self.num_types:]
        
        # if self.softmax_output:
        #     act_type_logits = F.softmax(act_type_logits, dim=-1)
        #     act_arg_logits = F.softmax(act_arg_logits, dim=-1)
            
        if self.joint_prob:
            # Output Type x Arg matrix of P(argument | type)
            b = act_arg_logits.shape[0]
            act_arg_logits = act_arg_logits.view([b, self.num_types, self.num_args])
            act_type_logprob = F.log_softmax(act_type_logits, dim=1)
            act_arg_logprob = F.log_softmax(act_arg_logits, dim=2)
            
        else:
            # Output P(argument), P(type) separately
            # TODO: This seems redundant given lines 41-42
            act_type_logits = x[:, :self.num_types]
            act_arg_logits = x[:, self.num_types:]

            act_type_logprob = F.log_softmax(act_type_logits, dim=1)
            act_arg_logprob = F.log_softmax(act_arg_logits, dim=1)
        
        return act_type_logprob, act_arg_logprob, state


class AlfredActorIndependent(AlfredActor):
    def __init__(self,   
                 type_action_shape: Sequence[int], # AlfredSubgoal.get_action_type_space_dim()
                 arg_action_shape: Sequence[int], # segdef.get_num_objects() + 1
                 device: Union[str, int, torch.device] = "cpu",):
        
        super().__init__(type_action_shape, arg_action_shape, device=device, joint_prob=False)
        
        
    def forward(self, 
                combined_embedding: Union[np.ndarray, torch.Tensor],
                state: Any = None,
                info=None):
        
        r"""Mapping: s -> Q(s, \*)."""
        # logits, hidden = self.preprocess(obs, state)
        # logits = self.last(logits)
        if self.device is not None:
            combined_embedding = torch.as_tensor(combined_embedding, device=self.device, dtype=torch.float32)
        x1 = self.act(self.linear_a(combined_embedding))
        x2 = self.act(self.linear_a1(x1))
        x12 = torch.cat([x1, x2], dim=1)
        x3 = self.act(self.linear_a2(x12))

        x123 = torch.cat([x1, x2, x3], dim=1)
        x = self.linear_b(x123)
        #x = self.linear_b(self.act(self.linear_a(combined_embedding)))

        act_type_logits = x[:, :self.num_types]
        act_arg_logits = x[:, self.num_types:]
 
        act_type_logprob = F.log_softmax(act_type_logits, dim=-1)
        act_arg_logprob = F.log_softmax(act_arg_logits, dim=-1)
        logprobs = torch.cat([act_type_logprob, act_arg_logprob],dim=-1)
        return logprobs, state


class AlfredCritic(nn.Module):
    def __init__(
        self,
        preprocess_net: nn.Module,
        hidden_sizes: Sequence[int] = (),
        last_size: int = 1,
        preprocess_net_output_dim: Optional[int] = None,
        device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.preprocess = preprocess_net
        self.output_dim = last_size
        input_dim = getattr(preprocess_net, "output_dim", preprocess_net_output_dim)
        self.last = MLP(
            input_dim,  # type: ignore
            last_size,
            hidden_sizes,
            device=self.device
        )

    def forward(
        self, obs: Union[np.ndarray, torch.Tensor], **kwargs: Any
    ) -> torch.Tensor:
        """Mapping: s -> V(s)."""
        logits, _ = self.preprocess(obs, state=kwargs.get("state", None))
        return self.last(logits)

