from flax import linen as nn
import jax
import jax.numpy as jnp


num_items = 200
feat_size = 100
hidden_size = 64


key = jax.random.PRNGKey(0)

Q_mat = jax.random.normal(key, (num_items, feat_size))
K_mat = jax.random.normal(key, (feat_size, hidden_size))

class Transformer(nn.Module):
    def __init__(self, 
                 feat_size: int, 
                 hidden_size: int, 
                 num_head: int):
        self.hidden_size = hidden_size
        self.feat_size = feat_size
        self.num_head = num_head

    def __call__(self, Q_mat, K_mat, V_mat):
        unscaled = Q_mat @ K_mat 
        scaled = unscaled / Q_mat.shape[1]
        out = softmax(scaled, V_mat)
        return out








from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.nn import PositionalEncoding
from typing import Any, Dict, List, Optional


# adapted from https://github.com/tgm-team/tgm/blob/main/tgm/core/timedelta.py
_UNIT_TO_NANOS = {
        'Y': 1000 * 1000 * 1000 * 60 * 60 * 24 * 365,
        'M': 1000 * 1000 * 1000 * 60 * 60 * 24 * 30,
        'W': 1000 * 1000 * 1000 * 60 * 60 * 24 * 7,
        'D': 1000 * 1000 * 1000 * 60 * 60 * 24,
        'h': 1000 * 1000 * 1000 * 60 * 60,
        'm': 1000 * 1000 * 1000 * 60,
        's': 1000 * 1000 * 1000,
        }

class HeteroTemporalEncoder(torch.nn.Module):
    def __init__(self, node_types: List[NodeType], channels: int, time_gran: str = 'D'):
        super().__init__()

        self.encoder_dict = torch.nn.ModuleDict(
            {node_type: PositionalEncoding(channels) for node_type in node_types}
        )
        self.lin_dict = torch.nn.ModuleDict(
            {node_type: torch.nn.Linear(channels, channels) for node_type in node_types}
        )
        self.default = "s"
        if (time_gran not in _UNIT_TO_NANOS):
          raise ValueError("time granularity not supported")
        self.unit = self._convert('s',time_gran)


    def _convert(self, time_1: str, time_2: str) -> float:
        self_nanos = _UNIT_TO_NANOS[time_1]
        other_nanos = _UNIT_TO_NANOS[time_2]

        if self_nanos > other_nanos:
          unit_ratio = self_nanos // other_nanos
        else:  
          unit_ratio = other_nanos // self_nanos
        return unit_ratio

    def reset_parameters(self):
        for encoder in self.encoder_dict.values():
            encoder.reset_parameters()
        for lin in self.lin_dict.values():
            lin.reset_parameters()

    def forward(
        self,
        seed_time: Tensor,
        time_dict: Dict[NodeType, Tensor],
        batch_dict: Dict[NodeType, Tensor],
    ) -> Dict[NodeType, Tensor]:
        out_dict: Dict[NodeType, Tensor] = {}

        for node_type, time in time_dict.items():
            rel_time = seed_time[batch_dict[node_type]] - time
            rel_time = rel_time / self.unit # convert seconds to a coarse graine time granularity

            x = self.encoder_dict[node_type](rel_time)
            x = self.lin_dict[node_type](x)
            out_dict[node_type] = x

        return out_dict



