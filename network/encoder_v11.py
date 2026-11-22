"""Untrained global encoder for l1.input/1.1. Not a policy and not a command path.

Valid tokens may read each other. Four-state codes stay separate from the
numbers, so a known zero is not the same input as an unknown zero. Old 54-dim
weights are not loaded. A later sequence model can add a [batch, tokens, 128]
history tensor; this version does not build that network.
"""

from __future__ import annotations

import torch
from torch import nn

from network.input_v10 import (
    CAT_DIM,
    CATEGORIES,
    D_CONT as D_CONT_10,
    KNOWN,
    N_OWN_CAP,
)
from network.input_v11 import D_CONT, ENTITY_KIND_11

SCHEMA = "l1.encoder/1.1"
D_MODEL = 128
N_HEAD = 4
N_LAYER = 4
FF = 1024
BLOCKED = -1.0e9
N_STATE = 4
VOCAB = (len(ENTITY_KIND_11),) + tuple(len(table) for _name, table in CATEGORIES[1:])


def inspect_links(
    entity_valid: torch.Tensor,
    part_parent: torch.Tensor,
    category: torch.Tensor,
    own_index: torch.Tensor,
    unit_valid: torch.Tensor,
) -> list[str]:
    """Find parent and own-slot links that the encoder must not quietly accept."""
    valid = entity_valid.reshape(-1).bool()
    parent = part_parent.reshape(-1).long()
    kind = category.reshape(-1, category.shape[-1])[:, 0].long()
    own = own_index.reshape(-1).long()
    slots = unit_valid.reshape(-1).bool()
    count = int(valid.shape[0])
    problems: list[str] = []
    for index in range(count):
        link = int(parent[index])
        if link == -1:
            continue
        if link < -1 or link >= count:
            problems.append(f"令牌 {index} 的父索引 {link} 越界")
        elif link == index:
            problems.append(f"令牌 {index} 指向自己")
        elif not bool(valid[link]):
            problems.append(f"令牌 {index} 的父令牌 {link} 无效")
        elif int(kind[link]) != 3:
            problems.append(f"令牌 {index} 的父令牌 {link} 不是平台")
    for slot in range(int(slots.shape[0])):
        link = int(own[slot])
        if not bool(slots[slot]):
            if link != -1:
                problems.append(f"无效己方槽 {slot} 仍指向令牌 {link}")
            continue
        if link < 0 or link >= count:
            problems.append(f"己方槽 {slot} 的索引 {link} 越界")
        elif not bool(valid[link]):
            problems.append(f"己方槽 {slot} 指向无效令牌 {link}")
        elif int(kind[link]) != 3:
            problems.append(f"己方槽 {slot} 指向的令牌 {link} 不是平台")
    return problems


def global_mask(valid: torch.Tensor) -> torch.Tensor:
    """Every valid token reads every valid token. Invalid tokens only see themselves."""
    flag = valid.reshape(-1).bool()
    n = int(flag.shape[0])
    mask = torch.full((n, n), BLOCKED, dtype=torch.float32, device=flag.device)
    index = flag.nonzero(as_tuple=False).flatten()
    if int(index.numel()):
        mask[index[:, None], index[None, :]] = 0.0
    missing = (~flag).nonzero(as_tuple=False).flatten()
    if int(missing.numel()):
        mask[missing, missing] = 0.0
    return mask


class HistoryEncoder(nn.Module):
    """Reserved sequence input. The frame already carries age, gap, and deltas."""

    def forward(
        self,
        hist_cont: torch.Tensor,
        hist_state: torch.Tensor,
        hist_time: torch.Tensor,
        hist_valid: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError("l1.encoder/1.1 只接收摘要后的历史向量，本轮没有序列网络")


class GlobalEncoderV11(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        if D_CONT != 38 or D_CONT_10 != 32 or CAT_DIM != 8:
            raise RuntimeError("encoder expects the accepted 38-column contract")
        self.value_proj = nn.Linear(D_CONT, D_MODEL)
        self.state_proj = nn.Linear(D_CONT * N_STATE, D_MODEL)
        self.categories = nn.ModuleList(nn.Embedding(size, D_MODEL) for size in VOCAB)
        self.parent_proj = nn.Linear(D_MODEL, D_MODEL)
        self.history_proj = nn.Linear(D_MODEL, D_MODEL)
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEAD,
            dim_feedforward=FF,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=N_LAYER)

    def pre_attention(
        self,
        cont: torch.Tensor,
        feature_state: torch.Tensor,
        category: torch.Tensor,
        part_parent: torch.Tensor,
        history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if cont.shape[-1] != D_CONT or feature_state.shape[-1] != D_CONT or category.shape[-1] != CAT_DIM:
            raise ValueError("expected 38 continuous columns, 38 states, and 8 categories")
        if int(feature_state.max()) > 3 or int(feature_state.min()) < 0:
            raise ValueError("feature state is outside the four codes")
        for index, table in enumerate(self.categories):
            if int(category[..., index].max()) >= table.num_embeddings:
                raise ValueError("category id is outside this encoder")
        known = feature_state == KNOWN
        hidden = self.value_proj(cont * known.to(cont.dtype))
        one_hot = torch.nn.functional.one_hot(feature_state.long(), N_STATE).to(cont.dtype)
        hidden = hidden + self.state_proj(one_hot.flatten(-2))
        cats = category.long()
        for index, table in enumerate(self.categories):
            hidden = hidden + table(cats[..., index])
        safe = part_parent.clamp(min=0)
        parent = hidden.gather(1, safe.unsqueeze(-1).expand(-1, hidden.shape[1], D_MODEL))
        hidden = hidden + self.parent_proj(parent) * (part_parent >= 0).unsqueeze(-1).to(hidden.dtype)
        if history is not None:
            if history.shape != hidden.shape:
                raise ValueError("history must be [batch, tokens, 128]")
            hidden = hidden + self.history_proj(history)
        return hidden

    def token_states(
        self,
        cont: torch.Tensor,
        feature_state: torch.Tensor,
        category: torch.Tensor,
        entity_valid: torch.Tensor,
        part_parent: torch.Tensor,
        history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if cont.shape[0] != 1:
            raise ValueError("expected one frame")
        hidden = self.pre_attention(cont, feature_state, category, part_parent, history)
        valid = entity_valid.bool()
        hidden = hidden * valid.unsqueeze(-1).to(hidden.dtype)
        mask = global_mask(valid[0]).to(device=hidden.device, dtype=hidden.dtype)
        encoded = self.encoder(hidden, mask=mask)
        return encoded * valid.unsqueeze(-1).to(encoded.dtype)

    def forward(
        self,
        cont: torch.Tensor,
        feature_state: torch.Tensor,
        category: torch.Tensor,
        entity_valid: torch.Tensor,
        part_parent: torch.Tensor,
        own_index: torch.Tensor,
        unit_valid: torch.Tensor,
        history: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if own_index.shape[-1] != N_OWN_CAP or unit_valid.shape[-1] != N_OWN_CAP:
            raise ValueError("expected 64 own slots")
        problems = inspect_links(entity_valid, part_parent, category, own_index, unit_valid)
        if problems:
            raise ValueError("；".join(problems))
        encoded = self.token_states(cont, feature_state, category, entity_valid, part_parent, history)
        index = own_index.clamp(min=0)
        gathered = encoded.gather(1, index.unsqueeze(-1).expand(-1, N_OWN_CAP, D_MODEL))
        own = gathered * unit_valid.unsqueeze(-1).to(gathered.dtype)
        return encoded, own
