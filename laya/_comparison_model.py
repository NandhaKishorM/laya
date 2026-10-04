"""Inference networks for the optional, locally loaded comparison specialist."""
import math

import torch
from torch import nn
import torch.nn.functional as F


class _Backbone(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.encoder, self.head = base.encoder, base.head
        self.type_emb, self.scorer = base.type_emb, base.scorer

    def contextualize(self, ids, mask):
        h = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        h = h + self.type_emb.weight[0][None, None, :]
        if self.head is not None:
            for layer in self.head.layers:
                h = layer(h, src_key_padding_mask=~mask)
        return h


class OperatorNetwork(_Backbone):
    """Predict a relation directly or compose local predicates, order and scope."""
    def __init__(self, base, variant, width=128):
        super().__init__(base)
        if variant not in ("flat", "local_program"):
            raise ValueError("Unknown comparison operator architecture")
        self.variant, self.width = variant, width
        hidden = self.encoder.config.hidden_size
        if variant == "local_program":
            self.program_norm = nn.LayerNorm(hidden)
            self.program_memory = nn.Linear(hidden, width)
            self.program_roles = nn.Parameter(torch.zeros(7, width))
            self.program_cross = nn.MultiheadAttention(width, 4, dropout=0., batch_first=True)
            self.program_self = nn.TransformerEncoderLayer(width, 4, width * 2, dropout=.1,
                                                           batch_first=True, norm_first=True)
            self.program_binary = nn.Linear(width, 1)
            self.token_pointer = nn.Linear(hidden, 1)
            self.token_negative = nn.Linear(hidden, 1)
            self.token_local_norm = nn.LayerNorm(hidden)
            self.token_local_project = nn.Linear(hidden, width)
            self.token_local_conv = nn.Conv1d(width, width, 5, padding=2)
            self.token_local_base = nn.Linear(width, 6)

    def forward(self, batch):
        h = self.contextualize(batch["ids"], batch["mask"])
        if self.variant == "flat":
            pos = batch["positions"].unsqueeze(-1).expand(-1, -1, h.shape[-1])
            return self.scorer(h.gather(1, pos)).squeeze(-1).float()
        h = self.program_norm(h)
        pointer = self.token_pointer(h).squeeze(-1).float().masked_fill(~batch["claim_mask"], -1e4).softmax(-1)
        lexical = self.encoder.get_input_embeddings()(batch["ids"])
        lexical = self.token_local_project(self.token_local_norm(lexical)) * batch["claim_mask"][..., None]
        lexical = F.gelu(self.token_local_conv(lexical.transpose(1, 2)).transpose(1, 2))
        probability = (pointer[..., None] * self.token_local_base(lexical).float().softmax(-1)).sum(1)
        memory = self.program_memory(h)
        roles = self.program_roles[None].expand(len(h), -1, -1)
        slots, _ = self.program_cross(roles, memory, memory, key_padding_mask=~batch["claim_mask"], need_weights=True)
        slots = self.program_self(slots + roles)
        swap = self.program_binary(slots[:, 1:]).squeeze(-1).float()[:, 0].sigmoid()
        events = self.token_negative(h).squeeze(-1).float().sigmoid()
        factors = torch.where(batch["claim_mask"], 1 - 2 * events, torch.ones_like(events))
        odd = .5 * (1 - factors.prod(-1))
        probability = (1 - swap[:, None]) * probability + swap[:, None] * probability[:, (2, 3, 0, 1, 4, 5)]
        probability = (1 - odd[:, None]) * probability + odd[:, None] * probability[:, (3, 2, 1, 0, 5, 4)]
        return probability.clamp_min(1e-8).log()


class OperandNetwork(_Backbone):
    """Learn operand mentions, bind exact identifiers, and mix exact comparisons."""
    def __init__(self, base, width=128):
        super().__init__(base)
        self.width = width
        hidden = self.encoder.config.hidden_size
        self.binding_norm = nn.LayerNorm(hidden)
        self.binding_gate = nn.Linear(hidden, 1)
        self.mention_key = nn.Linear(hidden, width)
        self.mention_roles = nn.Parameter(torch.zeros(2, width))
        self.mention_query = nn.Linear(hidden, width * 2)

    def forward(self, batch, operator_probability):
        h = self.contextualize(batch["ids"], batch["mask"])
        pos = batch["positions"].unsqueeze(-1).expand(-1, -1, h.shape[-1])
        neural = self.scorer(h.gather(1, pos)).squeeze(-1).float()
        h = self.binding_norm(h)
        query = self.mention_query(h[:, 0]).reshape(len(h), 2, -1) + self.mention_roles
        attention = torch.einsum("brd,btd->brt", query, self.mention_key(h)).float() / math.sqrt(self.width)
        attention = attention.masked_fill(~batch["assertion_mask"][:, None], -1e4).softmax(-1)
        # Mixing and arithmetic probabilities stay FP32 even under encoder autocast.
        with torch.autocast(device_type=h.device.type, enabled=False):
            linked = torch.einsum("brt,bft->brf", attention.float(), batch["identity_links"].float())
            pointers = linked.clamp_min(1e-12).log().masked_fill(~batch["field_mask"][:, None], -1e4)
            p = pointers.softmax(-1)
            exact = torch.einsum("bi,bj,bo,bijoc->bc", p[:, 0], p[:, 1], operator_probability.float(),
                                 batch["comparison_table"].float())
        gate = self.binding_gate(h[:, 0]).float().sigmoid().squeeze(-1)
        logits = ((1 - gate[:, None]) * neural.softmax(-1) + gate[:, None] * exact).clamp_min(1e-12).log()
        return logits, pointers, gate
