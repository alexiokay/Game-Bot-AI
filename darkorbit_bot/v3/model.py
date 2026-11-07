"""
ObjectTokenPolicy - the single V3 policy.

Tokens:  [CLS] [player] [previous action] [object 0..15]
Encoder: small Transformer (d=128, 3 layers, 4 heads) with a padding mask
Heads:
    target   pointer over the 16 objects + "none"          [B, N+1]
    mouse    next CHUNK cursor positions (logits of x, y)  [B, C, 2]
             = aim prior (pointer-weighted object position, or the current
               cursor for "none") + a learned correction
    buttons  next CHUNK left/right button + key logits     [B, C, 2+K]

The checkpoint stores PolicyConfig, so loading never guesses shapes.
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .schema import (MAX_OBJECTS, OBJECT_DIM, PLAYER_DIM, ACTION_DIM, KEYS,
                     CHUNK, REPLAN_EVERY)


@dataclass
class PolicyConfig:
    d_model: int = 128
    layers: int = 3
    heads: int = 4
    dropout: float = 0.1
    chunk: int = CHUNK
    max_objects: int = MAX_OBJECTS
    object_dim: int = OBJECT_DIM
    player_dim: int = PLAYER_DIM
    action_dim: int = ACTION_DIM
    keys: Tuple[str, ...] = field(default_factory=lambda: tuple(KEYS))

    @property
    def num_buttons(self) -> int:  # left, right, keys
        return 2 + len(self.keys)


def _logit(p: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    p = p.clamp(eps, 1 - eps)
    return torch.log(p / (1 - p))


class ObjectTokenPolicy(nn.Module):
    def __init__(self, cfg: Optional[PolicyConfig] = None):
        super().__init__()
        self.cfg = cfg or PolicyConfig()
        c = self.cfg
        d = c.d_model

        self.obj_in = nn.Sequential(nn.Linear(c.object_dim, d), nn.GELU(), nn.Linear(d, d))
        self.player_in = nn.Sequential(nn.Linear(c.player_dim, d), nn.GELU(), nn.Linear(d, d))
        self.action_in = nn.Sequential(nn.Linear(c.action_dim, d), nn.GELU(), nn.Linear(d, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.type_emb = nn.Embedding(4, d)  # 0 cls, 1 player, 2 prev action, 3 object

        layer = nn.TransformerEncoderLayer(d, c.heads, dim_feedforward=4 * d, dropout=c.dropout,
                                           batch_first=True, norm_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(layer, c.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)

        # Target pointer
        self.obj_score = nn.Linear(d, 1)
        self.none_score = nn.Linear(d, 1)

        # Action chunk from [cls, attended target, soft target xy]
        self.act_mlp = nn.Sequential(nn.Linear(2 * d + 2, 2 * d), nn.GELU(), nn.Linear(2 * d, 2 * d), nn.GELU())
        self.mouse_head = nn.Linear(2 * d, c.chunk * 2)
        self.button_head = nn.Linear(2 * d, c.chunk * c.num_buttons)

        nn.init.normal_(self.cls, std=0.02)
        # Start with "no correction" from the aim prior
        nn.init.zeros_(self.mouse_head.weight)
        nn.init.zeros_(self.mouse_head.bias)
        # Start buttons strongly "not pressed" (keys are rarely down)
        nn.init.constant_(self.button_head.bias, -3.0)

    def forward(self, player: torch.Tensor, prev_action: torch.Tensor,
                objects: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        player [B, P], prev_action [B, A], objects [B, N, O], mask [B, N] (1 = real object)
        """
        B, N, _ = objects.shape
        c = self.cfg
        dev = objects.device
        types = self.type_emb.weight
        tok_cls = self.cls.expand(B, 1, -1) + types[0]
        tok_player = self.player_in(player).unsqueeze(1) + types[1]
        tok_prev = self.action_in(prev_action).unsqueeze(1) + types[2]
        tok_obj = self.obj_in(objects) + types[3]
        x = torch.cat([tok_cls, tok_player, tok_prev, tok_obj], dim=1)

        valid = mask > 0.5
        pad = torch.cat([torch.zeros(B, 3, dtype=torch.bool, device=dev), ~valid], dim=1)
        h = self.norm(self.encoder(x, src_key_padding_mask=pad))
        h_cls, h_obj = h[:, 0], h[:, 3:]

        # Pointer: objects (masked) + "none"
        obj_logits = self.obj_score(h_obj).squeeze(-1).masked_fill(~valid, -1e4)
        target_logits = torch.cat([obj_logits, self.none_score(h_cls)], dim=1)  # [B, N+1]

        # Soft target (differentiable): attended embedding and position
        probs = F.softmax(target_logits, dim=1)
        p, p_none = probs[:, :N], probs[:, N:]                           # [B, N], [B, 1]
        att = torch.bmm(p.unsqueeze(1), h_obj).squeeze(1)                # [B, d]
        cursor_prev = prev_action[:, 0:2]                                # where the cursor was
        # Aim prior: the chosen object's position, or the current cursor for
        # "none". The mouse head only predicts a correction from it (lead,
        # offset, movement), so aiming is easy to learn - and the mouse loss
        # trains the pointer on every frame, not only on frames before a click.
        prior_xy = torch.bmm(p.unsqueeze(1), objects[:, :, 0:2]).squeeze(1) + p_none * cursor_prev

        z = self.act_mlp(torch.cat([h_cls, att, prior_xy], dim=-1))
        delta = self.mouse_head(z).view(B, c.chunk, 2)
        mouse = _logit(prior_xy).unsqueeze(1) + delta                    # logits of x, y
        buttons = self.button_head(z).view(B, c.chunk, c.num_buttons)
        return {'target_logits': target_logits, 'mouse_logits': mouse, 'button_logits': buttons}


# --------------------------------------------------------------------------
# Save / load
# --------------------------------------------------------------------------

def save_policy(model: ObjectTokenPolicy, path, extra: Optional[Dict] = None):
    from ..v2.training.training_utils import atomic_torch_save
    cfg = asdict(model.cfg)
    cfg['keys'] = list(cfg['keys'])
    atomic_torch_save({'format': 'v3_object_token_policy', 'config': cfg,
                       'state_dict': model.state_dict(), 'extra': extra or {}}, path)


def load_policy(path, device: str = "cpu") -> ObjectTokenPolicy:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt.get('format') != 'v3_object_token_policy':
        raise ValueError(f"{path} is not a V3 policy checkpoint")
    cfg_d = dict(ckpt['config'])
    cfg_d['keys'] = tuple(cfg_d['keys'])
    model = ObjectTokenPolicy(PolicyConfig(**cfg_d))
    model.load_state_dict(ckpt['state_dict'], strict=True)  # shapes must match exactly
    model.to(device).eval()
    model.extra = ckpt.get('extra', {})
    return model


# --------------------------------------------------------------------------
# Runtime: replanning + temporal ensembling
# --------------------------------------------------------------------------

class PolicyRunner:
    """Runs the model every REPLAN_EVERY steps and blends overlapping chunks
    (newer chunks weigh more), like V2's temporal ensembling."""

    def __init__(self, model: ObjectTokenPolicy, device: str = "cpu",
                 replan_every: int = REPLAN_EVERY, decay: float = 0.5,
                 button_threshold: float = 0.5):
        self.model = model.to(device).eval()
        self.device = device
        self.replan_every = replan_every
        self.decay = decay
        self.button_threshold = button_threshold
        self.keys = model.cfg.keys
        self.reset()

    def reset(self):
        self._step = 0
        self._chunks: List[Tuple[int, np.ndarray, np.ndarray]] = []  # (start, mouse[C,2], buttons[C,K])
        self._last_target = (-1, 0.0)

    @torch.no_grad()
    def act(self, feats: Dict) -> Dict:
        if self._step % self.replan_every == 0 or not self._chunks:
            t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=self.device).unsqueeze(0)
            out = self.model(t(feats['player']), t(feats['prev_action']), t(feats['objects']), t(feats['mask']))
            mouse = torch.sigmoid(out['mouse_logits'])[0].cpu().numpy()
            buttons = torch.sigmoid(out['button_logits'])[0].cpu().numpy()
            probs = F.softmax(out['target_logits'], dim=1)[0].cpu().numpy()
            ti = int(np.argmax(probs))
            self._last_target = (ti if ti < self.model.cfg.max_objects else -1, float(probs[ti]))
            self._chunks.append((self._step, mouse, buttons))
            C = self.model.cfg.chunk
            self._chunks = [ch for ch in self._chunks if self._step - ch[0] < C]

        mouse_acc, btn_acc, w_acc = np.zeros(2), 0.0, 0.0
        for start, mouse, buttons in self._chunks:
            k = self._step - start
            w = np.exp(-self.decay * k)  # newer prediction (small k) weighs more
            mouse_acc = mouse_acc + w * mouse[k]
            btn_acc = btn_acc + w * buttons[k]
            w_acc += w
        mouse_xy = mouse_acc / w_acc
        btn = btn_acc / w_acc
        self._step += 1

        th = self.button_threshold
        return {
            'mx': float(mouse_xy[0]), 'my': float(mouse_xy[1]),
            'lmb': bool(btn[0] > th), 'rmb': bool(btn[1] > th),
            'keys': [k for i, k in enumerate(self.keys) if btn[2 + i] > th],
            'target_idx': self._last_target[0], 'target_prob': self._last_target[1],
        }
