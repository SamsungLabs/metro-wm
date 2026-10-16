# Adapted from LeWorldModel (LeWM); MIT License, see LICENSES/LeWM-LICENSE.
"""LeWM / PLDM world model (inference only).

The authors' released `*_object.ckpt` files pickle `jepa.JEPA` and `module.*` by top-level module
name, so this file and `module.py` must stay at the repository root under these names.
"""
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def normalize_pixels(pixels):
    """uint8 channel-first pixels -> ImageNet-normalized floats, on the pixels' device."""
    pixels = pixels.float().div(255.0)
    view = [1] * pixels.ndim
    view[-3] = 3
    mean = torch.tensor(_IMAGENET_MEAN, device=pixels.device, dtype=pixels.dtype).view(view)
    std = torch.tensor(_IMAGENET_STD, device=pixels.device, dtype=pixels.dtype).view(view)
    return (pixels - mean) / std


def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v


class JEPA(nn.Module):
    def __init__(self, encoder, predictor, action_encoder, projector=None, pred_proj=None):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()

    def encode(self, info):
        """info["pixels"]: uint8 (B, T, C, H, W) -> info["emb"]: (B, T, D)."""
        pixels = normalize_pixels(info["pixels"])
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...")
        output = self.encoder(pixels, interpolate_pos_encoding=True, output_hidden_states=True)
        emb = self.projector(output.last_hidden_state[:, 0])
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)
        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])
        return info

    def predict(self, emb, act_emb):
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        return rearrange(preds, "(b t) d -> b t d", b=emb.size(0))

    def rollout(self, info, action_sequence, history_size: int = 3):
        """Roll the predictor out from the encoded current frame.

        info["pixels"]: (B, S, T, C, H, W); action_sequence: (B, S, T, action_dim).
        """
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, *([-1] * (_init["emb"].dim() - 1)))
        _init = {k: detach_clone(v) for k, v in _init.items()}

        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        HS = history_size
        for t in range(n_steps):
            act_emb = self.action_encoder(act)
            pred_emb = self.predict(emb[:, -HS:], act_emb[:, -HS:])[:, -1:]
            emb = torch.cat([emb, pred_emb], dim=1)
            act = torch.cat([act, act_future[:, t: t + 1, :]], dim=1)
        act_emb = self.action_encoder(act)
        pred_emb = self.predict(emb[:, -HS:], act_emb[:, -HS:])[:, -1:]
        emb = torch.cat([emb, pred_emb], dim=1)

        info["predicted_emb"] = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        return info

    def criterion(self, info_dict):
        """Squared L2 between the last predicted latent and the goal latent -> (B, S)."""
        pred_emb = info_dict["predicted_emb"]
        goal_emb = info_dict["goal_emb"]
        cost = F.mse_loss(pred_emb[..., -1:, :], goal_emb[..., -1:, :].detach(), reduction="none")
        return cost.sum(dim=tuple(range(2, pred_emb.ndim)))

    def get_cost(self, info_dict, action_candidates):
        """Cost of action candidates against the latent sub-goal in info_dict["goal_emb"]."""
        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)
        info_dict = self.rollout(info_dict, action_candidates)
        return self.criterion(info_dict)
