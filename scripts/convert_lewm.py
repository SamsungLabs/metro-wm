# Adapted from LeWorldModel (LeWM); MIT License, see LICENSES/LeWM-LICENSE.
"""Build a LeWM object checkpoint from the released HuggingFace weights.

    python scripts/convert_lewm.py --bench pusht --hf_dir checkpoints/lewm-pusht

Writes $STABLEWM_HOME/<bench>/lewm_object.ckpt, the same format as the released PLDM checkpoints.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
from transformers import ViTConfig, ViTModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import stable_worldmodel as swm

import module
from jepa import JEPA

VIT_SIZES = {"tiny": dict(hidden_size=192, num_hidden_layers=12, num_attention_heads=3, intermediate_size=768)}


def build(cfg):
    e, p, a = cfg["encoder"], cfg["predictor"], cfg["action_encoder"]
    vit = ViTModel(ViTConfig(**VIT_SIZES[e["size"]], image_size=e["image_size"], patch_size=e["patch_size"]),
                   add_pooling_layer=False, use_mask_token=e.get("use_mask_token", False))
    predictor = module.ARPredictor(num_frames=p["num_frames"], depth=p["depth"], heads=p["heads"],
                                   mlp_dim=p["mlp_dim"], input_dim=p["input_dim"],
                                   hidden_dim=p["hidden_dim"], output_dim=p["output_dim"],
                                   dim_head=p["dim_head"], dropout=p["dropout"],
                                   emb_dropout=p["emb_dropout"])
    action_encoder = module.Embedder(input_dim=a["input_dim"], emb_dim=a["emb_dim"])

    def mlp(c):
        return module.MLP(c["input_dim"], c["hidden_dim"], c["output_dim"], norm_fn=nn.BatchNorm1d)

    return JEPA(vit, predictor, action_encoder, mlp(cfg["projector"]), mlp(cfg["pred_proj"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=["pusht", "cube", "tworoom", "reacher"])
    ap.add_argument("--hf_dir", required=True, help="directory holding config.json and weights.pt")
    a = ap.parse_args()
    hf = Path(a.hf_dir)
    model = build(json.loads((hf / "config.json").read_text()))
    sd = torch.load(hf / "weights.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=True)
    got = model.state_dict()
    assert all(torch.equal(got[k], sd[k]) for k in sd), "weights did not load exactly"
    model.eval()
    out = swm.data.utils.get_cache_dir() / a.bench / "lewm_object.ckpt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, out)
    print(f"wrote {out} ({len(sd)} tensors)")


if __name__ == "__main__":
    main()
