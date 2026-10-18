# Adapted from H-JEPA; MIT License, see LICENSES/H-JEPA-LICENSE.
"""Port the flat LeWM of an H-JEPA release (Zhang et al., arXiv:2610.06805) into `jepa.JEPA`.

H-JEPA's flat LeWM is a LeWM network (HF ViT + CLS projector, AR predictor + projector, action
embedder) wrapped in H-JEPA's classes. The port renames its state dict onto Metro's LeWM class and
checks that the result reproduces H-JEPA's own encoder and predictor outputs.

    # 1. in H-JEPA's environment: weights, architecture and reference outputs
    python scripts/convert_hjepa.py export --ckpt $HJEPA_HOME/ckpts/pusht/pusht_lewm/seed42/pusht_lewm_object.ckpt \
        --out exports/pusht_lewm_s42.pt
    # 2. in Metro's environment: rebuild, check, write $STABLEWM_HOME/<bench>/<wm>_object.ckpt
    python scripts/convert_hjepa.py build --export exports/pusht_lewm_s42.pt --bench pusht --wm hj_lewm_s42
"""
import argparse
import sys
from pathlib import Path

import torch
from torch import nn

REPO = Path(__file__).resolve().parent.parent
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
RENAME = {  # H-JEPA level-1 JEPA -> jepa.JEPA
    "encoder.encoder.": "encoder.",
    "encoder.projector.": "projector.",
    "predictor.predictor.": "predictor.",
    "predictor.projector.": "pred_proj.",
    "action_embed.": "action_encoder.",
}
VIT_RENAME = {  # ViT block parameters: newer transformers (H-JEPA's environment) -> 5.3 (Metro's)
    "attention.q_proj.": "attention.attention.query.",
    "attention.k_proj.": "attention.attention.key.",
    "attention.v_proj.": "attention.attention.value.",
    "attention.o_proj.": "attention.output.dense.",
    "mlp.fc1.": "intermediate.dense.",
    "mlp.fc2.": "output.dense.",
}


def vit_to_installed(sd, expected):
    """Rename newer-transformers ViT keys ('encoder.layers.N.*') to the older layout
    ('encoder.encoder.layer.N.*') if that is what the installed transformers builds."""
    if all(k in expected for k in sd):
        return sd
    out = {}
    for k, v in sd.items():
        if k.startswith("encoder.layers."):
            k = k.replace("encoder.layers.", "encoder.encoder.layer.", 1)
            k = next((k.replace(o, n, 1) for o, n in VIT_RENAME.items() if o in k), k)
        out[k] = v
    return out


def mlp_dims(sd, prefix):
    """Linear -> BatchNorm1d -> GELU -> Linear, as both codebases build LeWM's projectors."""
    assert f"{prefix}net.1.running_mean" in sd, f"{prefix}: expected a BatchNorm1d after the first Linear"
    assert not any(k.startswith(f"{prefix}final_norm.") for k in sd), f"{prefix}: has a final LayerNorm"
    w0, w3 = sd[f"{prefix}net.0.weight"], sd[f"{prefix}net.3.weight"]
    return dict(input_dim=w0.shape[1], hidden_dim=w0.shape[0], output_dim=w3.shape[0])


def export(a):
    import h_jepa
    sys.path.insert(0, h_jepa.__path__[0])                      # H-JEPA's model classes, to unpickle the checkpoint
    obj = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    l1 = obj.jepas[0] if hasattr(obj, "jepas") else next(m for m in obj.modules() if hasattr(m, "get_cost"))
    l1 = l1.eval().float()
    assert type(l1.encoder).__name__ == "ProjectedEncoder", type(l1.encoder)
    assert isinstance(l1.action_encoder, nn.Identity), "level 1 should feed raw action blocks"
    pred = l1.predictor.predictor
    blk = pred.transformer.layers[0]
    vit = l1.encoder.encoder
    sd = {k: v for k, v in l1.state_dict().items() if "latent_action_queue" not in k}
    out = {}
    for k, v in sd.items():
        new = next((n + k[len(o):] for o, n in RENAME.items() if k.startswith(o)), None)
        assert new is not None, f"unmapped key {k}"
        out[new] = v
    pe = out["action_encoder.patch_embed.weight"]
    emb_w = out["action_encoder.embed.2.weight"]
    arch = dict(
        vit_config=vit.config.to_dict(),
        use_mask_token=getattr(vit.embeddings, "mask_token", None) is not None,
        predictor=dict(num_frames=pred.pos_embedding.shape[1], input_dim=pred.pos_embedding.shape[2],
                       depth=len(pred.transformer.layers), heads=blk.attn.heads,
                       dim_head=blk.attn.to_qkv.weight.shape[0] // 3 // blk.attn.heads,
                       mlp_dim=blk.mlp.net[1].weight.shape[0], hidden_dim=blk.attn.norm.weight.shape[0],
                       output_dim=(pred.transformer.output_proj.weight.shape[0]
                                   if isinstance(pred.transformer.output_proj, nn.Linear)
                                   else blk.attn.norm.weight.shape[0])),
        action_encoder=dict(input_dim=pe.shape[1], smoothed_dim=pe.shape[0], emb_dim=emb_w.shape[0],
                            mlp_scale=out["action_encoder.embed.0.weight"].shape[0] // emb_w.shape[0]),
        projector=mlp_dims(out, "projector."),
        pred_proj=mlp_dims(out, "pred_proj."),
    )

    # reference outputs on fixed random inputs: H-JEPA feeds ImageNet-normalised pixels to its encoder,
    # Metro's JEPA normalises uint8 pixels itself
    g = torch.Generator().manual_seed(0)
    img = vit.config.image_size
    pixels = torch.randint(0, 256, (2, 3, 3, img, img), generator=g, dtype=torch.uint8)
    actions = torch.randn(2, 3, arch["action_encoder"]["input_dim"], generator=g)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 1, 3, 1, 1)
    with torch.no_grad():
        info = l1.encode({"pixels": (pixels.float() / 255 - mean) / std, "action": actions})
        emb, act_emb = info["embed_0"], info["action_0"]
        pred_emb = l1.predict(emb, act_emb)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=out, arch=arch, src=str(a.ckpt),
                    ref=dict(pixels=pixels, actions=actions, emb=emb, act_emb=act_emb, pred=pred_emb)), a.out)
    print(f"exported {a.ckpt} -> {a.out} ({len(out)} tensors)")


def build(a):
    sys.path.insert(0, str(REPO))
    import stable_worldmodel as swm
    from transformers import ViTConfig, ViTModel

    import module
    from jepa import JEPA

    ex = torch.load(a.export, map_location="cpu", weights_only=False)
    arch, sd, ref = ex["arch"], ex["state_dict"], ex["ref"]
    vit = ViTModel(ViTConfig(**arch["vit_config"]), add_pooling_layer=False, use_mask_token=arch["use_mask_token"])
    p = arch["predictor"]
    predictor = module.ARPredictor(num_frames=p["num_frames"], depth=p["depth"], heads=p["heads"],
                                   mlp_dim=p["mlp_dim"], input_dim=p["input_dim"], hidden_dim=p["hidden_dim"],
                                   output_dim=p["output_dim"], dim_head=p["dim_head"])
    action_encoder = module.Embedder(**arch["action_encoder"])

    def mlp(c):
        return module.MLP(c["input_dim"], c["hidden_dim"], c["output_dim"], norm_fn=nn.BatchNorm1d)

    model = JEPA(vit, predictor, action_encoder, mlp(arch["projector"]), mlp(arch["pred_proj"]))
    model.load_state_dict(vit_to_installed(sd, model.state_dict()), strict=True)
    model.eval()
    # save before any forward pass (transformers then attaches hooks that cannot be pickled), and
    # check the reloaded file, i.e. exactly what the evaluation loads
    dst = swm.data.utils.get_cache_dir() / a.bench / f"{a.wm}_object.ckpt"
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, dst)
    model = torch.load(dst, map_location="cpu", weights_only=False).eval()
    with torch.no_grad():
        info = model.encode({"pixels": ref["pixels"], "action": ref["actions"]})
        pred = model.predict(info["emb"], info["act_emb"])
    for name, got, want in [("emb", info["emb"], ref["emb"]), ("act_emb", info["act_emb"], ref["act_emb"]),
                            ("pred", pred, ref["pred"])]:
        err = float((got - want).abs().max() / want.abs().max())
        print(f"  {name}: max rel err {err:.2e}")
        if err >= 1e-4:
            dst.unlink()
            raise SystemExit(f"{name} does not match H-JEPA's output (rel err {err:.2e}); removed {dst}")
    print(f"wrote {dst}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--ckpt", required=True, help="H-JEPA <bench>_lewm_object.ckpt")
    e.add_argument("--out", required=True)
    b = sub.add_parser("build")
    b.add_argument("--export", required=True)
    b.add_argument("--bench", required=True, choices=["pusht", "cube"])
    b.add_argument("--wm", required=True, help="name of the ported checkpoint, e.g. hj_lewm_s42")
    a = ap.parse_args()
    export(a) if a.cmd == "export" else build(a)


if __name__ == "__main__":
    main()
