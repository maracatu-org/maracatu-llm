"""
Maracatu 20M — Kaggle training runner.

Entry point para um Kaggle kernel que treina o modelo 20M usando:
    input: whereisanzi/maracatu-corpus-v1  (corpus + tokenizer)
    input: whereisanzi/maracatu-code       (model.py, data.py, train.py, config)
    output: /kaggle/working/               (tokens.npy, best.pt, latest.pt, final.pt)

Este runner replica a lógica de src/maracatu/train.py sem depender
da estrutura de diretórios do repo (que não existe no Kaggle).
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

try:
    import sentencepiece  # noqa: F401
except ImportError:
    os.system("pip install --quiet sentencepiece")
    import sentencepiece  # noqa: F401

WORK_DIR = Path("/kaggle/working")

print("=== Kaggle input mounts ===", flush=True)
input_root = Path("/kaggle/input")
mounts = sorted(input_root.glob("*")) if input_root.exists() else []
for p in mounts:
    print(f"  {p}", flush=True)
    for f in sorted(p.rglob("*"))[:15]:
        if f.is_file():
            print(f"    {f.relative_to(p)}", flush=True)
print(flush=True)

def find_containing(root: Path, expected_file: str) -> Path:
    """Procura recursivamente pelo diretório que contém `expected_file`."""
    for p in root.rglob(expected_file):
        return p.parent
    raise FileNotFoundError(f"{expected_file} não encontrado sob {root}")

CODE_DIR = find_containing(input_root, "model.py")
DATA_DIR = find_containing(input_root, "corpus.txt")
print(f"CODE_DIR resolvido: {CODE_DIR}", flush=True)
print(f"DATA_DIR resolvido: {DATA_DIR}", flush=True)
print(flush=True)

sys.path.insert(0, str(CODE_DIR))

from data import TokenBatchSampler, load_or_create_tokens, train_val_split  # noqa: E402
from model import Maracatu, ModelConfig  # noqa: E402

def get_git_revision() -> str:
    return os.environ.get("KAGGLE_GIT_REVISION", "<kaggle>")

def get_lr(step: int, cfg: dict) -> float:
    warmup = cfg["warmup_iters"]
    max_iters = cfg["max_iters"]
    lr_max = cfg["learning_rate"]
    lr_min = cfg.get("min_learning_rate", lr_max * 0.1)
    if step < warmup:
        return lr_max * (step + 1) / warmup
    if step >= max_iters:
        return lr_min
    progress = (step - warmup) / max(1, max_iters - warmup)
    coeff = 0.5 * (1.0 + np.cos(np.pi * progress))
    return lr_min + coeff * (lr_max - lr_min)

@torch.no_grad()
def estimate_loss(model, train_sampler, val_sampler, num_batches: int = 20) -> dict:
    model.eval()
    losses = {}
    for name, sampler in [("train", train_sampler), ("val", val_sampler)]:
        batch_losses = []
        for _ in range(num_batches):
            x, y = sampler.sample()
            _, loss = model(x, y)
            batch_losses.append(loss.item())
        losses[name] = float(np.mean(batch_losses))
    model.train()
    return losses

def save_checkpoint(path: Path, model, optimizer, step: int, loss: float, cfg: dict, git_rev: str):
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": step,
            "loss": loss,
            "config": cfg,
            "model_config": model.config.__dict__,
            "git_revision": git_rev,
        },
        path,
    )

def main() -> None:

    cfg_path = next(CODE_DIR.rglob("maracatu_20m.yaml"), CODE_DIR / "maracatu_20m.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    git_rev = get_git_revision()

    print("🥁 Maracatu — Treino (Kaggle)")
    print(f"  Config:  {cfg_path}")
    print(f"  Device:  {device}")
    print(f"  Run:     {cfg.get('run_name', 'unnamed')}")
    print(f"  Git rev: {git_rev}")
    if device == "cuda":
        print(f"  GPU:     {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    print()

    torch.manual_seed(cfg.get("seed", 42))
    np.random.seed(cfg.get("seed", 42))

    corpus_path = DATA_DIR / "corpus.txt"
    tokenizer_path = DATA_DIR / "maracatu.model"
    tokens_cache = WORK_DIR / "tokens.npy"

    tokens = load_or_create_tokens(corpus_path, tokenizer_path, tokens_cache)
    train_tokens, val_tokens = train_val_split(tokens, cfg.get("val_fraction", 0.005))
    print(f"  Tokens:  {len(train_tokens):,} treino / {len(val_tokens):,} validação")

    train_sampler = TokenBatchSampler(
        train_tokens, cfg["batch_size"], cfg["max_position_embeddings"], device
    )
    val_sampler = TokenBatchSampler(
        val_tokens, cfg["batch_size"], cfg["max_position_embeddings"], device
    )

    model_cfg = ModelConfig(
        vocab_size=cfg["vocab_size"],
        hidden_size=cfg["hidden_size"],
        intermediate_size=cfg["intermediate_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg.get("num_key_value_heads", cfg["num_attention_heads"]),
        max_position_embeddings=cfg["max_position_embeddings"],
        rms_norm_eps=cfg.get("rms_norm_eps", 1e-5),
        rope_theta=cfg.get("rope_theta", 10000.0),
        attention_dropout=cfg.get("attention_dropout", 0.0),
        tie_word_embeddings=cfg.get("tie_word_embeddings", True),
    )
    model = Maracatu(model_cfg).to(device)
    print(f"  Modelo:  {model.num_params() / 1e6:.2f}M parâmetros (não-embedding)")
    print()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        betas=(cfg.get("beta1", 0.9), cfg.get("beta2", 0.95)),
        weight_decay=cfg.get("weight_decay", 0.1),
    )

    ckpt_dir = WORK_DIR
    log_interval = cfg.get("log_interval", 50)
    eval_interval = cfg.get("eval_interval", 500)
    ckpt_interval = cfg.get("checkpoint_interval", 1000)
    grad_clip = cfg.get("grad_clip", 1.0)

    t_start = time.time()
    best_val = float("inf")

    for step in range(cfg["max_iters"]):
        lr = get_lr(step, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        x, y = train_sampler.sample()
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if step % log_interval == 0:
            elapsed = time.time() - t_start
            tok_per_sec = (step + 1) * cfg["batch_size"] * cfg["max_position_embeddings"] / max(elapsed, 1e-6)
            print(
                f"  step {step:6d} | loss {loss.item():.4f} | lr {lr:.2e} | "
                f"{tok_per_sec:.0f} tok/s | elapsed {elapsed / 60:.1f}min",
                flush=True,
            )

        if step % eval_interval == 0 and step > 0:
            losses = estimate_loss(model, train_sampler, val_sampler)
            print(
                f"  📊 step {step:6d} | train {losses['train']:.4f} | val {losses['val']:.4f}",
                flush=True,
            )
            if losses["val"] < best_val:
                best_val = losses["val"]
                save_checkpoint(ckpt_dir / "best.pt", model, optimizer, step, losses["val"], cfg, git_rev)
                print(f"  ✓ novo melhor checkpoint (val {best_val:.4f})", flush=True)

        if step % ckpt_interval == 0 and step > 0:
            save_checkpoint(ckpt_dir / "latest.pt", model, optimizer, step, loss.item(), cfg, git_rev)

    save_checkpoint(ckpt_dir / "final.pt", model, optimizer, cfg["max_iters"], loss.item(), cfg, git_rev)
    print()
    print(f"✓ Treino concluído. Checkpoints em {ckpt_dir}")
    print(f"  Melhor val loss: {best_val:.4f}")

if __name__ == "__main__":
    main()
