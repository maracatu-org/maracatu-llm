"""
Loop de treino do Maracatu.

Carrega config YAML, prepara dados, treina o modelo salvando checkpoints
periódicos e avalia em validação a cada N iterações.

Suporta:
    - CPU (debug)
    - CUDA (GPU NVIDIA, nuvem)
    - MPS (Apple Silicon, Mac M1/M2/M3)

Uso:
    python -m maracatu.train --config configs/maracatu_20m.yaml
    python -m maracatu.train --config configs/maracatu_20m.yaml --device cuda
    python -m maracatu.train --config configs/maracatu_20m.yaml --resume
    python -m maracatu.train --config configs/maracatu_20m.yaml --no-resume
"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from maracatu.data import TokenBatchSampler, load_or_create_tokens, train_val_split
from maracatu.model import Maracatu, ModelConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_RESUME_FIELDS = frozenset(
    ["rng_torch", "rng_numpy", "rng_python"]
)

def get_device(requested: str | None = None) -> str:
    """Detecta automaticamente o melhor device disponível."""
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_lr(step: int, cfg: dict) -> float:
    """Cosine schedule com warmup linear."""
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

def _autocast_ctx(device: str, use_bf16: bool):
    """Retorna contexto de autocast bf16 em CUDA, ou nullcontext em CPU/MPS.

    bf16 tem mesmo range que fp32 (nao precisa GradScaler) e dobra throughput
    aproximado em Ampere (RTX 30xx, A100). Llama-3 treina bf16.
    """
    import contextlib
    if use_bf16 and device == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()

@torch.no_grad()
def estimate_loss(
    model: Maracatu,
    train_sampler: TokenBatchSampler,
    val_sampler: TokenBatchSampler,
    device: str,
    use_bf16: bool,
    num_batches: int = 20,
) -> dict[str, float]:
    """Estima loss média em treino e validação."""
    model.eval()
    losses: dict[str, float] = {}
    for name, sampler in [("train", train_sampler), ("val", val_sampler)]:
        batch_losses = []
        for _ in range(num_batches):
            x, y = sampler.sample()
            with _autocast_ctx(device, use_bf16):
                _, loss = model(x, y)
            batch_losses.append(loss.item())
        losses[name] = float(np.mean(batch_losses))
    model.train()
    return losses

def get_git_revision() -> str:
    """Commit hash atual do repositório, ou 'unknown' se não disponível."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        suffix = "-dirty" if dirty.stdout.strip() else ""
        return result.stdout.strip() + suffix
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"

def _capture_rng_state() -> dict:
    """Captura estado completo dos tres geradores de numeros aleatorios usados no treino."""
    return {
        "rng_torch": torch.get_rng_state(),

        "rng_torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "rng_numpy": np.random.get_state(),
        "rng_python": random.getstate(),
    }

def _restore_rng_state(rng: dict) -> None:
    """Restaura estado dos geradores a partir de dict salvo por _capture_rng_state.

    torch.load(map_location="cuda") move tensors para GPU; o RNG state precisa
    voltar para CPU como ByteTensor antes de ser aplicado.
    """
    rng_torch = rng["rng_torch"].cpu().to(torch.uint8)
    torch.set_rng_state(rng_torch)
    if torch.cuda.is_available() and rng.get("rng_torch_cuda"):
        cuda_states = [s.cpu().to(torch.uint8) for s in rng["rng_torch_cuda"]]
        torch.cuda.set_rng_state_all(cuda_states)
    np.random.set_state(rng["rng_numpy"])
    random.setstate(rng["rng_python"])

def save_checkpoint(
    path: Path,
    model: Maracatu,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    config: dict,
    git_revision: str,
) -> None:
    """Salva checkpoint simples (best.pt / final.pt). API original preservada."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": step,
            "loss": loss,
            "config": config,
            "model_config": model.config.__dict__,
            "git_revision": git_revision,
        },
        path,
    )

def save_checkpoint_resume_safe(
    path: Path,
    model: Maracatu,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    config: dict,
    git_revision: str,
) -> None:
    """Salva checkpoint com state completo para resume exato.

    Inclui RNG states (torch / numpy / python) para reprodutibilidade deterministica.
    Escreve atomicamente via arquivo temporario + os.replace para nunca corromper
    o checkpoint se o processo morrer no meio do save (ex: preempcao em spot).

    O arquivo resultante e compativel com load_checkpoint_for_resume() e tambem
    com save_checkpoint() / torch.load() direto (campos extras sao ignorados).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".pt.tmp")

    payload = {

        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": step,
        "loss": loss,
        "config": config,
        "model_config": model.config.__dict__,
        "git_revision": git_revision,

        **_capture_rng_state(),
    }

    torch.save(payload, tmp_path)

    os.replace(tmp_path, path)

def load_checkpoint_for_resume(
    path: Path,
    model: Maracatu,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> int:
    """Carrega checkpoint para resume, retorna o proximo step a executar.

    Compativel com checkpoints legados (pre-resume-safe): se RNG states nao
    estiverem presentes, loga aviso e continua sem restaurar RNG. A sequencia
    de tokens sorteados sera diferente, mas o modelo e otimizador estarao corretos.

    Args:
        path: Caminho do arquivo .pt a carregar.
        model: Modelo ja instanciado e movido pro device correto.
        optimizer: Otimizador ja instanciado.
        device: Device string para map_location.

    Returns:
        Proximo step (ckpt["step"] + 1).
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    resumed_step = ckpt["step"] + 1

    missing = _RESUME_FIELDS - set(ckpt.keys())
    if missing:
        print(
            f"[resume-safe] aviso: checkpoint legado detectado, "
            f"campos ausentes: {sorted(missing)}. "
            f"RNG state NAO restaurado; sequencia de amostras sera diferente."
        )
    else:
        _restore_rng_state(ckpt)
        print("[resume-safe] RNG state restaurado (torch + numpy + python).")

    return resumed_step

def maybe_auto_resume(
    checkpoint_dir: Path,
    model: Maracatu,
    optimizer: torch.optim.Optimizer,
    device: str,
    force_resume: bool,
    force_no_resume: bool,
) -> int:
    """Tenta auto-resume a partir de latest.pt no checkpoint_dir.

    Logica de precedencia:
      --no-resume  -> comeca do zero, ignora latest.pt mesmo se existir.
      --resume     -> forca resume de latest.pt; falha se nao existir.
      default      -> se latest.pt existe, auto-resume silencioso.

    Args:
        checkpoint_dir: Diretorio onde latest.pt e procurado.
        model: Modelo ja instanciado.
        optimizer: Otimizador ja instanciado.
        device: Device string.
        force_resume: True quando --resume foi passado na CLI.
        force_no_resume: True quando --no-resume foi passado na CLI.

    Returns:
        start_step: Proximo step a executar (0 se comecando do zero).
    """
    latest = checkpoint_dir / "latest.pt"

    if force_no_resume:
        if latest.exists():
            print("[resume-safe] --no-resume: ignorando latest.pt existente, comecando do zero.")
        return 0

    if force_resume:
        if not latest.exists():
            raise FileNotFoundError(
                f"[resume-safe] --resume solicitado mas {latest} nao existe."
            )
        start_step = load_checkpoint_for_resume(latest, model, optimizer, device)
        print(f"[resume-safe] retomando de step {start_step - 1} -> proximo step {start_step}.")
        return start_step

    if latest.exists():
        start_step = load_checkpoint_for_resume(latest, model, optimizer, device)
        print(
            f"[resume-safe] latest.pt encontrado, retomando de step {start_step - 1} "
            f"-> proximo step {start_step}."
        )
        return start_step

    return 0

def main() -> None:
    parser = argparse.ArgumentParser(description="Treina o Maracatu.")
    parser.add_argument("--config", type=Path, required=True, help="Arquivo YAML de config")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Forca resume de latest.pt (falha se nao existir).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        default=False,
        dest="no_resume",
        help="Forca inicio do zero, ignorando latest.pt mesmo se existir.",
    )
    parser.add_argument("--compile", action="store_true", help="Usa torch.compile (CUDA)")
    args = parser.parse_args()

    if args.resume and args.no_resume:
        parser.error("--resume e --no-resume sao mutuamente exclusivos.")

    cfg = load_config(args.config)
    device = get_device(args.device)

    git_rev = get_git_revision()

    print("Maracatu -- Treino")
    print(f"  Config:  {args.config}")
    print(f"  Device:  {device}")
    print(f"  Run:     {cfg.get('run_name', 'unnamed')}")
    print(f"  Git rev: {git_rev}")
    print()

    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    corpus_path = PROJECT_ROOT / cfg["corpus_path"]
    tokenizer_path = PROJECT_ROOT / cfg["tokenizer_path"]
    tokens_cache = PROJECT_ROOT / "data" / "processed" / "tokens.npy"
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
        rms_norm_eps=cfg.get("rms_norm_eps", 1.0e-5),
        rope_theta=cfg.get("rope_theta", 10000.0),
        attention_dropout=cfg.get("attention_dropout", 0.0),
        tie_word_embeddings=cfg.get("tie_word_embeddings", True),
    )
    model = Maracatu(model_cfg).to(device)
    print(f"  Modelo:  {model.num_params() / 1e6:.2f}M parâmetros (sem pos emb)")
    print()

    if args.compile and device == "cuda":
        print("  -> Aplicando torch.compile...")
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        betas=(cfg.get("beta1", 0.9), cfg.get("beta2", 0.95)),
        weight_decay=cfg.get("weight_decay", 0.1),
    )

    ckpt_dir = PROJECT_ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    start_step = maybe_auto_resume(
        checkpoint_dir=ckpt_dir,
        model=model,
        optimizer=optimizer,
        device=device,
        force_resume=args.resume,
        force_no_resume=args.no_resume,
    )

    eval_interval = cfg.get("eval_interval", 500)
    log_interval = cfg.get("log_interval", 50)
    ckpt_interval = cfg.get("checkpoint_interval", 1000)

    wallclock_interval_min: float = cfg.get("checkpoint_interval_minutes", 30.0)
    wallclock_interval_sec: float = wallclock_interval_min * 60.0

    use_bf16: bool = cfg.get("use_bf16_autocast", True) and device == "cuda"
    if use_bf16:
        print("  -> bf16 autocast ativado (forward em bf16, weights e optimizer em fp32)")

    t_start = time.time()
    t_last_wallclock_save = t_start
    best_val = float("inf")

    for step in range(start_step, cfg["max_iters"]):
        lr = get_lr(step, cfg)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        x, y = train_sampler.sample()
        with _autocast_ctx(device, use_bf16):
            _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        grad_clip = cfg.get("grad_clip", 1.0)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        if step % log_interval == 0:
            elapsed = time.time() - t_start
            tokens_processed = (
                (step - start_step + 1) * cfg["batch_size"] * cfg["max_position_embeddings"]
            )
            tok_per_sec = tokens_processed / max(elapsed, 1e-6)
            print(
                f"  step {step:6d} | loss {loss.item():.4f} | "
                f"lr {lr:.2e} | {tok_per_sec:.0f} tok/s | "
                f"elapsed {elapsed / 60:.1f}min"
            )

        if step % eval_interval == 0 and step > 0:
            losses = estimate_loss(model, train_sampler, val_sampler, device, use_bf16)
            print(
                f"  step {step:6d} | "
                f"train {losses['train']:.4f} | val {losses['val']:.4f}"
            )
            if losses["val"] < best_val:
                best_val = losses["val"]
                save_checkpoint(
                    ckpt_dir / "best.pt", model, optimizer, step, losses["val"], cfg, git_rev
                )
                print(f"  novo melhor checkpoint (val {best_val:.4f})")

        if step % ckpt_interval == 0 and step > 0:
            save_checkpoint_resume_safe(
                ckpt_dir / "latest.pt", model, optimizer, step, loss.item(), cfg, git_rev
            )

        now = time.time()
        if now - t_last_wallclock_save >= wallclock_interval_sec:
            mins_since_last = (now - t_last_wallclock_save) / 60.0
            save_checkpoint_resume_safe(
                ckpt_dir / "latest.pt", model, optimizer, step, loss.item(), cfg, git_rev
            )
            t_last_wallclock_save = now
            print(
                f"[resume-safe] wallclock checkpoint salvo em step {step} "
                f"(last save {mins_since_last:.1f}min atras)."
            )

    save_checkpoint(
        ckpt_dir / "final.pt", model, optimizer, cfg["max_iters"], loss.item(), cfg, git_rev
    )
    print()
    print(f"Treino concluido. Checkpoints em {ckpt_dir}")
    print(f"  Melhor val loss: {best_val:.4f}")

if __name__ == "__main__":
    main()
