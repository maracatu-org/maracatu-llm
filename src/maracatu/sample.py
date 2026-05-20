"""
Geração de texto com o Maracatu treinado.

Uso:
    python -m maracatu.sample --checkpoint checkpoints/best.pt --prompt "O Brasil é"
    python -m maracatu.sample --checkpoint checkpoints/best.pt --prompt "Machado de Assis" \\
        --max-new-tokens 200 --temperature 0.8 --top-k 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import sentencepiece as spm
import torch

from maracatu.model import Maracatu, ModelConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def get_device(requested: str | None = None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def load_model(checkpoint_path: Path, device: str) -> tuple[Maracatu, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_cfg = ModelConfig(**ckpt["model_config"])
    model = Maracatu(model_cfg).to(device)

    state_dict = ckpt["model_state"]
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model, ckpt

def main() -> None:
    parser = argparse.ArgumentParser(description="Gera texto com o Maracatu.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=PROJECT_ROOT / "tokenizer" / "maracatu.model")
    parser.add_argument("--prompt", type=str, default="O Brasil é")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    device = get_device(args.device)
    if args.seed is not None:
        torch.manual_seed(args.seed)

    print(f"🥁 Maracatu — Geração de texto ({device})")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Prompt:     {args.prompt!r}")
    print(f"  Temperatura: {args.temperature} | top-k: {args.top_k}")
    print()

    model, ckpt = load_model(args.checkpoint, device)
    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))

    prompt_ids = sp.encode(args.prompt)
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    for i in range(args.num_samples):
        output_ids = model.generate(
            prompt_tensor,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        output_text = sp.decode(output_ids[0].tolist())
        print(f"— Amostra {i + 1} " + "-" * 40)
        print(output_text)
        print()

if __name__ == "__main__":
    main()
