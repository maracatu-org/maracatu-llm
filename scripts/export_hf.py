"""
Exporta um checkpoint do Maracatu treinado para o formato HuggingFace
transformers (LlamaForCausalLM + LlamaTokenizer).

Nossa arquitetura já foi desenhada para ser state_dict-compatible com
LlamaForCausalLM — mesma nomenclatura de chaves, mesma convenção de
RoPE "rotate-half", mesmas ops (RMSNorm, SwiGLU). Este script:

    1. Carrega o checkpoint PyTorch (best.pt / final.pt)
    2. Constrói um LlamaConfig equivalente ao nosso ModelConfig
    3. Carrega os pesos em um LlamaForCausalLM
    4. Salva em disco em formato HF (safetensors)
    5. Copia o tokenizer SentencePiece + configs
    6. Valida equivalência numérica contra a nossa implementação

Uso:
    python scripts/export_hf.py \\
        --checkpoint checkpoints/best.pt \\
        --tokenizer tokenizer/maracatu.model \\
        --output-dir exports/maracatu-20m-hf

Depois, para publicar no HF Hub:
    huggingface-cli upload maracatu-ai/maracatu-20m exports/maracatu-20m-hf .
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
)

from maracatu.model import Maracatu, ModelConfig

def build_llama_config(cfg: ModelConfig) -> LlamaConfig:
    """Mapeia o nosso ModelConfig para LlamaConfig do transformers."""
    return LlamaConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        max_position_embeddings=cfg.max_position_embeddings,
        rms_norm_eps=cfg.rms_norm_eps,
        rope_theta=cfg.rope_theta,
        tie_word_embeddings=cfg.tie_word_embeddings,
        hidden_act="silu",
        attention_dropout=cfg.attention_dropout,
        initializer_range=0.02,
        use_cache=True,
        bos_token_id=2,
        eos_token_id=3,
        pad_token_id=0,
        attention_bias=False,
        mlp_bias=False,
    )

def load_state_dict(checkpoint: Path, tie_word_embeddings: bool) -> dict:
    """Carrega o state_dict do checkpoint e prepara para o HF."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model_state"]

    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}

    if tie_word_embeddings and "lm_head.weight" in state:
        del state["lm_head.weight"]

    return state, ckpt

def write_tokenizer_files(output_dir: Path, tokenizer_path: Path, max_len: int) -> None:
    """Copia o .model do SentencePiece e escreve os configs que o HF espera."""
    shutil.copy(tokenizer_path, output_dir / "tokenizer.model")

    tokenizer_config = {
        "tokenizer_class": "LlamaTokenizer",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "model_max_length": max_len,
        "legacy": False,
        "clean_up_tokenization_spaces": False,
        "add_bos_token": True,
        "add_eos_token": False,
    }
    (output_dir / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    special_tokens = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
    }
    (output_dir / "special_tokens_map.json").write_text(
        json.dumps(special_tokens, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    from transformers import LlamaTokenizerFast
    fast = LlamaTokenizerFast.from_pretrained(output_dir, from_slow=True)
    fast.save_pretrained(output_dir)

def verify_equivalence(
    our_model: Maracatu,
    hf_model: LlamaForCausalLM,
    vocab_size: int,
    seq_len: int = 32,
    atol: float = 1e-3,
) -> None:
    """Compara logits das duas implementações no mesmo input."""
    torch.manual_seed(0)
    ids = torch.randint(0, vocab_size, (1, seq_len))

    our_model.eval()
    hf_model.eval()
    with torch.no_grad():

        our_logits, _ = our_model(ids, targets=ids)
        hf_logits = hf_model(ids).logits

    max_abs_diff = (our_logits - hf_logits).abs().max().item()
    max_rel_diff = (
        (our_logits - hf_logits).abs() / (our_logits.abs() + 1e-8)
    ).max().item()

    print(f"  logits shape: ours={tuple(our_logits.shape)} | hf={tuple(hf_logits.shape)}")
    print(f"  max abs diff: {max_abs_diff:.2e}")
    print(f"  max rel diff: {max_rel_diff:.2e}")

    if max_abs_diff > atol:
        raise RuntimeError(
            f"Equivalência numérica falhou: max_abs_diff={max_abs_diff:.2e} > atol={atol:.2e}. "
            "Provavelmente convenção de RoPE ou ordem dos heads não bate."
        )
    print(f"  ✓ equivalência ok (< {atol:.0e})")

def main() -> None:
    parser = argparse.ArgumentParser(description="Exporta checkpoint Maracatu para HF.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--skip-equivalence",
        action="store_true",
        help="Pula a verificação numérica (útil em máquinas com pouca RAM)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"→ Carregando checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    our_cfg = ModelConfig(**ckpt["model_config"])
    state, _ = load_state_dict(args.checkpoint, our_cfg.tie_word_embeddings)

    print(f"  step: {ckpt.get('step')} | val loss: {ckpt.get('loss'):.4f}")
    git_rev = ckpt.get("git_revision", "<unknown>")
    print(f"  git: {git_rev}")

    print("→ Construindo LlamaConfig equivalente...")
    llama_cfg = build_llama_config(our_cfg)

    print("→ Instanciando LlamaForCausalLM e carregando pesos...")
    hf_model = LlamaForCausalLM(llama_cfg)
    missing, unexpected = hf_model.load_state_dict(state, strict=False)

    real_missing = [k for k in missing if not (
        "rotary_emb" in k or k.endswith(".inv_freq")
    )]
    if real_missing:
        print(f"  ⚠ chaves ausentes no state_dict: {real_missing}")
    if unexpected:
        print(f"  ⚠ chaves inesperadas: {unexpected}")
    if not real_missing and not unexpected:
        print("  ✓ todas as chaves bateram")

    hf_model.eval()

    if not args.skip_equivalence:
        print("\n→ Verificando equivalência numérica com nossa impl...")
        our_model = Maracatu(our_cfg)
        our_model.load_state_dict(ckpt["model_state"], strict=True)
        verify_equivalence(our_model, hf_model, vocab_size=our_cfg.vocab_size)

    print(f"\n→ Salvando pesos em {args.output_dir}")
    hf_model.save_pretrained(args.output_dir, safe_serialization=True)

    print("→ Copiando tokenizer + configs...")
    write_tokenizer_files(args.output_dir, args.tokenizer, our_cfg.max_position_embeddings)

    print("\n→ Sanity check final via AutoModel/AutoTokenizer...")
    loaded_model = AutoModelForCausalLM.from_pretrained(args.output_dir)
    loaded_tok = AutoTokenizer.from_pretrained(args.output_dir)
    prompt = "O Brasil é"
    ids = loaded_tok(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        out = loaded_model.generate(ids, max_new_tokens=20, do_sample=False)
    print(f"  prompt: {prompt!r}")
    print(f"  output: {loaded_tok.decode(out[0], skip_special_tokens=True)!r}")

    print(f"\n✓ Export completo em {args.output_dir}")
    print("\nPara publicar no HuggingFace Hub:")
    print(f"  huggingface-cli upload maracatu-ai/maracatu-20m {args.output_dir} .")
    print("\nPara testar via transformers:")
    print(f"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("{args.output_dir}")
    model = AutoModelForCausalLM.from_pretrained("{args.output_dir}")
    ids = tok("O Brasil é", return_tensors="pt").input_ids
    out = model.generate(ids, max_new_tokens=50, temperature=0.8, top_k=50, do_sample=True)
    print(tok.decode(out[0]))
""")

if __name__ == "__main__":
    main()
