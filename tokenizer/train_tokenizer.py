"""
Treino do tokenizer SentencePiece para o Maracatu.

Treina um tokenizer BPE especializado em português brasileiro a partir do
corpus limpo em data/processed/corpus.txt. O modelo resultante é salvo em
tokenizer/maracatu.model (usado pelo data loader e pelo sampling).

Uso:
    python tokenizer/train_tokenizer.py
    python tokenizer/train_tokenizer.py --vocab-size 32000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import sentencepiece as spm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_FILE = PROJECT_ROOT / "data" / "processed" / "corpus.txt"
TOKENIZER_DIR = PROJECT_ROOT / "tokenizer"
MODEL_PREFIX = TOKENIZER_DIR / "maracatu"

def main() -> None:
    parser = argparse.ArgumentParser(description="Treina o tokenizer do Maracatu.")
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=16000,
        help="Tamanho do vocabulário (padrão: 16000)",
    )
    parser.add_argument(
        "--model-type",
        choices=["bpe", "unigram"],
        default="bpe",
        help="Algoritmo de tokenização (padrão: bpe)",
    )
    parser.add_argument(
        "--character-coverage",
        type=float,
        default=0.9995,
        help="Cobertura de caracteres (padrão: 0.9995 — bom para PT)",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=8,
        help="Número de threads (padrão: 8)",
    )
    parser.add_argument(
        "--input-sentence-size",
        type=int,
        default=10_000_000,
        help=(
            "Máximo de sentenças amostradas do corpus para treinar o BPE "
            "(padrão: 10M — suficiente para vocab de 16k e cabe em 16GB RAM). "
            "Use 0 para usar o corpus inteiro."
        ),
    )
    args = parser.parse_args()

    if not CORPUS_FILE.exists():
        raise SystemExit(
            f"✗ Corpus não encontrado em {CORPUS_FILE}.\n"
            "  Execute antes: python scripts/clean_corpus.py"
        )

    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

    corpus_size_mb = CORPUS_FILE.stat().st_size / 1_000_000

    print("🥁 Maracatu — Treino do tokenizer SentencePiece")
    print(f"  Corpus:           {CORPUS_FILE} ({corpus_size_mb:.1f} MB)")
    print(f"  Tamanho do vocab: {args.vocab_size:,}")
    print(f"  Tipo:             {args.model_type}")
    print(f"  Cobertura:        {args.character_coverage}")
    print(f"  Saída:            {MODEL_PREFIX}.model")
    print()

    spm.SentencePieceTrainer.train(
        input=str(CORPUS_FILE),
        model_prefix=str(MODEL_PREFIX),
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
        num_threads=args.num_threads,

        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",

        normalization_rule_name="nmt_nfkc_cf",
        remove_extra_whitespaces=True,

        split_digits=True,
        byte_fallback=True,

        input_sentence_size=args.input_sentence_size,
        shuffle_input_sentence=args.input_sentence_size > 0,
    )

    print()
    print(f"✓ Tokenizer salvo em {MODEL_PREFIX}.model")
    print()

    sp = spm.SentencePieceProcessor(model_file=f"{MODEL_PREFIX}.model")
    amostras = [
        "O Brasil é um país de dimensões continentais.",
        "Machado de Assis escreveu Dom Casmurro.",
        "A capital de São Paulo é São Paulo.",
        "Maracatu é um gênero musical pernambucano.",
    ]
    print("🧪 Testes de tokenização:")
    for texto in amostras:
        tokens = sp.encode(texto, out_type=str)
        ids = sp.encode(texto)
        print(f"  Texto:  {texto}")
        print(f"  Tokens ({len(tokens)}): {tokens}")
        print(f"  IDs:    {ids}")
        print(f"  Decoded: {sp.decode(ids)}")
        print()

    print(f"Vocabulário final: {sp.vocab_size():,} tokens")
    print("\nPróximo passo: python -m maracatu.train --config configs/maracatu_20m.yaml")

if __name__ == "__main__":
    main()
