"""
Data loader do Maracatu.

Tokeniza o corpus com o tokenizer SentencePiece treinado e gera batches
de sequências contíguas para o treino.

Estratégia:
    1. Tokeniza o corpus inteiro em memória (ok para corpora até alguns GB)
    2. Salva como um único array numpy (uint16 se vocab <= 65k)
    3. Amostra batches aleatórios de (batch_size, context_size + 1) e
       usa os primeiros context_size tokens como input e os últimos como target
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def tokenize_corpus(
    corpus_path: Path,
    tokenizer_path: Path,
    output_path: Path,
    chunk_size: int = 10_000,
) -> np.ndarray:
    """Tokeniza o corpus inteiro e salva como array numpy.

    Args:
        corpus_path: Caminho para o corpus.txt.
        tokenizer_path: Caminho para o arquivo .model do SentencePiece.
        output_path: Onde salvar o array .npy tokenizado.
        chunk_size: Linhas a processar por batch.

    Returns:
        Array numpy com todos os tokens concatenados.
    """
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    eos_id = sp.eos_id()
    vocab_size = sp.vocab_size()
    dtype = np.uint16 if vocab_size < 2**16 else np.uint32

    chunks: list[np.ndarray] = []
    buffer: list[str] = []

    def flush_buffer() -> None:
        if not buffer:
            return
        batch_ids = sp.encode(buffer)
        flat: list[int] = []
        for ids in batch_ids:
            flat.extend(ids)
            flat.append(eos_id)
        chunks.append(np.asarray(flat, dtype=dtype))
        buffer.clear()

    with open(corpus_path, encoding="utf-8") as f:
        for line in tqdm(f, desc="Tokenizando corpus", unit="linha"):
            line = line.strip()
            if not line:
                continue
            buffer.append(line)
            if len(buffer) >= chunk_size:
                flush_buffer()
        flush_buffer()

    arr = np.concatenate(chunks) if chunks else np.array([], dtype=dtype)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, arr)
    print(f"✓ {len(arr):,} tokens salvos em {output_path}")
    return arr

def load_or_create_tokens(
    corpus_path: Path,
    tokenizer_path: Path,
    cache_path: Path,
) -> np.ndarray:
    """Carrega tokens do cache ou tokeniza o corpus se não existir."""
    if cache_path.exists():
        print(f"→ Carregando tokens de {cache_path}")
        return np.load(cache_path)
    print(f"→ Tokenizando corpus (cache em {cache_path})")
    return tokenize_corpus(corpus_path, tokenizer_path, cache_path)

def train_val_split(tokens: np.ndarray, val_fraction: float = 0.005) -> tuple[np.ndarray, np.ndarray]:
    """Divide tokens em treino/validação. Val pequena por padrão — só pra monitorar."""
    split_idx = int(len(tokens) * (1 - val_fraction))
    return tokens[:split_idx], tokens[split_idx:]

class TokenBatchSampler:
    """Amostra batches de sequências contíguas a partir de um array de tokens."""

    def __init__(
        self,
        tokens: np.ndarray,
        batch_size: int,
        context_size: int,
        device: str = "cpu",
    ) -> None:
        if len(tokens) < context_size + 2:
            raise ValueError(
                f"Corpus tokenizado ({len(tokens)} tokens) é menor que "
                f"context_size ({context_size}) + 2."
            )
        self.tokens = tokens
        self.batch_size = batch_size
        self.context_size = context_size
        self.device = device

    def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Retorna (inputs, targets) — targets são inputs deslocados em 1 posição."""
        max_start = len(self.tokens) - self.context_size - 1
        starts = np.random.randint(0, max_start, size=self.batch_size)

        inputs = np.stack([self.tokens[s : s + self.context_size] for s in starts])
        targets = np.stack([self.tokens[s + 1 : s + 1 + self.context_size] for s in starts])

        inputs_t = torch.from_numpy(inputs.astype(np.int64))
        targets_t = torch.from_numpy(targets.astype(np.int64))

        if self.device != "cpu":
            inputs_t = inputs_t.to(self.device, non_blocking=True)
            targets_t = targets_t.to(self.device, non_blocking=True)

        return inputs_t, targets_t
