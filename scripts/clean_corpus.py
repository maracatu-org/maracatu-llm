"""
Preparação do corpus do Maracatu a partir da Wikipedia em português.

Usa o dataset `wikimedia/wikipedia` do HuggingFace, que fornece a Wikipedia
PT já extraída e limpa em arquivos parquet. Aplica heurísticas adicionais
de filtro e deduplicação e escreve um único `data/processed/corpus.txt`
pronto para treinar o tokenizer.

O dataset é baixado uma vez para o cache do HuggingFace
(~/.cache/huggingface/datasets/) e reutilizado em execuções seguintes.

Heurísticas de limpeza:
    - Remove documentos muito curtos (< 200 caracteres)
    - Remove linhas muito curtas (< 30 caracteres)
    - Remove linhas que são só pontuação/números
    - Deduplica parágrafos idênticos (hash SHA-1)
    - Normaliza espaços em branco

Uso:
    python scripts/clean_corpus.py
    python scripts/clean_corpus.py --wiki-version 20231101.pt
    python scripts/clean_corpus.py --min-doc-chars 500 --min-line-chars 50
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "corpus.txt"
MANIFEST_FILE = OUTPUT_DIR / "MANIFEST.txt"

WHITESPACE_RE = re.compile(r"\s+")
ONLY_SYMBOLS_RE = re.compile(r"^[\W\d_]+$", re.UNICODE)

def clean_line(line: str) -> str:
    """Normaliza espaços e remove caracteres de controle."""
    line = line.strip()
    line = WHITESPACE_RE.sub(" ", line)
    return line

def is_valid_line(line: str, min_chars: int) -> bool:
    """Verifica se uma linha é conteúdo textual utilizável."""
    if len(line) < min_chars:
        return False
    if ONLY_SYMBOLS_RE.match(line):
        return False
    return True

def line_hash(line: str) -> str:
    """Hash SHA-1 para deduplicação."""
    return hashlib.sha1(line.encode("utf-8")).hexdigest()

def sha256_file(path: Path, buf_size: int = 1024 * 1024) -> str:
    """SHA-256 do arquivo, em streaming para não carregar tudo em RAM."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(buf_size):
            h.update(chunk)
    return h.hexdigest()

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepara o corpus do Maracatu.")
    parser.add_argument(
        "--wiki-version",
        default="20231101.pt",
        help="Config do dataset wikimedia/wikipedia (padrão: 20231101.pt)",
    )
    parser.add_argument("--min-doc-chars", type=int, default=200)
    parser.add_argument("--min-line-chars", type=int, default=30)
    parser.add_argument("--no-dedupe", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("🥁 Maracatu — Preparação do corpus")
    print(f"  Dataset: wikimedia/wikipedia ({args.wiki_version})")
    print(f"  Saída:   {OUTPUT_FILE}")
    print()
    print("→ Carregando dataset do HuggingFace (download na primeira execução)...")

    ds = load_dataset("wikimedia/wikipedia", args.wiki_version, split="train")
    print(f"  Artigos: {len(ds):,}")
    print()

    dedupe = not args.no_dedupe
    stats = {
        "docs_total": 0,
        "docs_kept": 0,
        "lines_total": 0,
        "lines_kept": 0,
        "lines_deduped": 0,
        "chars_written": 0,
    }
    seen_hashes: set[str] = set()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for doc in tqdm(ds, desc="Processando", unit="doc", total=len(ds)):
            stats["docs_total"] += 1
            text = doc.get("text", "")

            if len(text) < args.min_doc_chars:
                continue

            kept_lines: list[str] = []
            for raw_line in text.split("\n"):
                stats["lines_total"] += 1
                line = clean_line(raw_line)
                if not is_valid_line(line, args.min_line_chars):
                    continue
                if dedupe:
                    h = line_hash(line)
                    if h in seen_hashes:
                        stats["lines_deduped"] += 1
                        continue
                    seen_hashes.add(h)
                kept_lines.append(line)
                stats["lines_kept"] += 1

            if not kept_lines:
                continue

            stats["docs_kept"] += 1
            block = "\n".join(kept_lines) + "\n\n"
            out.write(block)
            stats["chars_written"] += len(block)

    print()
    print("📊 Estatísticas:")
    print(f"  Documentos: {stats['docs_kept']:,} / {stats['docs_total']:,}")
    print(f"  Linhas:     {stats['lines_kept']:,} / {stats['lines_total']:,}")
    print(f"  Duplicadas: {stats['lines_deduped']:,}")
    print(f"  Caracteres: {stats['chars_written']:,}")
    print(f"  MB:         {stats['chars_written'] / 1_000_000:.1f}")
    print()

    print("→ Calculando SHA-256 do corpus...")
    corpus_sha256 = sha256_file(OUTPUT_FILE)

    with open(MANIFEST_FILE, "w", encoding="utf-8") as manifest:
        manifest.write("Maracatu — MANIFEST do corpus processado\n")
        manifest.write("=" * 50 + "\n\n")
        manifest.write(f"Fonte: wikimedia/wikipedia ({args.wiki_version})\n")
        manifest.write("Licença: CC BY-SA 4.0\n")
        manifest.write(f"SHA-256: {corpus_sha256}\n\n")
        for key, value in stats.items():
            manifest.write(f"{key}: {value:,}\n")
        manifest.write(
            f"\nParâmetros: min_doc_chars={args.min_doc_chars}, "
            f"min_line_chars={args.min_line_chars}, dedupe={dedupe}\n"
        )

    print(f"✓ Corpus salvo em {OUTPUT_FILE}")
    print(f"✓ Manifest em {MANIFEST_FILE}")
    print()
    print("Próximo passo: python tokenizer/train_tokenizer.py")

if __name__ == "__main__":
    main()
