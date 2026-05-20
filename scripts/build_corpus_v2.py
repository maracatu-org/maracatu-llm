"""
Build corpus v2 do Maracatu a partir de tres fontes abertas em PT-BR.

Fontes incluidas:
    1. Wikipedia PT (CC BY-SA 3.0) -- wikimedia/wikipedia, config 20231101.pt
    2. Project Gutenberg PT (Dominio Publico) -- scraping direto de IDs curados
    3. CulturaX-PT (ODC-BY) -- uonlp/CulturaX, subset pt, streaming

Fonte EXCLUIDA intencionalmente:
    - Corpus Carolina (CC BY-NC-4.0): restricao "nao comercial" e incompativel
      com a licenca Apache 2.0 do Maracatu e com qualquer plano de comercializacao
      futura. Reativar aqui somente se o escopo mudar para pesquisa sem fins
      comerciais e mediante confirmacao legal explicita.

Pipeline de filtros (em cadeia, por documento):
    1. min_doc_chars: descarta documentos muito curtos
    2. detect_portuguese: heuristica por vocabulario de stopwords
    3. limpeza linha a linha: min_line_chars + so-simbolos
    4. PII regex: remove linhas com CPF, email, telefone BR, CEP
    5. SHA-1 exata: dedup de linhas identicas dentro e entre documentos
    6. MinHash LSH: dedup fuzzy entre documentos (Jaccard >= threshold)

Saida:
    data/processed/corpus_v2.txt  -- um paragrafo logico por linha
    data/processed/MANIFEST_v2.txt -- SHA-256 + stats + parametros
    data/processed/stats_v2.json  -- stats legivel por maquina

Uso:
    python scripts/build_corpus_v2.py --source all
    python scripts/build_corpus_v2.py --source wikipedia --smoke-test
    python scripts/build_corpus_v2.py --source gutenberg --max-docs-per-source 50
    python scripts/build_corpus_v2.py --source culturax --max-docs-per-source 5000

Determinismo:
    Dado o mesmo conjunto de IDs Gutenberg e versoes fixas dos datasets HF,
    duas execucoes produzem SHA-256 identico. A ordem de escrita e deterministica:
    wikipedia -> gutenberg -> culturax.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "corpus_v2.txt"
MANIFEST_FILE = OUTPUT_DIR / "MANIFEST_v2.txt"
STATS_FILE = OUTPUT_DIR / "stats_v2.json"

SOURCE_META: dict[str, dict[str, str]] = {
    "wikipedia": {
        "nome": "Wikipedia PT",
        "hf_dataset": "wikimedia/wikipedia",
        "hf_config": "20231101.pt",
        "licenca": "CC BY-SA 3.0",
        "compativel_apache2": "sim",
        "url": "https://huggingface.co/datasets/wikimedia/wikipedia",
    },
    "gutenberg": {
        "nome": "Project Gutenberg PT",
        "url": "https://www.gutenberg.org/",
        "licenca": "Dominio Publico",
        "compativel_apache2": "sim",
        "nota": "Obras com autores falecidos ha mais de 70 anos (criterio BR/internacional)",
    },
    "culturax": {
        "nome": "CulturaX PT",
        "hf_dataset": "uonlp/CulturaX",
        "hf_config": "pt",
        "licenca": "ODC-BY 1.0",
        "compativel_apache2": "sim",
        "url": "https://huggingface.co/datasets/uonlp/CulturaX",
        "nota": (
            "ODC-BY exige atribuicao da fonte original. A atribuicao esta registrada "
            "neste MANIFEST e no data/README.md."
        ),
    },

}

GUTENBERG_PT_IDS: list[tuple[int, str, str]] = [

    (55752, "Machado de Assis", "Dom Casmurro"),
    (67752, "Machado de Assis", "Memorias Postumas de Bras Cubas"),
    (55682, "Machado de Assis", "Quincas Borba"),
    (55388, "Machado de Assis", "Helena"),
    (16370, "Jose de Alencar", "Iracema"),
    (36477, "Jose de Alencar", "O Guarani"),
    (29780, "Jose de Alencar", "Senhora"),
    (37993, "Aluisio Azevedo", "O Cortico"),
    (39065, "Aluisio Azevedo", "O Mulato"),
    (34964, "Euclides da Cunha", "Os Sertoes"),
    (14145, "Castro Alves", "Espumas Flutuantes"),
    (37558, "Alvares de Azevedo", "Noite na Taverna"),
    (35588, "Casimiro de Abreu", "As Primaveras"),
    (58041, "Eca de Queiros", "O Crime do Padre Amaro"),
    (55417, "Eca de Queiros", "A Reliquia"),
    (15925, "Eca de Queiros", "Os Maias"),
    (30765, "Graciliano Ramos", "Vidas Secas"),
    (62193, "Lima Barreto", "Triste Fim de Policarpo Quaresma"),
    (55405, "Lima Barreto", "O Cemiterio dos Vivos"),
    (38413, "Monteiro Lobato", "Urupes"),
    (53051, "Olavo Bilac", "Poesias"),
    (14536, "Goncalves Dias", "Primeiros Cantos"),
    (59417, "Raul Pompeia", "O Ateneu"),
    (38869, "Visconde de Taunay", "Inocencia"),
]

GUTENBERG_TEXT_URL = "https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"

GUTENBERG_START_MARKERS = [
    "*** START OF THE PROJECT GUTENBERG EBOOK",
    "***START OF THE PROJECT GUTENBERG EBOOK",
    "*** START OF THIS PROJECT GUTENBERG EBOOK",
]
GUTENBERG_END_MARKERS = [
    "*** END OF THE PROJECT GUTENBERG EBOOK",
    "***END OF THE PROJECT GUTENBERG EBOOK",
    "*** END OF THIS PROJECT GUTENBERG EBOOK",
]

PT_STOPWORDS: frozenset[str] = frozenset(
    [
        "de", "a", "o", "que", "e", "do", "da", "em", "um", "para", "com",
        "uma", "os", "no", "se", "na", "por", "mais", "as", "dos", "como",
        "mas", "ao", "ele", "das", "seu", "sua", "ou", "quando", "muito",
        "nos", "ja", "eu", "tambem", "so", "pelo", "pela", "ate", "isso",
        "ela", "entre", "depois", "sem", "mesmo", "aos", "seus", "quem",
        "nas", "me", "esse", "eles", "voce", "essa", "num", "nem", "suas",
        "meu", "minha", "numa", "pelos", "pelas", "foi", "ha", "esta",
        "ter", "ser", "tem", "nao", "pode", "fazer", "sobre",
    ]
)

PII_PATTERNS: list[re.Pattern[str]] = [

    re.compile(r"\b\d{3}[\.\-]?\d{3}[\.\-]?\d{3}[\.\-]?\d{2}\b"),

    re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),

    re.compile(r"(\+55\s?)?(\(?\d{2}\)?\s?)?\d{4,5}[\s\-]?\d{4}\b"),

    re.compile(r"\b\d{5}-\d{3}\b"),

    re.compile(r"\b(Rua|Av\.|Avenida|Alameda|Travessa|Estrada|Rod\.)\s+\w+", re.IGNORECASE),
]

WHITESPACE_RE: re.Pattern[str] = re.compile(r"\s+")
ONLY_SYMBOLS_RE: re.Pattern[str] = re.compile(r"^[\W\d_]+$", re.UNICODE)

@dataclass
class FilterConfig:
    """Thresholds e parametros de filtro."""
    min_doc_chars: int = 200
    min_line_chars: int = 30
    pt_stopword_min_ratio: float = 0.05
    minhash_threshold: float = 0.85
    minhash_num_perm: int = 128

@dataclass
class PathConfig:
    """Paths de saida."""
    output_file: Path = field(default_factory=lambda: OUTPUT_FILE)
    manifest_file: Path = field(default_factory=lambda: MANIFEST_FILE)
    stats_file: Path = field(default_factory=lambda: STATS_FILE)

@dataclass
class RunStats:
    """Contadores acumulados por fonte e total."""

    wikipedia_docs_total: int = 0
    wikipedia_docs_kept: int = 0
    gutenberg_docs_total: int = 0
    gutenberg_docs_kept: int = 0
    culturax_docs_total: int = 0
    culturax_docs_kept: int = 0

    lines_total: int = 0
    lines_kept: int = 0
    lines_deduped_exact: int = 0
    lines_pii_dropped: int = 0
    docs_dropped_short: int = 0
    docs_dropped_language: int = 0
    docs_dropped_minhash: int = 0
    chars_written: int = 0

    def total_docs_kept(self) -> int:
        return (
            self.wikipedia_docs_kept
            + self.gutenberg_docs_kept
            + self.culturax_docs_kept
        )

    def to_dict(self) -> dict[str, int]:
        return {k: v for k, v in self.__dict__.items()}

def clean_line(line: str) -> str:
    """Normaliza espacos e remove caracteres de controle."""
    line = line.strip()
    line = WHITESPACE_RE.sub(" ", line)
    return line

def is_valid_line(line: str, min_chars: int) -> bool:
    """Retorna True se a linha e conteudo textual utilizavel."""
    if len(line) < min_chars:
        return False
    if ONLY_SYMBOLS_RE.match(line):
        return False
    return True

def has_pii(line: str) -> bool:
    """Retorna True se a linha contem algum padrao de PII."""
    for pattern in PII_PATTERNS:
        if pattern.search(line):
            return True
    return False

def line_sha1(line: str) -> str:
    """Hash SHA-1 para dedup exata de linhas."""
    return hashlib.sha1(line.encode("utf-8")).hexdigest()

def doc_sha1(text: str) -> str:
    """Hash SHA-1 para identificador unico de documento no MinHash."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def sha256_file(path: Path, buf_size: int = 1024 * 1024) -> str:
    """SHA-256 do arquivo final em streaming (nao carrega em RAM)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(buf_size):
            h.update(chunk)
    return h.hexdigest()

def is_portuguese(text: str, min_ratio: float) -> bool:
    """
    Heuristica rapida de idioma por proporcao de stopwords PT.
    Divide os primeiros 500 tokens (palavras) e conta quantos sao stopwords.
    Rejeita o documento se a proporcao for menor que min_ratio.
    Robustez: textos muito curtos passam (o filtro min_doc_chars ja cobre).
    """
    words = text.lower().split()[:500]
    if len(words) < 20:
        return True
    hits = sum(1 for w in words if w in PT_STOPWORDS)
    return (hits / len(words)) >= min_ratio

def make_minhash(text: str, num_perm: int):  # type: ignore[return]
    """
    Cria um MinHash a partir dos shingles de trigramas de palavras do texto.
    Importacao lazy para nao quebrar se datasketch nao estiver instalado e
    o usuario estiver rodando apenas Wikipedia/Gutenberg sem dedup fuzzy.
    """
    try:
        from datasketch import MinHash
    except ImportError:
        raise SystemExit(
            "datasketch nao encontrado. Instale com: pip install datasketch>=1.6.0\n"
            "Ou use --no-minhash para desabilitar dedup fuzzy."
        )
    m = MinHash(num_perm=num_perm)

    words = text.lower().split()
    for i in range(len(words) - 2):
        shingle = " ".join(words[i : i + 3])
        m.update(shingle.encode("utf-8"))
    return m

def iter_wikipedia(
    wiki_version: str = "20231101.pt",
    max_docs: int | None = None,
) -> Generator[str, None, None]:
    """
    Itera sobre documentos da Wikipedia PT via HF datasets.
    Retorna o campo 'text' de cada artigo, sem limpeza adicional aqui
    (a limpeza e feita em filter_and_dedupe).

    Args:
        wiki_version: config do dataset (ex: '20231101.pt')
        max_docs: limite de documentos (None = todos)

    Yields:
        Texto bruto de cada artigo Wikipedia.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("datasets nao encontrado. pip install datasets>=2.18.0")

    print(f"[wikipedia] Carregando wikimedia/wikipedia ({wiki_version})...", flush=True)
    try:
        ds = load_dataset("wikimedia/wikipedia", wiki_version, split="train")
    except Exception as exc:
        raise SystemExit(
            f"[wikipedia] Falha ao carregar dataset HF: {exc}\n"
            "Verifique conexao com internet e cache HF (~/.cache/huggingface/)."
        )

    total = len(ds) if max_docs is None else min(max_docs, len(ds))
    print(f"[wikipedia] {total:,} artigos para processar.", flush=True)

    count = 0
    for doc in ds:
        if max_docs is not None and count >= max_docs:
            break
        text = doc.get("text", "")
        if text:
            yield text
        count += 1

def iter_gutenberg(
    book_ids: list[tuple[int, str, str]] | None = None,
    rate_limit_seconds: float = 1.0,
    max_docs: int | None = None,
) -> Generator[str, None, None]:
    """
    Itera sobre obras do Project Gutenberg via HTTP.
    Faz rate limiting para nao sobrecarregar o servidor (1 req/s por padrao).
    Remove header/footer canonico do Gutenberg antes de retornar.

    Args:
        book_ids: lista de (id, autor, titulo). Usa GUTENBERG_PT_IDS se None.
        rate_limit_seconds: espera minima entre requisicoes
        max_docs: limite de documentos (None = todos da lista)

    Yields:
        Texto da obra sem header/footer Gutenberg.
    """
    if book_ids is None:
        book_ids = GUTENBERG_PT_IDS

    seen_ids: set[int] = set()
    unique_books: list[tuple[int, str, str]] = []
    for entry in book_ids:
        if entry[0] not in seen_ids:
            unique_books.append(entry)
            seen_ids.add(entry[0])

    if max_docs is not None:
        unique_books = unique_books[:max_docs]

    print(f"[gutenberg] {len(unique_books)} obras para baixar.", flush=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "MaracatuAI-corpus-builder/2.0 (research)"})

    for idx, (book_id, autor, titulo) in enumerate(unique_books):
        url = GUTENBERG_TEXT_URL.format(book_id=book_id)
        print(
            f"[gutenberg] ({idx + 1}/{len(unique_books)}) {autor} -- {titulo} (ID {book_id})",
            flush=True,
        )

        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()

            try:
                text = resp.content.decode("utf-8")
            except UnicodeDecodeError:
                text = resp.content.decode("iso-8859-1", errors="replace")
        except requests.RequestException as exc:
            print(f"[gutenberg] AVISO: falha ao baixar ID {book_id}: {exc}", flush=True)

            time.sleep(rate_limit_seconds)
            continue

        text_clean = text
        for marker in GUTENBERG_START_MARKERS:
            pos = text_clean.find(marker)
            if pos != -1:

                eol = text_clean.find("\n", pos)
                if eol != -1:
                    text_clean = text_clean[eol + 1 :]
                break

        for marker in GUTENBERG_END_MARKERS:
            pos = text_clean.find(marker)
            if pos != -1:
                text_clean = text_clean[:pos]
                break

        text_clean = text_clean.strip()
        if text_clean:
            yield text_clean

        if idx < len(unique_books) - 1:
            time.sleep(rate_limit_seconds)

def iter_culturax(
    max_docs: int | None = None,
    hf_config: str = "pt",
) -> Generator[str, None, None]:
    """
    Itera sobre documentos do CulturaX-PT via HF datasets em modo streaming.
    Streaming e obrigatorio: o dataset completo tem dezenas de GBs.

    Licenca: ODC-BY 1.0. Atribuicao registrada em MANIFEST_v2.txt e data/README.md.

    Args:
        max_docs: limite de documentos (None = dataset inteiro, nao recomendado)
        hf_config: subset do CulturaX (padrao: 'pt')

    Yields:
        Texto bruto de cada documento CulturaX.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("datasets nao encontrado. pip install datasets>=2.18.0")

    print(
        f"[culturax] Carregando uonlp/CulturaX ({hf_config}) em streaming...",
        flush=True,
    )
    try:
        ds = load_dataset(
            "uonlp/CulturaX",
            hf_config,
            split="train",
            streaming=True,
        )
    except Exception as exc:
        raise SystemExit(
            f"[culturax] Falha ao carregar dataset HF: {exc}\n"
            "Verifique conexao com internet. CulturaX requer aceite de termos no HF Hub:\n"
            "https://huggingface.co/datasets/uonlp/CulturaX"
        )

    count = 0
    for doc in ds:
        if max_docs is not None and count >= max_docs:
            break
        text = doc.get("text", "")
        if text:
            yield text
        count += 1
        if count % 50_000 == 0:
            print(f"[culturax] {count:,} documentos lidos...", flush=True)

def filter_and_dedupe(
    docs: Generator[str, None, None],
    source_name: str,
    cfg: FilterConfig,
    stats: RunStats,
    seen_line_hashes: set[str],
    lsh,
    use_minhash: bool = True,
    total_hint: int | None = None,
) -> Generator[str, None, None]:
    """
    Aplica filtros em cadeia a um generator de documentos e yield dos blocos limpos.

    Ordem dos filtros:
        1. min_doc_chars (descarte rapido antes de processar)
        2. detect_portuguese (heuristica por stopwords)
        3. MinHash LSH (dedup fuzzy entre documentos)
        4. limpeza linha a linha + min_line_chars + so-simbolos + PII
        5. SHA-1 exata por linha (dedup entre fontes)
        6. descarta documento se nenhuma linha sobrou

    Args:
        docs: generator de strings (documentos brutos)
        source_name: nome da fonte para logs (wikipedia, gutenberg, culturax)
        cfg: FilterConfig com thresholds
        stats: RunStats acumulado (modificado in-place)
        seen_line_hashes: set compartilhado entre fontes para dedup exata
        lsh: MinHashLSH ou None se use_minhash=False
        use_minhash: se False, pula dedup fuzzy (mais rapido, menos agressivo)
        total_hint: total esperado de docs (para tqdm)

    Yields:
        Blocos de texto limpos, prontos para escrita no corpus.
    """
    source_total_attr = f"{source_name}_docs_total"
    source_kept_attr = f"{source_name}_docs_kept"

    pbar = tqdm(
        docs,
        desc=f"  {source_name}",
        unit="doc",
        total=total_hint,
        file=sys.stdout,
        dynamic_ncols=True,
    )

    for raw_text in pbar:
        setattr(stats, source_total_attr, getattr(stats, source_total_attr) + 1)

        if len(raw_text) < cfg.min_doc_chars:
            stats.docs_dropped_short += 1
            continue

        if not is_portuguese(raw_text, cfg.pt_stopword_min_ratio):
            stats.docs_dropped_language += 1
            continue

        if use_minhash and lsh is not None:
            mh = make_minhash(raw_text, cfg.minhash_num_perm)
            d_key = doc_sha1(raw_text)
            neighbors = lsh.query(mh)
            if neighbors:

                stats.docs_dropped_minhash += 1
                continue
            lsh.insert(d_key, mh)

        kept_lines: list[str] = []
        for raw_line in raw_text.split("\n"):
            stats.lines_total += 1
            line = clean_line(raw_line)

            if not is_valid_line(line, cfg.min_line_chars):
                continue

            if has_pii(line):
                stats.lines_pii_dropped += 1
                continue

            h = line_sha1(line)
            if h in seen_line_hashes:
                stats.lines_deduped_exact += 1
                continue
            seen_line_hashes.add(h)

            kept_lines.append(line)
            stats.lines_kept += 1

        if not kept_lines:
            continue

        setattr(stats, source_kept_attr, getattr(stats, source_kept_attr) + 1)
        yield "\n".join(kept_lines)

    pbar.close()

def write_corpus(
    sources: list[tuple[str, Generator[str, None, None], int | None]],
    cfg: FilterConfig,
    path_cfg: PathConfig,
    use_minhash: bool = True,
) -> RunStats:
    """
    Orquestra a escrita do corpus_v2.txt processando todas as fontes em sequencia.
    A ordem de escrita e deterministica: wikipedia -> gutenberg -> culturax.

    Args:
        sources: lista de (nome, generator, total_hint_opcional)
        cfg: FilterConfig
        path_cfg: PathConfig
        use_minhash: se True, inicializa MinHashLSH e aplica dedup fuzzy

    Returns:
        RunStats com contadores finais.
    """
    path_cfg.output_file.parent.mkdir(parents=True, exist_ok=True)
    stats = RunStats()
    seen_line_hashes: set[str] = set()

    lsh = None
    if use_minhash:
        try:
            from datasketch import MinHashLSH
            lsh = MinHashLSH(threshold=cfg.minhash_threshold, num_perm=cfg.minhash_num_perm)
            print(
                f"[minhash] LSH inicializado: threshold={cfg.minhash_threshold}, "
                f"num_perm={cfg.minhash_num_perm}",
                flush=True,
            )
        except ImportError:
            print(
                "[minhash] AVISO: datasketch nao encontrado. Dedup fuzzy desabilitado.\n"
                "  Instale com: pip install datasketch>=1.6.0",
                flush=True,
            )
            use_minhash = False

    with open(path_cfg.output_file, "w", encoding="utf-8") as out:
        for source_name, doc_gen, total_hint in sources:
            print(f"\n[{source_name}] Iniciando processamento...", flush=True)
            filtered = filter_and_dedupe(
                docs=doc_gen,
                source_name=source_name,
                cfg=cfg,
                stats=stats,
                seen_line_hashes=seen_line_hashes,
                lsh=lsh,
                use_minhash=use_minhash,
                total_hint=total_hint,
            )
            for block in filtered:
                line = block + "\n"
                out.write(line)
                stats.chars_written += len(line)

    return stats

def write_manifest(
    stats: RunStats,
    cfg: FilterConfig,
    path_cfg: PathConfig,
    sources_used: list[str],
    corpus_sha256: str,
) -> None:
    """Escreve MANIFEST_v2.txt com SHA-256, stats e parametros do run."""
    with open(path_cfg.manifest_file, "w", encoding="utf-8") as mf:
        mf.write("Maracatu -- MANIFEST do corpus v2\n")
        mf.write("=" * 60 + "\n\n")
        mf.write(f"corpus_sha256: {corpus_sha256}\n\n")

        mf.write("Fontes utilizadas:\n")
        for src in sources_used:
            meta = SOURCE_META.get(src, {})
            mf.write(f"  {src}:\n")
            for k, v in meta.items():
                mf.write(f"    {k}: {v}\n")
        mf.write("\n")

        mf.write("Estatisticas:\n")
        for k, v in stats.to_dict().items():
            mf.write(f"  {k}: {v:,}\n")
        mf.write(f"  total_docs_kept: {stats.total_docs_kept():,}\n\n")

        mf.write("Parametros de filtro:\n")
        mf.write(f"  min_doc_chars: {cfg.min_doc_chars}\n")
        mf.write(f"  min_line_chars: {cfg.min_line_chars}\n")
        mf.write(f"  pt_stopword_min_ratio: {cfg.pt_stopword_min_ratio}\n")
        mf.write(f"  minhash_threshold: {cfg.minhash_threshold}\n")
        mf.write(f"  minhash_num_perm: {cfg.minhash_num_perm}\n")

def write_stats_json(
    stats: RunStats,
    corpus_sha256: str,
    path_cfg: PathConfig,
) -> None:
    """Escreve stats_v2.json para consumo programatico."""
    data = stats.to_dict()
    data["total_docs_kept"] = stats.total_docs_kept()
    data["corpus_sha256"] = corpus_sha256
    with open(path_cfg.stats_file, "w", encoding="utf-8") as sf:
        json.dump(data, sf, indent=2, ensure_ascii=False)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Constroi o corpus v2 do Maracatu (Wikipedia + Gutenberg + CulturaX)."
    )
    parser.add_argument(
        "--source",
        choices=["wikipedia", "gutenberg", "culturax", "all"],
        default="all",
        help="Fonte a processar (padrao: all)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Modo rapido: 1000 docs por fonte. "
            "Util para validar o pipeline sem rodar tudo."
        ),
    )
    parser.add_argument(
        "--max-docs-per-source",
        type=int,
        default=None,
        help="Limite de documentos por fonte (override do smoke-test)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUTPUT_FILE,
        help=f"Path do corpus de saida (padrao: {OUTPUT_FILE})",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_FILE,
        help=f"Path do manifest (padrao: {MANIFEST_FILE})",
    )
    parser.add_argument(
        "--no-minhash",
        action="store_true",
        help="Desabilita dedup fuzzy por MinHash LSH (mais rapido, menos agressivo)",
    )
    parser.add_argument(
        "--min-doc-chars",
        type=int,
        default=200,
        help="Minimo de caracteres por documento (padrao: 200)",
    )
    parser.add_argument(
        "--min-line-chars",
        type=int,
        default=30,
        help="Minimo de caracteres por linha (padrao: 30)",
    )
    parser.add_argument(
        "--minhash-threshold",
        type=float,
        default=0.85,
        help="Threshold de similaridade Jaccard para dedup MinHash (padrao: 0.85)",
    )
    parser.add_argument(
        "--minhash-num-perm",
        type=int,
        default=128,
        help="Numero de permutacoes MinHash (padrao: 128)",
    )
    parser.add_argument(
        "--wiki-version",
        default="20231101.pt",
        help="Config do dataset wikimedia/wikipedia (padrao: 20231101.pt)",
    )
    parser.add_argument(
        "--culturax-config",
        default="pt",
        help="Subset do CulturaX (padrao: pt)",
    )
    args = parser.parse_args()

    max_docs: int | None = args.max_docs_per_source
    if args.smoke_test and max_docs is None:
        max_docs = 1_000
        print("[smoke-test] Limite de 1.000 documentos por fonte ativado.", flush=True)

    cfg = FilterConfig(
        min_doc_chars=args.min_doc_chars,
        min_line_chars=args.min_line_chars,
        minhash_threshold=args.minhash_threshold,
        minhash_num_perm=args.minhash_num_perm,
    )
    path_cfg = PathConfig(
        output_file=args.out,
        manifest_file=args.manifest,
        stats_file=args.out.parent / "stats_v2.json",
    )
    use_minhash = not args.no_minhash

    print("Maracatu -- Build corpus v2", flush=True)
    print(f"  Saida:          {path_cfg.output_file}", flush=True)
    print(f"  Manifest:       {path_cfg.manifest_file}", flush=True)
    print(f"  Fonte(s):       {args.source}", flush=True)
    print(f"  MinHash LSH:    {'sim' if use_minhash else 'nao'}", flush=True)
    if max_docs:
        print(f"  Max docs/fonte: {max_docs:,}", flush=True)
    print(flush=True)

    sources_to_run: list[str] = (
        ["wikipedia", "gutenberg", "culturax"]
        if args.source == "all"
        else [args.source]
    )

    sources: list[tuple[str, Generator[str, None, None], int | None]] = []

    if "wikipedia" in sources_to_run:
        wiki_gen = iter_wikipedia(wiki_version=args.wiki_version, max_docs=max_docs)
        sources.append(("wikipedia", wiki_gen, max_docs))

    if "gutenberg" in sources_to_run:
        gut_ids = GUTENBERG_PT_IDS[:max_docs] if max_docs else GUTENBERG_PT_IDS
        gut_gen = iter_gutenberg(book_ids=gut_ids, rate_limit_seconds=1.0)
        sources.append(("gutenberg", gut_gen, len(gut_ids)))

    if "culturax" in sources_to_run:
        cx_gen = iter_culturax(max_docs=max_docs, hf_config=args.culturax_config)
        sources.append(("culturax", cx_gen, max_docs))

    stats = write_corpus(
        sources=sources,
        cfg=cfg,
        path_cfg=path_cfg,
        use_minhash=use_minhash,
    )

    print("\nCalculando SHA-256 do corpus...", flush=True)
    corpus_sha256 = sha256_file(path_cfg.output_file)

    write_manifest(
        stats=stats,
        cfg=cfg,
        path_cfg=path_cfg,
        sources_used=sources_to_run,
        corpus_sha256=corpus_sha256,
    )
    write_stats_json(
        stats=stats,
        corpus_sha256=corpus_sha256,
        path_cfg=path_cfg,
    )

    print("\n" + "=" * 60, flush=True)
    print("Corpus v2 concluido.", flush=True)
    print(f"  Arquivo:             {path_cfg.output_file}", flush=True)
    print(f"  SHA-256:             {corpus_sha256}", flush=True)
    print(f"  Chars escritos:      {stats.chars_written:,}", flush=True)
    print(f"  MB escritos:         {stats.chars_written / 1_000_000:.1f}", flush=True)
    print(f"  Docs kept (total):   {stats.total_docs_kept():,}", flush=True)
    print(f"    wikipedia:         {stats.wikipedia_docs_kept:,}", flush=True)
    print(f"    gutenberg:         {stats.gutenberg_docs_kept:,}", flush=True)
    print(f"    culturax:          {stats.culturax_docs_kept:,}", flush=True)
    print(f"  Linhas kept:         {stats.lines_kept:,}", flush=True)
    print(f"  Linhas deduped:      {stats.lines_deduped_exact:,}", flush=True)
    print(f"  Linhas PII drop:     {stats.lines_pii_dropped:,}", flush=True)
    print(f"  Docs drop (short):   {stats.docs_dropped_short:,}", flush=True)
    print(f"  Docs drop (idioma):  {stats.docs_dropped_language:,}", flush=True)
    print(f"  Docs drop (minhash): {stats.docs_dropped_minhash:,}", flush=True)
    print(f"  Manifest:            {path_cfg.manifest_file}", flush=True)
    print(f"  Stats JSON:          {path_cfg.stats_file}", flush=True)
    print("=" * 60, flush=True)
    print("\nProximo passo: python tokenizer/train_tokenizer.py", flush=True)

if __name__ == "__main__":
    main()
