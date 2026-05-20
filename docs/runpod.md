# Treinar o Maracatu no RunPod (A100 Spot)

Guia passo-a-passo para treinar o Maracatu-80M em uma instancia RunPod A100 80GB com spot pricing. Cobre criacao do pod, bootstrap, monitoramento, recuperacao de preempcao e download de artefatos.

Custo estimado: **$10-13 por run completo** de 100k iters (~11h em A100 80GB a $0.90-1.20/h).

## Pre-requisitos (faca uma vez, antes da primeira run)

### 1. Conta RunPod

- Cadastro em [runpod.io](https://runpod.io) com cartao pessoal
- Credito inicial carregado (sugestao: $20 pra cobrir 1 run + margem)
- Chave SSH cadastrada em **Settings > SSH Public Keys** (use a chave do seu Mac: `cat ~/.ssh/id_ed25519.pub`)

### 2. Assets de treino acessiveis

Os tres arquivos abaixo **nao estao no git** (gitignored) e precisam estar disponiveis para o pod baixar:

| Arquivo | Tamanho | Onde fica localmente |
|---|---|---|
| `data/processed/corpus_v2.txt` | 6.2 GB | `data/processed/corpus_v2.txt` |
| `data/processed/tokens.npy` | 3.0 GB | `data/processed/tokens.npy` |
| `tokenizer/maracatu.model` | 519 KB | `tokenizer/maracatu.model` |

**Metodo canonico recomendado: HF Dataset Privado**

Suba os tres arquivos para um dataset privado na org `maracatu-ai` no HF Hub:

```bash
# No seu Mac, dentro do repo
huggingface-cli login   # use seu HF_TOKEN

# Cria o dataset privado (uma vez)
huggingface-cli repo create maracatu-corpus-v2 --type dataset --organization maracatu-ai --private

# Upload dos assets (mantendo estrutura de diretorios)
huggingface-cli upload maracatu-ai/maracatu-corpus-v2 \
    data/processed/corpus_v2.txt data/processed/corpus_v2.txt \
    --repo-type dataset

huggingface-cli upload maracatu-ai/maracatu-corpus-v2 \
    data/processed/tokens.npy data/processed/tokens.npy \
    --repo-type dataset

huggingface-cli upload maracatu-ai/maracatu-corpus-v2 \
    tokenizer/maracatu.model tokenizer/maracatu.model \
    --repo-type dataset
```

Por que HF dataset privado em vez das outras opcoes:

- **HF > S3/R2 presigned**: URLs presignadas expiram (tipicamente 1-7 dias). Se a URL expirar durante uma preempcao e o pod tentar re-baixar pos-restart, o download falha silenciosamente. HF Hub com token nao expira.
- **HF > SCP direto do Mac**: SCP requer que o pod alcance o IP do seu Mac, o que nao funciona se o Mac estiver atras de CGNAT (maioria dos ISPs residenciais BR). Tambem requer o Mac ligado e sem suspensao, o que e impraticavel em runs longas.
- HF dataset privado e acessivel por token, de qualquer pod, 24/7, sem URL que expire.

### 3. HF_TOKEN disponivel

O token precisa ter permissao de leitura no dataset privado `maracatu-ai/maracatu-corpus-v2` e permissao de escrita no repo `maracatu-ai/maracatu-80m-checkpoints`.

Criar em [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) com scope `write` (write inclui read).

## Criacao do pod

1. Acesse [runpod.io/console/pods](https://runpod.io/console/pods)
2. Clique em **Deploy**
3. Configuracoes obrigatorias:

| Campo | Valor |
|---|---|
| GPU | **NVIDIA A100 80GB** |
| Cloud type | **Community Cloud** (mais barato) ou Secure Cloud (mais estavel, mais caro) |
| Pricing | **Spot** - clique em "Spot" perto do preco |
| Container image | `runpod/pytorch:2.4.0-py3.11-cuda12.1.1-devel-ubuntu22.04` |
| Container disk | 20 GB |
| Volume disk | **80 GB** (montado em `/workspace`, persiste entre preempcoes no Secure Cloud) |
| Volume mount path | `/workspace` |

4. Em **Environment Variables**, adicione:

| Chave | Valor |
|---|---|
| `HF_TOKEN` | seu token HF com permissao write |

5. Expanda **SSH** e confirme que sua chave aparece listada
6. Clique em **Deploy**
7. Aguarde status mudar para **Running** (normalmente 1-3 minutos)

**Nota importante sobre preempcao e persistencia de dados:**

- **Community Cloud**: o volume `/workspace` pode ou nao persistir entre preempcoes dependendo do provider fisico. Considere Community Cloud aceitavel para runs onde o corpus ja esta no HF (re-download e rapido comparado ao treino).
- **Secure Cloud**: `/workspace` persiste garantido entre preempcoes do mesmo template. Mais caro (~20-30%), mas elimina re-download do corpus.
- O pod pode ser preemptado a qualquer momento em spot pricing. O train.py salva `latest.pt` a cada 1000 iters E a cada 30 minutos por wallclock, entao a perda maxima e 30 minutos de treino.

## Primeira execucao

Apos o pod estar Running, copie o comando SSH do console RunPod (botao "Connect") e execute no seu Mac:

```bash
ssh root@<ip-do-pod> -p <porta>
```

Dentro do pod, rode o bootstrap (substitua o HF_TOKEN se nao estiver na env):

```bash
export HF_TOKEN=hf_xxxxxxxxxxxx   # se nao configurou no pod como env var
export CORPUS_METHOD=hf
export HF_DATASET_REPO=maracatu-ai/maracatu-corpus-v2

curl -fsSL https://raw.githubusercontent.com/maracatu-ai/maracatu/main/scripts/runpod_bootstrap.sh \
    | bash
```

Ou, se preferir clonar antes de rodar:

```bash
git clone https://github.com/maracatu-labs/maracatu.git /workspace/maracatu
bash /workspace/maracatu/scripts/runpod_bootstrap.sh
```

O bootstrap vai:

1. Verificar GPU e espaco em disco
2. Instalar dependencias de sistema (git, tmux, wget...)
3. Clonar o repo em `/workspace/maracatu`
4. Instalar uv e criar venv Python 3.12
5. Instalar PyTorch cu121 + deps do projeto
6. Fazer login no HF Hub
7. Baixar os 3 assets de treino (~9.2 GB total)
8. Iniciar o treino em sessao tmux detached chamada `train`

**Tempo estimado do bootstrap ao primeiro step de treino: 12-20 minutos** (dominado pelo download dos assets: corpus 6.2 GB + tokens 3.0 GB dependem da banda do pod, tipicamente 500 MB/s-1 GB/s nos pods RunPod, entao ~10-20s por GB = 1-3 min; instalacao de deps ~5-8 min).

Apos o bootstrap imprimir as instrucoes de reconexao, voce pode fechar o SSH.

## Monitoramento

Para acompanhar o treino:

```bash
# Reconectar ao pod
ssh root@<ip-do-pod> -p <porta>

# Acompanhar log em tempo real
tmux attach -t train
# (Ctrl+B D para sair sem matar o processo)

# Ou via tail, sem precisar do tmux
tail -f /workspace/maracatu/checkpoints/training.log

# Ver uso de GPU
watch -n 5 nvidia-smi
```

Sinais de treino saudavel no log:

- Linha `Device: cuda` e nome da GPU presentes no inicio
- `loss` caindo de ~9-10 inicial para ~3-4 em alguns milhares de steps
- `tok/s` entre 60.000-120.000 em A100 (depende de batch_size e compile)
- `[resume-safe] wallclock checkpoint salvo` aparecendo a cada ~30 minutos

## Recuperacao apos preempcao

O pod spot pode ser preemptado a qualquer momento. Quando isso acontece:

1. RunPod encerra o pod e, se configurado, pode tentar realocar
2. Voce recebe notificacao por email (configure em Settings > Notifications)

Para retomar:

1. No console RunPod, verifique se o pod foi realocado automaticamente ou crie um novo pod com as mesmas configuracoes
2. SSH no novo pod
3. Rode exatamente o mesmo comando de bootstrap:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxx
export CORPUS_METHOD=hf
export HF_DATASET_REPO=maracatu-ai/maracatu-corpus-v2
bash /workspace/maracatu/scripts/runpod_bootstrap.sh
```

O bootstrap e idempotente:

- Se `/workspace/maracatu` ja existe, faz `git pull` em vez de clonar
- Se o venv ja existe, nao recria
- Se os assets ja existem em disco, nao re-baixa
- Se `checkpoints/latest.pt` existe, o `runpod_train.sh` detecta e usa auto-resume

**Cenario Community Cloud pos-preempcao** (volume nao persistiu):

O bootstrap vai re-clonar o repo e re-baixar os assets do HF. O `latest.pt` nao existira, entao o treino reinicia do zero. Para mitigar:

- Use Secure Cloud para runs longas
- Ou faca upload manual do `latest.pt` pro HF Hub apos cada sessao longa (veja secao de download abaixo)

## Download de artefatos

### Ao final do treino

O `runpod_train.sh` faz upload automatico de `best.pt` e `final.pt` pro HF Hub ao terminar. Verifique em:

```
https://huggingface.co/maracatu-ai/maracatu-80m-checkpoints
```

### Download manual do pod para o Mac

```bash
# No seu Mac
scp -P <porta> root@<ip-do-pod>:/workspace/maracatu/checkpoints/best.pt \
    ./checkpoints/runpod/maracatu-80m-best.pt

scp -P <porta> root@<ip-do-pod>:/workspace/maracatu/checkpoints/final.pt \
    ./checkpoints/runpod/maracatu-80m-final.pt
```

### Validar checkpoint baixado

```bash
python -c "
import torch
ckpt = torch.load('checkpoints/runpod/maracatu-80m-best.pt', weights_only=False)
print('step:', ckpt['step'])
print('loss:', ckpt['loss'])
print('model_config:', ckpt['model_config'])
print('git_revision:', ckpt['git_revision'])
"
```

### Backup preventivo do latest.pt durante treino

Para proteger contra perda em Community Cloud, voce pode fazer upload manual a qualquer momento (reconectando ao pod):

```bash
ssh root@<ip-do-pod> -p <porta>
source /workspace/maracatu/.venv/bin/activate
huggingface-cli upload maracatu-ai/maracatu-80m-checkpoints \
    /workspace/maracatu/checkpoints/latest.pt checkpoints/latest.pt \
    --repo-type model
```

## Custo e alarmes

**Custo esperado:**

- A100 80GB spot: $0.90-1.20/h (varia com oferta/demanda RunPod)
- Run completa 100k iters: ~10-13h estimado, portanto $9-16
- Bootstrap (download + install, ~20 min): $0.30-0.40 adicionais
- **Total realista: $10-17 por run**

**Alarmes:**

| Sinal | Causa provavel | Acao |
|---|---|---|
| `loss: nan` nas primeiras centenas de steps | lr muito alto ou dados corrompidos | Checar tokens.npy; o config atual com lr=2.5e-4 foi calibrado, nao altere |
| `CUDA out of memory` | batch_size muito grande | A100 80GB aguenta batch=64-128 facilmente com o config atual (batch=16); improvavel, mas cheque se outra coisa esta usando VRAM |
| `GPU: Tesla P100` em vez de A100 no log | Pod alocado com GPU errada | Encerre o pod e crie novamente escolhendo A100 explicitamente |
| Treino para mas `latest.pt` nao atualizou por >30min | Processo morreu antes do wallclock save | Retome do ultimo latest.pt disponivel; perda maxima teorica e os primeiros iters antes do primeiro save em 1000 iters |
| Disco cheio | tokens.npy + corpus + checkpoints > volume | Crie pod com 80GB+ de volume; checkpoints crescem ~1.2 GB por save |

## Troubleshooting

**"CUDA nao disponivel" no log do bootstrap:**

O pod pode ter subido com CUDA nao inicializado ainda. Aguarde 30s e rode o bootstrap novamente (e idempotente). Se persistir, o pod pode ter alocado uma maquina com problema; encerre e crie outro.

**"No space left on device" durante download:**

Volume de 80GB deve ser suficiente (corpus 6.2 + tokens 3.0 + repo ~0.5 + checkpoints ~5-10 = ~20 GB). Se estourar, algo esta errado com o volume configurado. Verifique o tamanho do volume no console RunPod.

**Download do HF trava ou e muito lento:**

Use a variante com `wget` diretamente se tiver URL presignada. Alternativamente, `CORPUS_METHOD=presigned` com URLs geradas no console do HF (Settings > Access Tokens nao gera URLs; use a API Python com `hf_hub_url()` + `requests`).

**`tmux: command not found`:**

O script instala tmux via apt no inicio. Se o apt falhou por algum motivo, rode manualmente: `apt-get install -y tmux` e re-execute o bootstrap.

**train.py importa mas falha com `ModuleNotFoundError: maracatu`:**

O projeto e instalado como pacote editavel (`-e`). Se o clone do repo falhou ou esta incompleto, o `src/maracatu/` nao existe. Verifique: `ls /workspace/maracatu/src/maracatu/`. Se vazio, apague o repo e deixe o bootstrap re-clonar.
