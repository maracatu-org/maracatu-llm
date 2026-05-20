"""
Implementação do transformer decoder-only do Maracatu — arquitetura Llama-style.

Peças modernas em relação a GPT-2:
    - RMSNorm (sem mean-subtraction, sem bias) no lugar de LayerNorm
    - RoPE (Rotary Positional Embeddings) aplicado em Q/K no lugar de
      position embeddings aprendíveis
    - SwiGLU (gate ⊙ up → down) no lugar da FFN GELU simples
    - Suporte opcional a GQA (num_key_value_heads < num_attention_heads)

O state_dict usa as mesmas chaves do LlamaForCausalLM do HuggingFace
(model.embed_tokens, model.layers[i].self_attn.q_proj, mlp.gate_proj, etc.)
e a convenção "rotate-half" de RoPE do HF. Isso permite carregar os pesos
diretamente em transformers.LlamaForCausalLM, sem script de conversão.

Referências:
    - Touvron et al., "Llama: Open and Efficient Foundation Language Models"
    - Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    - Shazeer, "GLU Variants Improve Transformer"
    - Karpathy, nanoGPT (https://github.com/karpathy/nanoGPT) — ponto de partida
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class ModelConfig:
    """Configuração da arquitetura do Maracatu (nomes alinhados ao LlamaConfig do HF)."""

    vocab_size: int = 16000
    hidden_size: int = 384
    intermediate_size: int = 1024
    num_hidden_layers: int = 6
    num_attention_heads: int = 6
    num_key_value_heads: int = 6
    max_position_embeddings: int = 512
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    attention_dropout: float = 0.0
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) deve ser divisível por "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) deve ser "
                f"divisível por num_key_value_heads ({self.num_key_value_heads})"
            )

class RMSNorm(nn.Module):
    """Root Mean Square LayerNorm (Llama-style).

    Normaliza por rms em vez de (x - mean) / std — é 10-15% mais rápido e
    empiricamente equivalente em qualidade. Sem bias.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype)

def precompute_rope_cache(
    head_dim: int, max_seq_len: int, theta: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precomputa cos/sin das frequências de RoPE (convenção HF 'rotate-half')."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotaciona metade das dimensões — convenção do HF Llama."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Aplica RoPE a x de shape [..., T, head_dim]."""
    seq_len = x.size(-2)
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)
    return x * cos + rotate_half(x) * sin

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention com RoPE e suporte a GQA."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.kv_repeats = self.num_heads // self.num_kv_heads
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.size()

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if self.kv_repeats > 1:
            k = k.repeat_interleave(self.kv_repeats, dim=1)
            v = v.repeat_interleave(self.kv_repeats, dim=1)

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=True,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(attn_out)

class MLP(nn.Module):
    """SwiGLU MLP (Llama-style): down_proj(silu(gate_proj(x)) * up_proj(x))."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class DecoderLayer(nn.Module):
    """Um bloco transformer Llama-style: pre-norm RMS + attn, pre-norm RMS + MLP, com residuais."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = CausalSelfAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

class Maracatu(nn.Module):
    """O modelo Maracatu — decoder-only transformer Llama-style.

    Estrutura do state_dict espelha LlamaForCausalLM do HuggingFace:
        model.embed_tokens, model.layers[i].{input_layernorm, self_attn,
        post_attention_layernorm, mlp}, model.norm, lm_head
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.model = nn.ModuleDict(
            dict(
                embed_tokens=nn.Embedding(config.vocab_size, config.hidden_size),
                layers=nn.ModuleList(
                    [DecoderLayer(config) for _ in range(config.num_hidden_layers)]
                ),
                norm=RMSNorm(config.hidden_size, eps=config.rms_norm_eps),
            )
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model["embed_tokens"].weight

        head_dim = config.hidden_size // config.num_attention_heads
        cos, sin = precompute_rope_cache(head_dim, config.max_position_embeddings, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)

        for name, param in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(
                    param, mean=0.0, std=0.02 / math.sqrt(2 * config.num_hidden_layers)
                )

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = True) -> int:
        """Conta parâmetros. Por padrão desconta embeddings (convenção Llama)."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.model["embed_tokens"].weight.numel()
        return n

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        bsz, seq_len = input_ids.size()
        if seq_len > self.config.max_position_embeddings:
            raise ValueError(
                f"Sequência de tamanho {seq_len} excede "
                f"max_position_embeddings={self.config.max_position_embeddings}"
            )

        x = self.model["embed_tokens"](input_ids)
        cos = self.rope_cos[:seq_len].to(x.device)
        sin = self.rope_sin[:seq_len].to(x.device)

        for layer in self.model["layers"]:
            x = layer(x, cos, sin)
        x = self.model["norm"](x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
            return logits, loss

        logits = self.lm_head(x[:, -1:, :])
        return logits, None

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Geração autoregressiva com temperatura e top-k sampling."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -self.config.max_position_embeddings :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids
