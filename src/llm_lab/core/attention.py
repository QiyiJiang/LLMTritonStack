import math
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from ..config import TritonMindConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """将KV头重复n_rep次以匹配Q头数量"""
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )

class BaseAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def _softmax(self, x, dim=-1):
        x_max = torch.max(x, dim=dim, keepdim=True).values
        exp_x = torch.exp(x - x_max)
        return exp_x / exp_x.sum(dim=dim, keepdim=True)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / math.sqrt(self.hidden_size)

        att = self._softmax(scores, dim=-1)

        return torch.matmul(att, v)


class MaskedAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def _softmax(self, x, dim=-1):
        x_max = torch.max(x, dim=dim, keepdim=True).values
        exp_x = torch.exp(x - x_max)
        return exp_x / exp_x.sum(dim=dim, keepdim=True)

    def forward(self, x):

        _, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / math.sqrt(self.hidden_size)
        logger.debug(f"scores: {scores}")
        logger.debug(f"scores shape: {scores.shape}")

        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask == 1, float("-inf"))
        logger.debug(f"scores (after mask): {scores}")
        logger.debug(f"scores shape: {scores.shape}")

        att = self._softmax(scores, dim=-1)

        return torch.matmul(att, v)



class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()

        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def _softmax(self, x, dim=-1):
        x_max = torch.max(x, dim=dim, keepdim=True).values
        exp_x = torch.exp(x - x_max)
        return exp_x / exp_x.sum(dim=dim, keepdim=True)

    def forward(self, x):

        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)

        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask == 1, float("-inf"))

        att = self._softmax(scores, dim=-1)
        att = torch.matmul(att, v)

        """
        NOTE .contiguous() 可以将逻辑上正确但是内存不连续的 Tensor 转换为连续的，因为 view 要求 Tensor 内存连续
        """
        att = att.transpose(1, 2).contiguous()
        att = att.view(batch_size, seq_len, self.hidden_size)
        att = self.o_proj(att)

        return att

class PaddingMaskedAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()

        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def _softmax(self, x, dim=-1):
        x_max = torch.max(x, dim=dim, keepdim=True).values
        exp_x = torch.exp(x - x_max)
        return exp_x / exp_x.sum(dim=dim, keepdim=True)

    def forward(self, x, seq_lengths=None):

        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)

        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
        causal_mask = causal_mask[None, None, :, :]  # [1,1,L,L]
        if seq_lengths is not None:
            # padding mask
            padding_mask = torch.arange(seq_len, device=x.device)[None, :] >= seq_lengths[:, None]  # [B,L]
            padding_mask = padding_mask[:, None, None, :]  # [B,1,1,L]
            final_mask = causal_mask | padding_mask
        else:
            final_mask = causal_mask

        scores = scores.masked_fill(final_mask == 1, float("-inf"))

        att = self._softmax(scores, dim=-1)
        att = torch.matmul(att, v)

        att = att.transpose(1, 2).contiguous()
        att = att.view(batch_size, seq_len, self.hidden_size)
        att = self.o_proj(att)

        return att



class MultiHeadAttentionWithKVCache(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()

        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def _softmax(self, x, dim=-1):
        x_max = torch.max(x, dim=dim, keepdim=True).values
        exp_x = torch.exp(x - x_max)
        return exp_x / exp_x.sum(dim=dim, keepdim=True)

    def forward(self, x, seq_lengths=None, cached_k=None, cached_v=None):

        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if cached_k is None:
            cached_k = k
            cached_v = v
        else:
            cached_k = torch.cat([cached_k, k], dim=2)
            cached_v = torch.cat([cached_v, v], dim=2)

        scores = torch.matmul(q, cached_k.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)

        total_len = cached_k.size(2)
        causal_mask = torch.triu(torch.ones(seq_len, total_len, dtype=torch.bool, device=x.device), diagonal=1)
        causal_mask = causal_mask[None, None, :, :]  # [1,1,L,L]
        if seq_lengths is not None:
            # padding mask
            padding_mask = torch.arange(total_len, device=x.device)[None, :] >= seq_lengths[:, None]  # [B,L]
            padding_mask = padding_mask[:, None, None, :]  # [B,1,1,L]
            final_mask = causal_mask | padding_mask
        else:
            final_mask = causal_mask

        scores = scores.masked_fill(final_mask == 1, float("-inf"))

        att = self._softmax(scores, dim=-1)
        att = torch.matmul(att, cached_v)

        att = att.transpose(1, 2).contiguous()
        att = att.view(batch_size, seq_len, self.hidden_size)
        att = self.o_proj(att)

        return att, cached_k, cached_v


class IncrementalKVAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, max_seq_len: int):
        super().__init__()

        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_seq_len = max_seq_len
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def _softmax(self, x, dim=-1):
        x_max = torch.max(x, dim=dim, keepdim=True).values
        exp_x = torch.exp(x - x_max)
        return exp_x / exp_x.sum(dim=dim, keepdim=True)

    def forward(self, x, seq_lengths=None, cached=None, cached_pos=None, is_training=False):

        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if is_training is False:
            assert seq_len == 1

            if cached is None:
                cached = {
                    "k": torch.zeros(batch_size, self.num_heads, self.max_seq_len, self.head_dim, device=x.device, dtype=x.dtype),
                    "v": torch.zeros(batch_size, self.num_heads, self.max_seq_len, self.head_dim, device=x.device, dtype=x.dtype),
                }
                cached_pos = 0

            cached["k"][:, :, cached_pos, :] = k[:, :, 0, :]
            cached["v"][:, :, cached_pos, :] = v[:, :, 0, :]
            k = cached["k"][:, :, :cached_pos+1, :]
            v = cached["v"][:, :, :cached_pos+1, :]

            scores = torch.matmul(q, k.transpose(-2, -1))
            scores = scores / math.sqrt(self.head_dim)
        else:
            total_len = k.size(2)
            causal_mask = torch.triu(torch.ones(seq_len, total_len, dtype=torch.bool, device=x.device), diagonal=1)
            causal_mask = causal_mask[None, None, :, :]  # [1,1,L,L]

            if seq_lengths is not None:
                # padding mask
                padding_mask = torch.arange(total_len, device=x.device)[None, :] >= seq_lengths[:, None]  # [B,L]
                padding_mask = padding_mask[:, None, None, :]  # [B,1,1,L]
                final_mask = causal_mask | padding_mask
            else:
                final_mask = causal_mask

            scores = torch.matmul(q, k.transpose(-2, -1))
            scores = scores / math.sqrt(self.head_dim)
            scores = scores.masked_fill(final_mask == 1, float("-inf"))

        att = self._softmax(scores, dim=-1)
        att = torch.matmul(att, v)

        att = att.transpose(1, 2).contiguous()
        att = att.view(batch_size, seq_len, self.hidden_size)
        att = self.o_proj(att)

        return att, cached, cached_pos+1


class FusedQKVAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, max_seq_len: int):
        super().__init__()

        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_seq_len = max_seq_len
        
        self.qkv_proj = nn.Linear(self.hidden_size, 3* self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(self, x, seq_lengths=None, cached=None, cached_pos=None, is_training=False):

        batch_size, seq_len, _ = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if is_training is False:
            assert seq_len == 1

            if cached is None:
                cached = {
                    "k": torch.zeros(batch_size, self.num_heads, self.max_seq_len, self.head_dim, device=x.device, dtype=x.dtype),
                    "v": torch.zeros(batch_size, self.num_heads, self.max_seq_len, self.head_dim, device=x.device, dtype=x.dtype),
                }
                cached_pos = 0

            cached["k"][:, :, cached_pos, :] = k[:, :, 0, :]
            cached["v"][:, :, cached_pos, :] = v[:, :, 0, :]
            k = cached["k"][:, :, :cached_pos+1, :]
            v = cached["v"][:, :, :cached_pos+1, :]

            scores = torch.matmul(q, k.transpose(-2, -1))
            scores = scores / math.sqrt(self.head_dim)
        else:
            total_len = k.size(2)
            causal_mask = torch.triu(torch.ones(seq_len, total_len, dtype=torch.bool, device=x.device), diagonal=1)
            causal_mask = causal_mask[None, None, :, :]  # [1,1,L,L]

            if seq_lengths is not None:
                # padding mask
                padding_mask = torch.arange(total_len, device=x.device)[None, :] >= seq_lengths[:, None]  # [B,L]
                padding_mask = padding_mask[:, None, None, :]  # [B,1,1,L]
                final_mask = causal_mask | padding_mask
            else:
                final_mask = causal_mask

            scores = torch.matmul(q, k.transpose(-2, -1))
            scores = scores / math.sqrt(self.head_dim)
            scores = scores.masked_fill(final_mask == 1, float("-inf"))

        att = torch.softmax(scores, dim=-1)
        att = torch.matmul(att, v)

        att = att.transpose(1, 2).contiguous()
        att = att.view(batch_size, seq_len, self.hidden_size)
        att = self.o_proj(att)

        return att, cached, cached_pos+1


class FlashAttentionFusedAttention(nn.Module):
    """Fused QKV Attention，支持训练模式和 KV cache 增量解码。"""
    
    def __init__(self, config: TritonMindConfig):
        super().__init__()

        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_key_value_heads = config.num_key_value_heads

        self.max_seq_len = config.max_seq_len
        self.dropout = config.dropout
        
        assert config.hidden_size % config.num_heads == 0
        assert config.num_heads % config.num_key_value_heads == 0

        self.head_dim = self.hidden_size // self.num_heads
        self.n_rep = self.num_heads // self.num_key_value_heads

        # fused QKV
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.kv_proj = nn.Linear(self.hidden_size, 2 * self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.resid_dropout = nn.Dropout(self.dropout)

    def forward(
        self,
        x: torch.Tensor,
        seq_lengths: Optional[torch.Tensor] = None,
        cached: Optional[Dict[str, torch.Tensor]] = None,
        cached_pos: Optional[int] = None,
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]], Optional[int]]:
        """
        前向传播，支持训练和推理两种模式。
        
        Args:
            x: Input tensor，shape (batch_size, seq_len, hidden_size)
            seq_lengths: 每个样本的有效长度，shape (batch_size,)，用于 padding mask
            cached: KV cache dict，包含 "k" 和 "v"，用于增量解码
            cached_pos: Cache 位置，表示当前缓存到第几个位置
        
        Returns:
            (att_output, cached, cached_pos)
            - att_output: Attention 输出，shape (batch_size, seq_len, hidden_size)
            - cached: 更新后的 KV cache（推理时），训练时为 None
            - cached_pos: 更新后的 cache 位置（推理时），训练时为 None
        """
        batch_size, seq_len, _ = x.shape

        # fused QKV
        q = self.q_proj(x)
        kv = self.kv_proj(x)

        k, v = kv.chunk(2, dim=-1)

        # reshape为 [B,H,L,D] 并 contiguous
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = k.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2).contiguous()
        v = v.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2).contiguous()
        
        if freqs_cos is not None and freqs_sin is not None:
            from ..utils.rope import apply_rotary_pos_emb
            q, k = apply_rotary_pos_emb(q, k, freqs_cos, freqs_sin)

        if self.training:
            # 训练模式：构造 causal + padding mask
            k = repeat_kv(k.transpose(1, 2), self.n_rep).transpose(1, 2)  # [B, num_heads, L, D]
            v = repeat_kv(v.transpose(1, 2), self.n_rep).transpose(1, 2)  # [B, num_heads, L, D]
            total_len = k.size(2)

            if seq_lengths is not None:
                # SDPA 的 bool mask 语义：True=可见，False=不可见
                key_pos = torch.arange(total_len, device=x.device)[None, :]          # [1, L]
                valid_k = key_pos < seq_lengths[:, None]                             # [B, L]
                attn_mask = valid_k[:, None, None, :]                                # [B, 1, 1, L]
            else:
                attn_mask = None
        else:
            if cached is None:
                cached = {
                    "k": torch.zeros(batch_size, self.num_key_value_heads, self.max_seq_len, self.head_dim,
                                     device=x.device, dtype=x.dtype),
                    "v": torch.zeros(batch_size, self.num_key_value_heads, self.max_seq_len, self.head_dim,
                                     device=x.device, dtype=x.dtype),
                }
                if seq_len > 1:
                    cached["k"][:, :, :seq_len, :] = k
                    cached["v"][:, :, :seq_len, :] = v
                    cached_pos = seq_len
                else:
                    # 第一次 forward，seq_len=1，从位置 0 开始
                    cached_pos = 0
                    cached["k"][:, :, cached_pos, :] = k[:, :, 0, :]
                    cached["v"][:, :, cached_pos, :] = v[:, :, 0, :]
                    cached_pos = cached_pos + 1
                    k = cached["k"][:, :, :cached_pos, :]
                    v = cached["v"][:, :, :cached_pos, :]
            else:
                assert seq_len == 1
                cached["k"][:, :, cached_pos, :] = k[:, :, 0, :]
                cached["v"][:, :, cached_pos, :] = v[:, :, 0, :]
                cached_pos = cached_pos + 1
                k = cached["k"][:, :, :cached_pos + 1, :]
                v = cached["v"][:, :, :cached_pos + 1, :]

            k = repeat_kv(k.transpose(1, 2), self.n_rep).transpose(1, 2)  # [B, num_heads, L, D]
            v = repeat_kv(v.transpose(1, 2), self.n_rep).transpose(1, 2)  # [B, num_heads, L, D]
            attn_mask = None

        # scaled dot product attention
        att = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=self.training, dropout_p=self.dropout if self.training else 0.0)

        # 输出 reshape [B,H,L,D] -> [B,L,H*D]
        att = att.transpose(1, 2).contiguous().reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        att = self.o_proj(att)
        att = self.resid_dropout(att)

        return att, cached, cached_pos if not self.training else None


if __name__ == "__main__":
    pass
