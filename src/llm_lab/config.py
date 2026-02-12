from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class TritonMindConfig:
    """TritonMind 模型配置类，只包含模型结构参数。训练相关参数由脚本通过命令行传入。"""

    # 模型结构（默认 ~100M 参数：512 hidden, 8 layers, 8 heads）
    vocab_size: int = 6400
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 8
    num_key_value_heads: int = 2
    max_seq_len: int = 2048
    dropout: float = 0.1
    eps: float = 1e-5
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"

    # RoPE
    rope_base: float = 1e6

    # 数据管道默认（datasets 未显式传 max_length 时使用，应 <= max_seq_len）
    train_max_length: int = 340
    
    def __post_init__(self):
        """确保 device 是 torch.device 对象。"""
        if isinstance(self.device, str):
            self.device = torch.device(self.device)
