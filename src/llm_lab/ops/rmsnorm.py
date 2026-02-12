import torch
from torch import nn
from ..config import TritonMindConfig


class RMSNorm(nn.Module):
    """RMS Normalization：对最后一维做 RMS 归一化。"""
    
    def __init__(self, config: TritonMindConfig) -> None:
        super().__init__()
        self.eps = config.eps
        self.weight = nn.Parameter(torch.ones(config.hidden_size))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """RMS 归一化：x / sqrt(mean(x²) + eps)。"""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        
        Args:
            x: Input tensor，shape (..., hidden_size)
        
        Returns:
            Normalized tensor，shape (..., hidden_size)
        """
        return self.weight * self._norm(x)
