"""llm_lab: TritonMind LLM playground package."""

from .config import TritonMindConfig
from .core.model import TritonMindModel, TritonMindForCausalLM
from .datasets.datasets import (
    PretrainDataset,
    SimplePretrainDataset,
    SimpleSFTDataset,
    SFTDataset,
)
from .utils.rope import precompute_freqs_cis, apply_rotary_pos_emb
from .utils.logger import (
    get_logger,
    setup_logger,
    debug,
    info,
    warning,
    error,
    critical,
)

__all__ = [
    "TritonMindConfig",
    "TritonMindModel",
    "TritonMindForCausalLM",
    "PretrainDataset",
    "SimplePretrainDataset",
    "SimpleSFTDataset",
    "SFTDataset",
    "precompute_freqs_cis",
    "apply_rotary_pos_emb",
    "get_logger",
    "setup_logger",
    "debug",
    "info",
    "warning",
    "error",
    "critical",
]
