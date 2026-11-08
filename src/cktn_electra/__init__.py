"""CKTN-ELECTRA: script-aware calibrated ELECTRA adaptation for Cham, Khmer and Tay-Nung."""

from cktn_electra.config import PretrainConfig, SamplerConfig, load_pretrain_config
from cktn_electra.modeling import CKTNElectra

__all__ = ["CKTNElectra", "PretrainConfig", "SamplerConfig", "load_pretrain_config"]
__version__ = "1.0.0"
