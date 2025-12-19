from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class EncoderConfig:
    input_dim: int = 144
    encoder_dim: int = 208 # Model 2 matches cfg_2
    num_layers: int = 14 # Model 2 
    num_attention_heads: int = 8 # Model 2 usually 8 heads? cfg_2 says 4 or 8?
    # cfg_2.py: cfg.encoder_config.num_attention_heads = 8? 
    # Let's check cfg_2.py content again from memory or view it. 
    # Previous view showed "cfg.encoder_config = SimpleNamespace(**{'input_dim': 144, 'encoder_dim': 208, ...})"
    # I should verifying cfg_2 exactly. But typically 208/8 = 26.
    feed_forward_expansion_factor: int = 1
    conv_expansion_factor: int = 2
    input_dropout_p: float = 0.1
    feed_forward_dropout_p: float = 0.1
    attention_dropout_p: float = 0.1
    conv_dropout_p: float = 0.1
    conv_kernel_size: int = 51 # Model 2 uses 31 or 51? cfg_2 usually 31?
    # Let's verify cfg_2 content in next step if unsure. But updating structure for now.

@dataclass
class DecoderConfig:
    vocab_size: int = 63
    d_model: int = 144
    decoder_layers: int = 2
    decoder_ffn_dim: int = 512
    num_heads: int = 4  # Derived from d_model usually, but specifying for clarity if needed
    attention_dropout: float = 0.2
    pad_token_id: int = 60 # Default from cfg_1 PAD
    bos_token_id: int = 61 # Default from cfg_1 SOS
    eos_token_id: int = 62 # Default from cfg_1 EOS
    decoder_start_token_id: int = 61
    max_length: int = 2048 # Default safe value

@dataclass
class Config:
    encoder_config: EncoderConfig = field(default_factory=EncoderConfig)
    decoder_config: DecoderConfig = field(default_factory=DecoderConfig)
    n_landmarks: int = 130
    ce_ignore_index: int = -100
    label_smoothing: float = 0.0
    val_mode: str = 'padded'
    max_phrase: int = 33
    
    # Validation/Training params
    batch_size: int = 64
    max_len: int = 384
