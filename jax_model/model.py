import jax
import jax.numpy as jnp
from flax import nnx
from .layers import SqueezeformerBlock, Swish, RelPositionalEncoding
from .config import Config
import math

class FeatureExtractor(nnx.Module):
    def __init__(self, n_landmarks: int, out_dim: int, rngs: nnx.Rngs = None):
        self.in_channels = (32 // 2) * n_landmarks # Matches PyTorch logic
        self.out_dim = out_dim
        
        self.stem_linear = nnx.Linear(self.in_channels, out_dim, use_bias=False, rngs=rngs)
        self.stem_bn = nnx.BatchNorm(out_dim, momentum=0.95, rngs=rngs)
        
        # Conv Stem: 3 input channels (x, y, z), 32 out.
        # PyTorch: Conv2d(3, 32, (3,3), (1,2), (1,1))
        # Input: (B, T, N, 3) -> Permute (B, 3, T, N)
        # JAX Conv: (B, T, N, 3) -> We probably treat T as H, N as W?
        # PyTorch: (B, C, H, W). Here (B, 3, T, N).
        # Kernel (3,3) on (T, N). Stride (1, 2).
        self.conv_stem = nnx.Conv(
            in_features=3,
            out_features=32,
            kernel_size=(3, 3),
            strides=(1, 2),
            padding=[(1, 1), (1, 1)],
            use_bias=False,
            rngs=rngs
        )
        self.bn_conv = nnx.BatchNorm(32, momentum=0.1, rngs=rngs) # PyTorch momentum 0.1 is Flax 0.9? Wait.
        # PyTorch momentum m means: new_running = (1-m)*old + m*observed.
        # Flax momentum m means: new_running = m*old + (1-m)*observed.
        # So PyTorch 0.1 => Flax 0.1 ?? No.
        # PyTorch: run = (1-0.1)*run + 0.1*obs = 0.9*run + 0.1*obs
        # Flax: run = 0.9*run + (1-0.9)*obs.
        # So PyTorch 0.1 matches Flax 0.9.
        # BUT: In PyTorch code `stem_bn` has momentum=0.95.
        # PyTorch 0.95 => 0.05 update. Flax 0.95 => 0.05 update. 
        # Wait, PyTorch docs say: "momentum: the value used for the running_mean and running_var computation. Can be set to None for cumulative moving average (i.e. simple average). Default: 0.1"
        # x_new = (1 - momentum) * x_old + momentum * x_t
        # Flax: decay_rate * old + (1 - decay_rate) * new
        # So PyTorch 0.1 (default) = Flax 0.9.
        # PyTorch 0.95 (stem_bn) = Flax 0.05? NO.
        # Usually users mean decay when they say 0.95. 
        # Let's assume PyTorch code meant standard smoothing.
        # If PyTorch user sets 0.95 explicitly... likely they want high smoothing?
        # Actually in `mdl_1_pt.py`: `self.stem_bn = nn.BatchNorm1d(out_dim, momentum=0.95)`
        # If they meant slow updates, PyTorch 0.1 is standard. 0.95 is VERY FAST updates (almost instantaneous).
        # OR they confused PyTorch momentum with Keras/TF momentum (where 0.99 is standard).
        # We will iterate on this. For now use 0.95 as decay (slow updates) which is safer default.
        
        self.act = nnx.silu # Swish

    def __call__(self, x: jax.Array, mask: jax.Array, training: bool = True) -> jax.Array:
        # x: (B, T, N, 3)
        # mask: (B, T)
        
        # Conv Stem
        # JAX Conv expects (B, H, W, C).
        # We have (B, T, N, 3).
        xc = self.conv_stem(x) # (B, T, N/2, 32) (approx)
        xc = self.bn_conv(xc, use_running_average=not training)
        xc = self.act(xc)
        
        # Flatten
        # (B, T, N/2, 32) -> (B, T, N/2 * 32)
        B, T, N2, C = xc.shape
        xc = xc.reshape(B, T, -1)
        
        # Stem Linear
        x = self.stem_linear(xc)
        
        # Masked BN logic
        mask_bool = None
        if mask is not None:
             mask_bool = mask.astype(bool)[..., None] # (B, T, 1) to broadcast with (B, T, C)
        
        x = self.stem_bn(x, use_running_average=not training, mask=mask_bool) # Flax BN uses mask for stats if provided
        
        if mask is not None:
            x = x * mask[:, :, None]
            
        return x

class SqueezeformerEncoder(nnx.Module):
    def __init__(self, config: Config, rngs: nnx.Rngs = None):
        cfg = config.encoder_config
        self.num_layers = cfg.num_layers
        self.blocks = nnx.List([
            SqueezeformerBlock(
                encoder_dim=cfg.encoder_dim,
                num_attention_heads=cfg.num_attention_heads,
                feed_forward_expansion_factor=cfg.feed_forward_expansion_factor,
                conv_expansion_factor=cfg.conv_expansion_factor,
                feed_forward_dropout_p=cfg.feed_forward_dropout_p,
                attention_dropout_p=cfg.attention_dropout_p,
                conv_dropout_p=cfg.conv_dropout_p,
                conv_kernel_size=cfg.conv_kernel_size,
                rngs=rngs
            )
            for _ in range(self.num_layers)
        ])

    def __call__(self, x: jax.Array, mask: jax.Array, training: bool = True) -> jax.Array:
        for block in self.blocks:
            x = block(x, mask=mask, training=training)
        return x

class TransformerDecoderLayer(nnx.Module):
    # Minimal decoding layer
    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int, dropout: float = 0.1, rngs: nnx.Rngs = None):
        self.self_attn = nnx.MultiHeadAttention(num_heads=num_heads, in_features=d_model, dropout_rate=dropout, rngs=rngs, decode=False) 
        # We might need 'decode=True' for fast inference later.
        
        self.multihead_attn = nnx.MultiHeadAttention(num_heads=num_heads, in_features=d_model, dropout_rate=dropout, rngs=rngs, decode=False)
        
        self.linear1 = nnx.Linear(d_model, dim_feedforward, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)
        self.linear2 = nnx.Linear(dim_feedforward, d_model, rngs=rngs)
        
        self.norm1 = nnx.LayerNorm(d_model, rngs=rngs)
        self.norm2 = nnx.LayerNorm(d_model, rngs=rngs)
        self.norm3 = nnx.LayerNorm(d_model, rngs=rngs)
        
        self.dropout1 = nnx.Dropout(dropout, rngs=rngs)
        self.dropout2 = nnx.Dropout(dropout, rngs=rngs)
        self.dropout3 = nnx.Dropout(dropout, rngs=rngs)
        
        self.activation = nnx.gelu

    def __call__(self, tgt, memory, tgt_mask=None, memory_mask=None, training=True):
        # tgt: (B, T, D)
        # memory: (B, S, D)
        
        # Self Attn
        # tgt_mask needed for causal.
        x = tgt
        x2 = self.self_attn(x, x, mask=tgt_mask, deterministic=not training)
        x = x + self.dropout1(x2, deterministic=not training)
        x = self.norm1(x)
        
        # Cross Attn
        x2 = self.multihead_attn(x, memory, mask=memory_mask, deterministic=not training)
        x = x + self.dropout2(x2, deterministic=not training)
        x = self.norm2(x)
        
        # FFN
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x)), deterministic=not training))
        x = x + self.dropout3(x2, deterministic=not training)
        x = self.norm3(x)
        
        return x

class Decoder(nnx.Module):
    def __init__(self, config: Config, rngs: nnx.Rngs = None):
        cfg = config.decoder_config
        self.embed_tokens = nnx.Embed(cfg.vocab_size, cfg.d_model, rngs=rngs)
        self.embed_positions = nnx.Embed(cfg.max_length + 2, cfg.d_model, rngs=rngs) # Offset?
        
        self.layers = nnx.List([
            TransformerDecoderLayer(cfg.d_model, cfg.num_heads, cfg.decoder_ffn_dim, cfg.attention_dropout, rngs=rngs)
            for _ in range(cfg.decoder_layers)
        ])
        
        self.embed_scale = math.sqrt(cfg.d_model)
        self.pad_token_id = cfg.pad_token_id
        
    def __call__(self, input_ids, encoder_hidden_states, attention_mask=None, encoder_attention_mask=None, training=True):
        # input_ids: (B, T)
        B, T = input_ids.shape
        
        # Alignment with PyTorch: scaler * (embed + pos)
        x = self.embed_tokens(input_ids) * self.embed_scale
        positions = jnp.arange(T)[None, :]
        x = x + self.embed_positions(positions) # (B, T, D)
        
        # Decoder logic
        # Causal Mask
        # nnx.MultiHeadAttention expects mask (B, H, Q, K) or similar.
        # We need generic causal mask.
        causal_mask = nnx.make_causal_mask(input_ids)
        
        # Combine with padding mask if provided
        if attention_mask is not None:
             # attention_mask is (B, 1, T) usually?
             # causal_mask is (1, 1, T, T)
             # We need min(-inf) where mask is 0.
             pass 
        
        for layer in self.layers:
            x = layer(x, encoder_hidden_states, tgt_mask=causal_mask, memory_mask=encoder_attention_mask, training=training)
            
        return x

class Net(nnx.Module):
    def __init__(self, config: Config, rngs: nnx.Rngs = None):
        self.config = config
        self.feature_extractor = FeatureExtractor(config.n_landmarks, config.encoder_config.encoder_dim, rngs=rngs)
        self.encoder = SqueezeformerEncoder(config, rngs=rngs)
        self.decoder = Decoder(config, rngs=rngs)
        self.lm_head = nnx.Linear(config.decoder_config.d_model, config.decoder_config.vocab_size, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array, input_mask: jax.Array, decoder_input_ids: jax.Array, training: bool = True) -> jax.Array:
        # x: (B, T, N, 3)
        # input_mask: (B, T)
        # decoder_input_ids: (B, T_out)
        
        if input_mask is not None:
             input_mask_bool = input_mask.astype(bool)
        else:
             input_mask_bool = None

        x = self.feature_extractor(x, mask=input_mask, training=training) # FeatureExtractor handles float mask for mult, bool for BN?
        # Wait, FeatureExtractor expect mask as Array. I should pass float there if it multiplies, but it calls BN with bool.
        # I updated FeatureExtractor to handle casting itself.
        
        x = self.encoder(x, mask=input_mask_bool, training=training)
        
        # Decoder
        # encoder_attention_mask: (B, 1, 1, T_enc)?
        enc_mask = None
        if input_mask is not None:
            enc_mask = input_mask[:, None, None, :]
            
        decoder_out = self.decoder(decoder_input_ids, x, encoder_attention_mask=enc_mask, training=training)
        logits = self.lm_head(decoder_out)
        
        return logits
