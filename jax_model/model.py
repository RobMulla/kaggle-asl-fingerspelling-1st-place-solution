
import jax
import jax.numpy as jnp
from flax import nnx
from typing import Optional, List
from . import layers, config
import math # Added for math.ceil

# --- Feature Extractor ---

class FeatureExtractor(nnx.Module):
    def __init__(self, n_landmarks: int, out_dim: int, rngs: nnx.Rngs = None):
        # self.in_channels = (32 // 2) * n_landmarks 
        # Match PyTorch exactly: 32 * ceil(n / 2)
        self.in_channels = 32 * math.ceil(n_landmarks / 2)
        self.out_dim = out_dim
        
        self.stem_linear = nnx.Linear(self.in_channels, out_dim, use_bias=False, rngs=rngs)
        # PyTorch momentum 0.95 -> Flax decay 0.05
        self.stem_bn = nnx.BatchNorm(out_dim, momentum=0.05, rngs=rngs)
        
        # Conv Stem: 3 input channels (x, y, z), 32 out.
        # PyTorch: Conv2d(3, 32, (3,3), (1,2), (1,1))
        # Input: (B, T, N, 3) -> Permute (B, 3, T, N)
        # JAX Conv: (B, T, N, 3) -> We probably treat T as H, N as W?
        # PyTorch: (B, C, H, W). Here (B, 3, T, N).
        # Kernel (3,3) on (T, N). Stride (1, 2).
        self.conv_stem = nnx.Conv(
            in_features=3,
            out_features=32,
            kernel_size=(3, 3), # (time, landmarks)
            strides=(1, 2),
            padding=[(1, 1), (1, 1)],
            use_bias=False,
            rngs=rngs
        )
        self.bn_conv = nnx.BatchNorm(32, momentum=0.9, rngs=rngs) # PyTorch momentum 0.1 -> Flax decay 0.9
        # PyTorch momentum m means: new_running = (1-m)*old + m*observed.
        # Flax momentum m means: new_running = m*old + (1-m)*observed.
        # So PyTorch 0.1 matches Flax 0.9.
        # BUT: In PyTorch code `stem_bn` has momentum=0.95.
        
        self.act = nnx.silu # Swish

    def __call__(self, x: jax.Array, mask: jax.Array, training: bool = True) -> jax.Array:
        # x: (B, T, N, 3) (or N, 3 for simpler cases but we invoke with T)
        
        xc = self.conv_stem(x) # (B, T, ceil(N/2), 32)
        xc = self.bn_conv(xc, use_running_average=not training)
        xc = self.act(xc)
        
        # Flatten
        # (B, T, ceil(N/2), 32) -> (B, T, ceil(N/2)*32)
        B, T = xc.shape[:2]
        xc = xc.reshape(B, T, -1)
        
        # Stem Linear
        x = self.stem_linear(xc)
        
        # Masked BN logic
        mask_bool = None
        if mask is not None:
             mask_bool = mask.astype(bool)[..., None] # (B, T, 1)

        x = self.stem_bn(x, use_running_average=not training, mask=mask_bool)
        
        if mask is not None:
            x = x * mask[:, :, None]
            
        return x

# --- Encoder ---

class SqueezeformerEncoder(nnx.Module):
    def __init__(self, config: config.EncoderConfig, rngs: nnx.Rngs = None):
        self.config = config
        # Embedding removed to match PyTorch (Post-Norm, no projection here)
        
        # RoPE
        # PyTorch: rotary_emb = LlamaRotaryEmbedding(encoder_dim // num_heads, ...)
        # We need to implement LlamaRotaryEmbedding in layers.py first and use it here or passing cos/sin?
        # In mdl_2_pt, rotary_emb is in Net, and cos/sin are passed to Encoder.
        # So we don't define it here necessarily, but we can usage consistent with Net.
        
        # Blocks
        self.blocks = nnx.List([
            layers.SqueezeformerBlock(
                encoder_dim=config.encoder_dim,
                num_attention_heads=config.num_attention_heads,
                feed_forward_expansion_factor=config.feed_forward_expansion_factor,
                conv_expansion_factor=config.conv_expansion_factor,
                feed_forward_dropout_p=config.feed_forward_dropout_p,
                attention_dropout_p=config.attention_dropout_p,
                conv_dropout_p=config.conv_dropout_p,
                conv_kernel_size=config.conv_kernel_size,
                rngs=rngs
            ) for _ in range(config.num_layers)
        ])
        
        # self.norm removed to match PyTorch
 
        # mdl_2_pt SqueezeformerEncoder has: self.norm = nn.LayerNorm(encoder_dim)
        
    def __call__(self, x: jax.Array, cos: jax.Array, sin: jax.Array, mask: Optional[jax.Array] = None, training: bool = True) -> jax.Array:
        # x: (B, T, D)
        # No embedding projection

        
        for block in self.blocks:
            x = block(x, cos, sin, mask=mask, training=training)
            
        # No final norm

        return x

# --- Decoder ---





# --- Net ---

class Net(nnx.Module):
    def __init__(self, config: config.Config, rngs: nnx.Rngs = None):
        self.config = config
        
        # Extractors
        # PyTorch passes n_landmarks (e.g. 21) not coordinates (63).
        self.feature_extractor = FeatureExtractor(config.n_landmarks, config.encoder_config.encoder_dim, rngs=rngs)
        self.feature_extractor_lhand = FeatureExtractor(21, config.encoder_config.encoder_dim // 4, rngs=rngs)
        self.feature_extractor_rhand = FeatureExtractor(21, config.encoder_config.encoder_dim // 4, rngs=rngs)
        self.feature_extractor_face = FeatureExtractor(76, config.encoder_config.encoder_dim // 4, rngs=rngs)
        self.feature_extractor_pose = FeatureExtractor(12, config.encoder_config.encoder_dim // 4, rngs=rngs)
        
        # Encoder
        self.encoder = SqueezeformerEncoder(config.encoder_config, rngs=rngs)
        
        # Decoders
        self.decoder = layers.Decoder(config.decoder_config, rngs=rngs)
        self.decoder2 = layers.Decoder(config.decoder_config, rngs=rngs) # Backward
        
        # Heads
        self.fc = nnx.Linear(config.decoder_config.d_model, config.decoder_config.vocab_size, use_bias=False, rngs=rngs)
        self.fc_bwd = nnx.Linear(config.decoder_config.d_model, config.decoder_config.vocab_size, use_bias=False, rngs=rngs)
        
        self.aux_fc = nnx.Linear(config.encoder_config.encoder_dim, 1, rngs=rngs)
        
        # Helper for landmarks
        # 0-21: lhand, 21-42: rhand, 42-75: pose, 75-130: face
        # We need to indices or logic to split inputs.
        # But in JAX, we expect inputs to be (B, T, N, 3).
        
        # RoPE (Shared)
        head_dim = config.encoder_config.encoder_dim // config.encoder_config.num_attention_heads
        self.rotary_emb = layers.LlamaRotaryEmbedding(head_dim, max_position_embeddings=config.max_len, rngs=rngs)

    def __call__(self, x: jax.Array, mask: Optional[jax.Array] = None, token_ids: Optional[jax.Array] = None, training: bool = True):
        # x: (B, T, N, 3)
        # mask: (B, T)
        
        # Preprocessing (Normalization) - Assuming preprocessed or doing it here?
        # mdl_2_pt does normalization inside forward.
        # We'll implement normalization in a separate helper or assume 'x' is raw and normalize here.
        # But JAX normalization on 3D data might be expensive to JIT if not careful?
        # Let's do feature extraction logic.
        
        if mask is None:
             mask = jnp.ones((x.shape[0], x.shape[1]), dtype=bool)
             
        # Extract Parts
        # x is (B, T, 130, 3)
        # Indices:
        # LHand: 0-21
        # RHand: 21-42
        # Pose: 42-75
        # Face: 75-130
        
        # We need flattening for extractors: (B, T, P*3)
        # Wait, FeatureExtractor expects (B, T, N, 3).
        def get_part(start, end):
             return x[:, :, start:end, :] # (B, T, P, 3)
             
        # Normalize parts (Match PyTorch: Center and Scale per part)
        # Normalize parts (Match PyTorch: Center and Scale per part)
        # Checkpoint structure: LHand(21), RHand(21), Pose(12), Face(76)
        # Ranges: 0-21, 21-42, 42-54, 54-130
        ranges = [(0, 21), (21, 42), (42, 54), (54, 130)]
        x_parts = []
        
        # Check dropped (sum mismatch)
        dropped_mask = (x[..., :2].sum(-1) == 0) # (B, T, N)
        
        for start_idx, end_idx in ranges:
            part = get_part(start_idx, end_idx) # (B, T, N_part, 3)
            mu = part.mean(axis=2, keepdims=True)
            std = part.std(axis=2, keepdims=True)
            # PyTorch does NOT use epsilon here, and handles NaNs/Infs later.
            # (feat - mean) / std. If std is 0, we get Inf/NaN.
            part_norm = (part - mu) / std 
            part_norm = jnp.nan_to_num(part_norm, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Apply dropped mask (set dropped landmarks to 0)
            m_part = dropped_mask[:, :, start_idx:end_idx] # (B, T, N_part)
            part_norm = part_norm * (1.0 - m_part[..., None].astype(part_norm.dtype))
            
            x_parts.append(part_norm)
            
        x_lhand = self.feature_extractor_lhand(x_parts[0], mask, training)
        x_rhand = self.feature_extractor_rhand(x_parts[1], mask, training)
        x_pose = self.feature_extractor_pose(x_parts[2], mask, training)
        x_face = self.feature_extractor_face(x_parts[3], mask, training)
        
        x_global = self.feature_extractor(x, mask, training)
        
        x1 = jnp.concatenate([x_lhand, x_rhand, x_face, x_pose], axis=-1)
        x_combined = x_global + x1
        
        # Get Cos/Sin
        cos, sin = self.rotary_emb(x_combined, seq_len=x_combined.shape[1])
        
        # Encoder
        enc_out = self.encoder(x_combined, cos, sin, mask=mask, training=training)
        
        # Decoder logic (just forward for now)
        logits = None
        if token_ids is not None:
             dec_out = self.decoder(token_ids, enc_out, encoder_attention_mask=mask, training=training)
             logits = self.fc(dec_out)
             
        # Aux logits
        # Aux logits
        # Match PT: aux_logits = self.aux_fc(x[:,0])
        aux_logits = self.aux_fc(enc_out[:, 0]).squeeze(-1) # (B, 1) -> (B)
        # if mask is not None:
        #     aux_logits = (aux_logits * mask).sum(1) / (mask.sum(1) + 1e-6)
        # else:
        #     aux_logits = aux_logits.mean(1)

        return logits, aux_logits
