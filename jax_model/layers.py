import jax
import jax.numpy as jnp
from flax import nnx
import math
from typing import Optional, Tuple, Callable

# --- Basic Activations ---

class Swish(nnx.Module):
    def __call__(self, x: jax.Array) -> jax.Array:
        return nnx.swish(x)

class GLU(nnx.Module):
    def __init__(self, dim: int = -1):
        self.dim = dim

    def __call__(self, x: jax.Array) -> jax.Array:
        out, gate = jnp.split(x, 2, axis=self.dim)
        return out * nnx.sigmoid(gate)

# --- Convolution Modules ---

class DepthwiseConv1d(nnx.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False, rngs: nnx.Rngs = None):
        assert out_channels % in_channels == 0, "out_channels must be a multiple of in_channels"
        self.padding = padding
        # flax.linen.Conv doesn't take separate padding arg in int, usually 'SAME' or 'VALID' or [(pad, pad)]
        # We'll handle padding manually if needed or translate int padding.
        # PyTorch padding=p adds p zeroes to both sides.
        pad_tuple = [(padding, padding)] 
        
        self.conv = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(kernel_size,),
            strides=(stride,),
            padding=pad_tuple,
            feature_group_count=in_channels,
            use_bias=bias,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        # x is (batch, time, channels) in Flax usually, but let's confirm input expectation.
        # PyTorch uses (batch, channels, time). Flax NNX Conv expects (batch, spatial..., features).
        # We will assume inputs are (batch, time, features) for JAX consistency.
        return self.conv(x)

class PointwiseConv1d(nnx.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, padding: int = 0, bias: bool = True, rngs: nnx.Rngs = None):
        # Pointwise is just Dense or Conv with kernel 1.
        # Using Conv to support stride/padding if strictly needed, though pointwise usually stride 1 pad 0.
        self.conv = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(1,),
            strides=(stride,),
            padding=[(padding, padding)],
            use_bias=bias,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.conv(x)

# --- Positional Encoding ---

class RelPositionalEncoding(nnx.Module):
    def __init__(self, d_model: int = 512, max_len: int = 5000):
        self.d_model = d_model
        self.max_len = max_len
        self.pe = None # Cache behavior in JAX is tricky, usually better to recompute or pass as state.
        # For simple JAX implementation, we might just compute it on the fly or use a constant.
        # Let's compute on call or cache if we make this stateful.
        # Since 'pe' is not a parameter (not trainable), we treat it as a constant or variable.
        self.pe_cache = nnx.Variable(jnp.zeros((1, max_len, d_model))) # Simple cache variable

    def extend_pe(self, x: jax.Array):
        # x: (batch, time, channels)
        length = x.shape[1]
        current_pe = self.pe_cache.value
        if current_pe.shape[1] >= length * 2 - 1:
             return

        # Compute PE
        pe_positive = jnp.zeros((length, self.d_model))
        pe_negative = jnp.zeros((length, self.d_model))
        position = jnp.arange(0, length, dtype=jnp.float32)[:, None]
        div_term = jnp.exp(
            jnp.arange(0, self.d_model, 2, dtype=jnp.float32) * -(math.log(10000.0) / self.d_model)
        )
        
        # Calculate sin/cos
        sin_pos = jnp.sin(position * div_term)
        cos_pos = jnp.cos(position * div_term)
        
        pe_positive = pe_positive.at[:, 0::2].set(sin_pos)
        pe_positive = pe_positive.at[:, 1::2].set(cos_pos)
        
        pe_negative = pe_negative.at[:, 0::2].set(jnp.sin(-1 * position * div_term))
        pe_negative = pe_negative.at[:, 1::2].set(jnp.cos(-1 * position * div_term))

        pe_positive = jnp.flip(pe_positive, axis=0)[None] # (1, length, d_model)
        pe_negative = pe_negative[1:][None] # (1, length-1, d_model)
        
        new_pe = jnp.concatenate([pe_positive, pe_negative], axis=1)
        self.pe_cache.value = new_pe # Update cache

    def __call__(self, x: jax.Array) -> jax.Array:
        self.extend_pe(x)
        pe = self.pe_cache.value
        # Slicing logic from PyTorch:
        # self.pe[:, self.pe.size(1) // 2 - x.size(1) + 1 : self.pe.size(1) // 2 + x.size(1)]
        center = pe.shape[1] // 2
        start = center - x.shape[1] + 1
        end = center + x.shape[1]
        return pe[:, start:end]

# --- Feed Forward ---

class FeedForwardModule(nnx.Module):
    def __init__(self, encoder_dim: int = 512, expansion_factor: int = 4, dropout_p: float = 0.1, rngs: nnx.Rngs = None):
        self.ffn1 = nnx.Linear(encoder_dim, encoder_dim * expansion_factor, rngs=rngs)
        self.act = Swish()
        self.do1 = nnx.Dropout(dropout_p, rngs=rngs)
        self.ffn2 = nnx.Linear(encoder_dim * expansion_factor, encoder_dim, rngs=rngs)
        self.do2 = nnx.Dropout(dropout_p, rngs=rngs)

    def __call__(self, x: jax.Array, training: bool = True) -> jax.Array:
        x = self.ffn1(x)
        x = self.act(x)
        x = self.do1(x, deterministic=not training)
        x = self.ffn2(x)
        x = self.do2(x, deterministic=not training)
        return x

# --- Relative Multi Head Attention ---

class RelativeMultiHeadAttention(nnx.Module):
    def __init__(self, d_model: int = 512, num_heads: int = 16, dropout_p: float = 0.1, rngs: nnx.Rngs = None):
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."
        self.d_model = d_model
        self.d_head = d_model // num_heads
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(self.d_head)

        self.query_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.key_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.value_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.pos_proj = nnx.Linear(d_model, d_model, use_bias=False, rngs=rngs)
        
        self.dropout = nnx.Dropout(dropout_p, rngs=rngs)
        self.out_proj = nnx.Linear(d_model, d_model, rngs=rngs)

        # Trainable bias parameters
        # PyTorch: self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.u_bias = nnx.Param(jax.random.uniform(rngs.params(), (self.num_heads, self.d_head)))
        self.v_bias = nnx.Param(jax.random.uniform(rngs.params(), (self.num_heads, self.d_head)))
        # Note: Initialization should ideally encompass xavier_uniform, here using default uniform for simplicity placeholder.
        # We can implement custom init if needed.

    def _relative_shift(self, pos_score: jax.Array) -> jax.Array:
        # pos_score: (B, H, T, 2*T - 1)
        B, H, T, width = pos_score.shape
        # We want P[i-j]. Center is at T-1 in the width dimension.
        # indices[i, j] = (T - 1) + (i - j)
        
        # arange(T)
        range_vec = jnp.arange(T)
        # i[:, None] - j[None, :]
        diff = range_vec[:, None] - range_vec[None, :]
        indices = (T - 1) + diff # (T, T)
        
        # Broadcast to (B, H, T, T)
        indices = indices[None, None, :, :]
        indices = jnp.broadcast_to(indices, (B, H, T, T))
        
        # Gather
        # take_along_axis on last axis.
        # Input: (B, H, T, width). Indices: (B, H, T, T). Output: (B, H, T, T).
        return jnp.take_along_axis(pos_score, indices, axis=-1)

    def __call__(self, query: jax.Array, key: jax.Array, value: jax.Array, pos_embedding: jax.Array, mask: Optional[jax.Array] = None, training: bool = True) -> jax.Array:
        batch_size = value.shape[0]
        
        # (B, T, D) -> (B, T, H, K) -> (B, H, T, K)
        query = self.query_proj(query).reshape(batch_size, -1, self.num_heads, self.d_head).transpose(0, 2, 1, 3)
        key = self.key_proj(key).reshape(batch_size, -1, self.num_heads, self.d_head).transpose(0, 2, 1, 3)
        value = self.value_proj(value).reshape(batch_size, -1, self.num_heads, self.d_head).transpose(0, 2, 1, 3)
        pos_embedding = self.pos_proj(pos_embedding).reshape(batch_size, -1, self.num_heads, self.d_head).transpose(0, 2, 1, 3)

        # Content score: (Q + u) * K^T
        # u_bias is (H, D_h). We need to broadcast to (B, H, T, D_h)
        u = self.u_bias.value[None, :, None, :] # (1, H, 1, K)
        content_score = jnp.matmul((query + u), key.transpose(0, 1, 3, 2)) # (B, H, T, T)

        # Pos score: (Q + v) * P^T
        v = self.v_bias.value[None, :, None, :]
        # pos_embedding is (B, H, 2T-1, K) usually? Or aligned?
        # In PyTorch code: permute(0, 2, 3, 1) -> (B, H, K, T_pos)
        pos_score = jnp.matmul((query + v), pos_embedding.transpose(0, 1, 3, 2)) # (B, H, T, T_pos)
        
        pos_score = self._relative_shift(pos_score)

        score = (content_score + pos_score) / self.sqrt_dim

        if mask is not None:
            # mask is (B, 1, T) or (B, T, T), broadcast to (B, 1, T, T) or similar
            # If mask is boolean, use where
            min_value = -1e9
            score = jnp.where(mask, score, min_value) # Auto-broadcast

        attn = nnx.softmax(score, axis=-1)
        attn = self.dropout(attn, deterministic=not training)

        context = jnp.matmul(attn, value) # (B, H, T, K)
        
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.d_model) # (B, T, D)

        return self.out_proj(context)

# --- Other Modules ---

class MultiHeadedSelfAttentionModule(nnx.Module):
    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.1, rngs: nnx.Rngs = None):
        self.positional_encoding = RelPositionalEncoding(d_model)
        self.attention = RelativeMultiHeadAttention(d_model, num_heads, dropout_p, rngs=rngs)
        self.dropout = nnx.Dropout(dropout_p, rngs=rngs)

    def __call__(self, inputs: jax.Array, mask: Optional[jax.Array] = None, training: bool = True) -> jax.Array:
        # inputs: (B, T, D)
        batch_size = inputs.shape[0]
        pos_embedding = self.positional_encoding(inputs)
        # Repeat pos_embedding for batch if it's singleton? 
        # The RelPosEnc returns (1, T_pos, D) or similar? 
        # PyTorch: pos_embedding.repeat(batch_size, 1, 1)
        if pos_embedding.shape[0] == 1:
            pos_embedding = jnp.repeat(pos_embedding, batch_size, axis=0)

        outputs = self.attention(inputs, inputs, inputs, pos_embedding=pos_embedding, mask=mask, training=training)
        return self.dropout(outputs, deterministic=not training)

class MaskedBatchNorm(nnx.Module):
    # CRITICAL: Batch Norm that ignores padded values for stats.
    def __init__(self, num_features: int, momentum: float = 0.9, epsilon: float = 1e-5, rngs: nnx.Rngs = None):
        self.num_features = num_features
        self.momentum = momentum
        self.epsilon = epsilon
        
        # Parameters
        self.scale = nnx.Param(jnp.ones((num_features,)))
        self.bias = nnx.Param(jnp.zeros((num_features,)))
        
        # Running stats
        self.running_mean = nnx.BatchStat(jnp.zeros((num_features,)))
        self.running_var = nnx.BatchStat(jnp.ones((num_features,)))

    def __call__(self, x: jax.Array, mask: Optional[jax.Array] = None, training: bool = True) -> jax.Array:
        # x: (B, T, C) or (B, C)
        # mask: (B, T) boolean
        
        if training:
            if mask is not None:
                # Compute stats only on valid tokens
                # x[mask] in JAX is problematic for JIT if mask is dynamic.
                # However, we can use weighted mean/var.
                mask_bc = mask[..., None] # (B, T, 1)
                valid_count = mask_bc.sum()
                
                # Mean
                mean = (x * mask_bc).sum(axis=(0, 1)) / (valid_count + 1e-6)
                
                # Var
                # sum((x - mean)^2 * mask) / count
                var = ((x - mean[None, None, :]) ** 2 * mask_bc).sum(axis=(0, 1)) / (valid_count + 1e-6)
            else:
                mean = x.mean(axis=(0, 1))
                var = x.var(axis=(0, 1))

            # Update running stats
            # Note: PyTorch momentum is 0.1 usually (1 - 0.9). Flax uses decay (0.9).
            # We implemented PyTorch-like momentum arg.
            decay = self.momentum
            self.running_mean.value = decay * self.running_mean.value + (1 - decay) * mean
            self.running_var.value = decay * self.running_var.value + (1 - decay) * var
            
            # Use batch stats for normalization
            norm_mean = mean
            norm_var = var
        else:
            norm_mean = self.running_mean.value
            norm_var = self.running_var.value

        # Normalize
        out = (x - norm_mean[None, None, :]) / jnp.sqrt(norm_var[None, None, :] + self.epsilon)
        out = out * self.scale.value[None, None, :] + self.bias.value[None, None, :]
        
        if mask is not None:
            out = out * mask[..., None] # Zero out padding again just in case
            
        return out

class ConvModule(nnx.Module):
    def __init__(self, in_channels: int, kernel_size: int = 31, expansion_factor: int = 2, dropout_p: float = 0.1, rngs: nnx.Rngs = None):
        self.pw_conv_1 = PointwiseConv1d(in_channels, in_channels * expansion_factor, stride=1, padding=0, bias=True, rngs=rngs)
        self.act1 = GLU(dim=-1) # Last dim is channels in JAX (NHWC logic)
        self.dw_conv = DepthwiseConv1d(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2, rngs=rngs)
        self.bn = MaskedBatchNorm(in_channels, rngs=rngs)
        self.act2 = Swish()
        self.pw_conv_2 = PointwiseConv1d(in_channels, in_channels, stride=1, padding=0, bias=True, rngs=rngs)
        self.do = nnx.Dropout(dropout_p, rngs=rngs)

    def __call__(self, x: jax.Array, mask_pad: Optional[jax.Array] = None, training: bool = True) -> jax.Array:
        # mask_pad: (B, 1, T) or similar. 
        # JAX ConvModule usually expects (B, T, C)
        
        x = self.pw_conv_1(x)
        x = self.act1(x)
        x = self.dw_conv(x)
        
        # BatchNorm with mask
        # Note: PyTorch permutes (B, C, T) -> (B, T, C) for some steps.
        # We stay in (B, T, C) mostly.
        mask_flat = None
        if mask_pad is not None:
             # mask_pad might be (B, 1, T) or (B, T). Squeeze to (B, T)
             if mask_pad.ndim == 3:
                 mask_flat = mask_pad.squeeze(1)
             else:
                 mask_flat = mask_pad
             
             mask_flat = mask_flat.astype(bool)

        x = self.bn(x, mask=mask_flat, training=training)
        x = self.act2(x)
        x = self.pw_conv_2(x)
        x = self.do(x, deterministic=not training)
        
        if mask_flat is not None:
            x = x * mask_flat[..., None]
        
        return x

class SqueezeformerBlock(nnx.Module):
    def __init__(self, encoder_dim: int, num_attention_heads: int, feed_forward_expansion_factor: int, conv_expansion_factor: int, feed_forward_dropout_p: float, attention_dropout_p: float, conv_dropout_p: float, conv_kernel_size: int, rngs: nnx.Rngs = None):
        
        # Scaling params
        self.scale_mhsa = nnx.Param(jnp.ones((1, 1, encoder_dim)))
        self.bias_mhsa = nnx.Param(jnp.zeros((1, 1, encoder_dim)))
        
        self.scale_ff_mhsa = nnx.Param(jnp.ones((1, 1, encoder_dim)))
        self.bias_ff_mhsa = nnx.Param(jnp.zeros((1, 1, encoder_dim)))
        
        self.scale_conv = nnx.Param(jnp.ones((1, 1, encoder_dim)))
        self.bias_conv = nnx.Param(jnp.zeros((1, 1, encoder_dim)))
        
        self.scale_ff_conv = nnx.Param(jnp.ones((1, 1, encoder_dim)))
        self.bias_ff_conv = nnx.Param(jnp.zeros((1, 1, encoder_dim)))

        self.mhsa = MultiHeadedSelfAttentionModule(encoder_dim, num_attention_heads, attention_dropout_p, rngs=rngs)
        self.ln_mhsa = nnx.LayerNorm(encoder_dim, rngs=rngs)
        
        self.ff_mhsa = FeedForwardModule(encoder_dim, feed_forward_expansion_factor, feed_forward_dropout_p, rngs=rngs)
        self.ln_ff_mhsa = nnx.LayerNorm(encoder_dim, rngs=rngs)
        
        self.conv = ConvModule(encoder_dim, conv_kernel_size, conv_expansion_factor, conv_dropout_p, rngs=rngs)
        self.ln_conv = nnx.LayerNorm(encoder_dim, rngs=rngs)
        
        self.ff_conv = FeedForwardModule(encoder_dim, feed_forward_expansion_factor, feed_forward_dropout_p, rngs=rngs)
        self.ln_ff_conv = nnx.LayerNorm(encoder_dim, rngs=rngs)

    def __call__(self, x: jax.Array, mask: Optional[jax.Array] = None, training: bool = True) -> jax.Array:
        # mask: (B, T) or (B, 1, T) boolean
        # Pre-Norm architecture with "Post-Norm" scaling residuals? 
        # PyTorch code: x = x * scale + bias; x = residual + module(x)
        # Wait, PyTorch code:
        # residual = x
        # x = x * self.scale_mhsa + self.bias_mhsa
        # x = residual + self.mhsa(x)
        # x = self.ln_mhsa(x)
        
        # Block 1: MHSA
        residual = x
        x = x * self.scale_mhsa + self.bias_mhsa
        # MHSA expects mask for attention scores (B, 1, T, T) usually
        # We need to construct attention mask from padding mask
        attn_mask = None
        if mask is not None:
            # mask is (B, T) or (B, 1, T). Self-attention mask (B, 1, T, T)
            # mask_pad = ~( mask_pad.permute(0, 2,1) * mask_pad) in PyTorch
            # JAX: mask (B, 1, T) -> (B, 1, 1, T) broadcast against (B, 1, T, 1)?
            # Standard causal/padding mask: (B, 1, 1, T) is enough for "key" masking.
            # But SqueezeFormer uses (B, 1, T, T) logic?
            # Let's assume input 'mask' is simple padding mask (B, T) where True is valid.
            m = mask[:, None, None, :] # (B, 1, 1, T)
            attn_mask = m # Broadcasts effectively
            
        x = residual + self.mhsa(x, mask=attn_mask, training=training)
        x = self.ln_mhsa(x)
        
        # Block 2: FF MHSA
        residual = x
        x = x * self.scale_ff_mhsa + self.bias_ff_mhsa
        x = residual + self.ff_mhsa(x, training=training)
        x = self.ln_ff_mhsa(x)
        
        # Block 3: Conv
        residual = x
        x = x * self.scale_conv + self.bias_conv
        
        # Pass mask to Conv for MaskedBN
        # mask for ConvModule: (B, 1, T) or (B, T)
        conv_mask = mask.squeeze() if mask is not None else None
        if conv_mask is not None and conv_mask.ndim == 1: conv_mask = conv_mask[None, :] # Handle single batch
        
        x = residual + self.conv(x, mask_pad=mask, training=training)
        x = self.ln_conv(x)
        
        # Block 4: FF Conv
        residual = x
        x = x * self.scale_ff_conv + self.bias_ff_conv
        x = residual + self.ff_conv(x, training=training)
        x = self.ln_ff_conv(x)
        
        return x
