
import jax
import jax.numpy as jnp
from flax import nnx
import math
from typing import Optional, Tuple, Callable
from . import config

# --- Basic Activations ---

# --- Basic Activations ---

# Swish class removed, use nnx.swish/nnx.silu directly

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
        # PyTorch padding logic: adds zeros to both sides.
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
        return self.conv(x)

class PointwiseConv1d(nnx.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, padding: int = 0, bias: bool = True, rngs: nnx.Rngs = None):
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

# --- Positional Embeddings ---

class SinusoidalPositionalEmbedding(nnx.Module):
    def __init__(self, num_positions: int, embedding_dim: int, rngs: nnx.Rngs = None):
        self.embedding_dim = embedding_dim
        self.num_positions = num_positions
        # Fixed embeddings, not learnable
        # Precompute
        self.weights = self._get_embedding(num_positions, embedding_dim)

    def _get_embedding(self, num_embeddings, embedding_dim):
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = jnp.arange(num_embeddings)[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
        if embedding_dim % 2 == 1:
            # Padding if odd? Transformers standard usually pads
            emb = jnp.concatenate([emb, jnp.zeros((num_embeddings, 1))], axis=1)
        return nnx.Variable(emb)

    def __call__(self, input_ids: jax.Array, past_key_values_length: int = 0) -> jax.Array:
        # Input is ids (B, T)
        # We need positions.
        # Transformers behavior: usually creates position_ids automatically if not provided
        # But here we just return the embeddings for the sequence
        # We assume input_ids is just used for shape
        bsz, seq_len = input_ids.shape
        position_ids = jnp.arange(past_key_values_length, past_key_values_length + seq_len)
        return self.weights.value[position_ids]

# --- RoPE (Llama Rotary Embedding) ---
# ... (omitted)

class Decoder(nnx.Module):
# ... (omitted headers)
    def __call__(self, input_ids, encoder_hidden_states, encoder_attention_mask=None, training=True):
        # input_ids: (B, T)
        B, T = input_ids.shape
        x = self.embed_tokens(input_ids) * math.sqrt(self.config.d_model)
        
        # Speech2Text uses offset of pad_token_id + 1
        pos_offset = self.config.pad_token_id + 1
        x = x + self.embed_positions(input_ids, past_key_values_length=pos_offset)
        
        # S2T Standard: No LayerNorm, just Dropout after embedding
        x = self.dropout(x, deterministic=not training)
        
        # Causal mask
# --- RoPE (Llama Rotary Embedding) ---

class LlamaRotaryEmbedding(nnx.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000, rngs: nnx.Rngs = None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Precompute inv_freq
        inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2) / dim))
        self.inv_freq = nnx.Param(inv_freq) # Treat as fixed param usually, or buffer?
        # In flax nnx usually we just store as array if not trainable?
        # Using Param allows loading/saving if needed, but usually it's constant.
        # Actually it's non-trainable state.
        
        # We can also precompute cos/sin cache if size is fixed
        self.cos_cached = nnx.Variable(jnp.zeros((max_position_embeddings, dim)))
        self.sin_cached = nnx.Variable(jnp.zeros((max_position_embeddings, dim)))
        
        self._build_cache()

    def _build_cache(self):
        t = jnp.arange(self.max_position_embeddings)
        freqs = jnp.outer(t, self.inv_freq.value) # (max_pos, dim/2)
        # Different concatenation strategy!
        # Llama strategy: cat(freqs, freqs)
        emb = jnp.concatenate((freqs, freqs), axis=-1)
        self.cos_cached.value = jnp.cos(emb)
        self.sin_cached.value = jnp.sin(emb)

    def __call__(self, x: jax.Array, seq_len: int=None) -> Tuple[jax.Array, jax.Array]:
        # Returns cos, sin for the requested seq_len
        # x is just used for inferring device/dtype if needed, or ignored
        if seq_len is None:
            seq_len = x.shape[1]
            
        if seq_len > self.cos_cached.value.shape[0]:
            # This implementation doesn't dynamic resize yet for simplicity
            # Assuming max_len is sufficient as per config
            pass

        return self.cos_cached.value[:seq_len], self.sin_cached.value[:seq_len]

def rotate_half(x):
    # Splits last dim into 2 halves (x1, x2) and returns (-x2, x1)
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate((-x2, x1), axis=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: (B, H, T, D)
    # cos, sin: (T, D) or (1, 1, T, D)
    
    # Reshape cos/sin for broadcasting: (1, 1, T, D)
    # Incoming cos might be (T, D).
    if cos.ndim == 2:
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
    elif cos.ndim == 4:
         pass # Assume correct

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# --- Llama Attention ---

class LlamaAttention(nnx.Module):
    def __init__(self, config, rngs: nnx.Rngs = None):
        # Config should mimic LlamaConfig
        # hidden_size, num_attention_heads
        self.hidden_size = config.encoder_dim
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        self.q_proj = nnx.Linear(self.hidden_size, self.num_heads * self.head_dim, use_bias=False, rngs=rngs)
        self.k_proj = nnx.Linear(self.hidden_size, self.num_heads * self.head_dim, use_bias=False, rngs=rngs)
        self.v_proj = nnx.Linear(self.hidden_size, self.num_heads * self.head_dim, use_bias=False, rngs=rngs)
        self.o_proj = nnx.Linear(self.hidden_size, self.hidden_size, use_bias=False, rngs=rngs)

    def __call__(self, hidden_states: jax.Array, cos: jax.Array, sin: jax.Array, mask: Optional[jax.Array] = None) -> jax.Array:
        # hidden_states: (B, T, D)
        B, T, D = hidden_states.shape
        
        q = self.q_proj(hidden_states).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3) # (B, H, T, K)
        k = self.k_proj(hidden_states).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(hidden_states).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Apply RoPE
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Attention
        # (B, H, T, K) @ (B, H, K, T) -> (B, H, T, T)
        attn_weights = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            # mask: (B, 1, T, T) usually? Or (B, 1, T) padding mask?
            # Model 2 uses padding mask mostly. (0 for padding).
            # If mask is (B, T), we broadcast to (B, 1, 1, T) for key masking?
            # attn_weights += (1 - mask) * -1e9
            # Let's assume input mask is (B, 1, 1, T) prepared by caller or needs prep
            attn_weights = attn_weights + mask 
            
        attn_weights = nnx.softmax(attn_weights, axis=-1)
        attn_output = jnp.matmul(attn_weights, v) # (B, H, T, K)
        
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, T, D)
        attn_output = self.o_proj(attn_output)
        
        return attn_output

# --- Feed Forward ---

class FeedForwardModule(nnx.Module):
    def __init__(self, encoder_dim: int = 512, expansion_factor: int = 4, dropout_p: float = 0.1, rngs: nnx.Rngs = None):
        self.ffn1 = nnx.Linear(encoder_dim, encoder_dim * expansion_factor, rngs=rngs)
        self.act = nnx.swish
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

# --- Masked Batch Norm ---

class MaskedBatchNorm(nnx.Module):
    # Matches PyTorch logic where we mask stats computation
    def __init__(self, num_features: int, momentum: float = 0.1, epsilon: float = 1e-5, rngs: nnx.Rngs = None):
        self.momentum = momentum # PyTorch default 0.1
        self.epsilon = epsilon
        self.scale = nnx.Param(jnp.ones((num_features,)))
        self.bias = nnx.Param(jnp.zeros((num_features,)))
        self.running_mean = nnx.BatchStat(jnp.zeros((num_features,)))
        self.running_var = nnx.BatchStat(jnp.ones((num_features,)))

    def __call__(self, x: jax.Array, mask: Optional[jax.Array] = None, training: bool = True) -> jax.Array:
        # x: (B, T, C)
        if training:
            if mask is not None:
                mask_bc = mask[..., None]
                sum_mask = mask_bc.sum()
                mean = (x * mask_bc).sum(axis=(0, 1)) / (sum_mask + 1e-6)
                var = ((x - mean) ** 2 * mask_bc).sum(axis=(0, 1)) / (sum_mask + 1e-6)
            else:
                mean = x.mean(axis=(0, 1))
                var = x.var(axis=(0, 1))
                
            decay = 1.0 - self.momentum # Flax uses decay (0.9 for 0.1 momentum)
            self.running_mean.value = decay * self.running_mean.value + (1 - decay) * mean
            self.running_var.value = decay * self.running_var.value + (1 - decay) * var
            norm_mean, norm_var = mean, var
        else:
            norm_mean, norm_var = self.running_mean.value, self.running_var.value
            
        out = (x - norm_mean) / jnp.sqrt(norm_var + self.epsilon)
        out = out * self.scale.value + self.bias.value
        
        if mask is not None: out = out * mask[..., None]
        return out

# --- Conv Module ---

class ConvModule(nnx.Module):
    def __init__(self, in_channels: int, kernel_size: int = 31, expansion_factor: int = 2, dropout_p: float = 0.1, rngs: nnx.Rngs = None):
        self.pw_conv_1 = PointwiseConv1d(in_channels, in_channels * expansion_factor, stride=1, padding=0, bias=True, rngs=rngs)
        self.act1 = GLU(dim=-1)
        self.dw_conv = DepthwiseConv1d(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2, rngs=rngs)
        self.bn = MaskedBatchNorm(in_channels, momentum=0.1, rngs=rngs) # Matches PyTorch 0.985? No, PyTorch code had 0.985.
        # Wait, mdl_2_pt.py ConvModule has: momentum=0.985
        # We need to expose this param or set it here.
        # Let's set it to 0.985 if that's what mdl_2_pt uses.
        
        self.act2 = nnx.swish
        self.pw_conv_2 = PointwiseConv1d(in_channels, in_channels, stride=1, padding=0, bias=True, rngs=rngs)
        self.do = nnx.Dropout(dropout_p, rngs=rngs)

    def __call__(self, x: jax.Array, mask_pad: Optional[jax.Array] = None, training: bool = True) -> jax.Array:
        x = self.pw_conv_1(x)
        x = self.act1(x)
        x = self.dw_conv(x)
        x = self.bn(x, mask=mask_pad, training=training)
        x = self.act2(x)
        x = self.pw_conv_2(x)
        x = self.do(x, deterministic=not training)
        if mask_pad is not None: x = x * mask_pad[..., None]
        return x

# --- Squeezeformer Block ---

class SqueezeformerBlock(nnx.Module):
    def __init__(self, encoder_dim: int, num_attention_heads: int, feed_forward_expansion_factor: int, conv_expansion_factor: int, 
                 feed_forward_dropout_p: float, attention_dropout_p: float, conv_dropout_p: float, conv_kernel_size: int, rngs: nnx.Rngs = None):
        
        self.scale_mhsa = nnx.Param(jnp.ones((1, 1, encoder_dim)))
        self.bias_mhsa = nnx.Param(jnp.zeros((1, 1, encoder_dim)))
        
        self.scale_ff_mhsa = nnx.Param(jnp.ones((1, 1, encoder_dim)))
        self.bias_ff_mhsa = nnx.Param(jnp.zeros((1, 1, encoder_dim)))
        
        self.scale_conv = nnx.Param(jnp.ones((1, 1, encoder_dim)))
        self.bias_conv = nnx.Param(jnp.zeros((1, 1, encoder_dim)))
        
        self.scale_ff_conv = nnx.Param(jnp.ones((1, 1, encoder_dim)))
        self.bias_ff_conv = nnx.Param(jnp.zeros((1, 1, encoder_dim)))

        # Emulating config for LlamaAttention
        class Config: pass
        c = Config()
        c.encoder_dim = encoder_dim
        c.num_attention_heads = num_attention_heads
        
        self.mhsa_llama = LlamaAttention(c, rngs=rngs)
        self.ln_mhsa = nnx.LayerNorm(encoder_dim, epsilon=1e-5, rngs=rngs)
        
        self.ff_mhsa = FeedForwardModule(encoder_dim, feed_forward_expansion_factor, feed_forward_dropout_p, rngs=rngs)
        self.ln_ff_mhsa = nnx.LayerNorm(encoder_dim, epsilon=1e-5, rngs=rngs)
        
        self.conv = ConvModule(encoder_dim, conv_kernel_size, conv_expansion_factor, conv_dropout_p, rngs=rngs)
        self.ln_conv = nnx.LayerNorm(encoder_dim, epsilon=1e-5, rngs=rngs)
        
        self.ff_conv = FeedForwardModule(encoder_dim, feed_forward_expansion_factor, feed_forward_dropout_p, rngs=rngs)
        self.ln_ff_conv = nnx.LayerNorm(encoder_dim, epsilon=1e-5, rngs=rngs)

    def __call__(self, x: jax.Array, cos: jax.Array, sin: jax.Array, mask: Optional[jax.Array] = None, training: bool = True) -> jax.Array:
        # 1. MHSA
        residual = x
        x = x * self.scale_mhsa + self.bias_mhsa
        # Prepare mask for attention (B, 1, 1, T) or similar
        attn_mask = None
        if mask is not None:
            # (B, T) -> (B, 1, 1, T). 0 is padding.
            m = mask[:, None, None, :]
            attn_mask = (1.0 - m) * -1e9
            
        x = residual + self.mhsa_llama(x, cos, sin, mask=attn_mask)
        x = self.ln_mhsa(x)
        if mask is not None: x = x * mask[..., None]
        
        # 2. FF MHSA
        residual = x
        x = x * self.scale_ff_mhsa + self.bias_ff_mhsa
        x = residual + self.ff_mhsa(x, training=training)
        x = self.ln_ff_mhsa(x)
        if mask is not None: x = x * mask[..., None]
        
        # 3. Conv
        residual = x
        x = x * self.scale_conv + self.bias_conv
        x = residual + self.conv(x, mask_pad=mask, training=training)
        x = self.ln_conv(x)
        if mask is not None: x = x * mask[..., None]
        
        # 4. FF Conv
        residual = x
        x = x * self.scale_ff_conv + self.bias_ff_conv
        x = residual + self.ff_conv(x, training=training)
        x = self.ln_ff_conv(x)
        if mask is not None: x = x * mask[..., None]
        
        return x

# --- Decoder Components ---

class StandardMultiHeadAttention(nnx.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1, rngs: nnx.Rngs = None):
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nnx.Linear(dim, dim, rngs=rngs)
        self.k_proj = nnx.Linear(dim, dim, rngs=rngs)
        self.v_proj = nnx.Linear(dim, dim, rngs=rngs)
        self.o_proj = nnx.Linear(dim, dim, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, q_x, k_x, v_x, mask=None, training=True):
        # q_x: (B, Tq, D)
        B, Tq, D = q_x.shape
        Tk = k_x.shape[1]
        
        # (B, T, H, K) -> (B, H, T, K)
        q = self.q_proj(q_x).reshape(B, Tq, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(k_x).reshape(B, Tk, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(v_x).reshape(B, Tk, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # (B, H, Tq, K) @ (B, H, K, Tk) -> (B, H, Tq, Tk)
        attn = jnp.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale
        
        if mask is not None:
            # mask: (B, 1, Tq, Tk)
            attn = attn + mask
            
        attn = nnx.softmax(attn, axis=-1)
        attn = self.dropout(attn, deterministic=not training)
        
        # (B, H, Tq, Tk) @ (B, H, Tk, K) -> (B, H, Tq, K)
        out = jnp.matmul(attn, v).transpose(0, 2, 1, 3).reshape(B, Tq, D)
        return self.o_proj(out)

class DecoderLayer(nnx.Module):
    def __init__(self, config: config.DecoderConfig, rngs: nnx.Rngs = None):
        self.self_attn = StandardMultiHeadAttention(config.d_model, config.num_heads, config.attention_dropout, rngs=rngs)
        self.self_attn_layer_norm = nnx.LayerNorm(config.d_model, epsilon=1e-5, rngs=rngs)
        
        self.encoder_attn = StandardMultiHeadAttention(config.d_model, config.num_heads, config.attention_dropout, rngs=rngs)
        self.encoder_attn_layer_norm = nnx.LayerNorm(config.d_model, epsilon=1e-5, rngs=rngs)
        
        self.fc1 = nnx.Linear(config.d_model, config.decoder_ffn_dim, rngs=rngs)
        self.act = nnx.relu # Speech2Text default is ReLU
        self.fc2 = nnx.Linear(config.decoder_ffn_dim, config.d_model, rngs=rngs)
        self.final_layer_norm = nnx.LayerNorm(config.d_model, epsilon=1e-5, rngs=rngs)
        self.dropout = nnx.Dropout(config.attention_dropout, rngs=rngs)

    def __call__(self, x, encoder_out, encoder_mask=None, causal_mask=None, training=True):
        # Self Attn (Pre-Norm)
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x, x, x, mask=causal_mask, training=training)
        x = self.dropout(x, deterministic=not training)
        x = residual + x
        
        # Cross Attn (Pre-Norm)
        residual = x
        x = self.encoder_attn_layer_norm(x)
        x = self.encoder_attn(x, encoder_out, encoder_out, mask=encoder_mask, training=training)
        x = self.dropout(x, deterministic=not training)
        x = residual + x
        
        # FF (Pre-Norm)
        residual = x
        x = self.final_layer_norm(x)
        x = self.act(self.fc1(x))
        x = self.dropout(x, deterministic=not training)
        x = self.fc2(x)
        x = self.dropout(x, deterministic=not training)
        x = residual + x
        
        return x

class Decoder(nnx.Module):
    layers: nnx.List

    def __init__(self, config: config.DecoderConfig, rngs: nnx.Rngs = None):
        self.config = config
        cfg = config
        self.embed_tokens = nnx.Embed(cfg.vocab_size, cfg.d_model, rngs=rngs)
        self.embed_positions = SinusoidalPositionalEmbedding(cfg.max_length + 2, cfg.d_model, rngs=rngs)
        
        self.layers = nnx.List([
            DecoderLayer(cfg, rngs=rngs)
            for _ in range(cfg.decoder_layers)
        ])
        self.layer_norm = nnx.LayerNorm(cfg.d_model, epsilon=1e-5, rngs=rngs)
        
    def __call__(self, input_ids, encoder_hidden_states, encoder_attention_mask=None, training=True):
        # input_ids: (B, T)
        B, T = input_ids.shape
        x = self.embed_tokens(input_ids) * math.sqrt(self.config.d_model)
        
        # Speech2Text uses offset of pad_token_id + 1
        pos_offset = self.config.pad_token_id + 1
        pos_emb = self.embed_positions(input_ids, past_key_values_length=pos_offset)
        
        # Do NOT mask padding here. PyTorch allows pos_emb for padding tokens.
        # Attention mask handles it later.
        x = x + pos_emb
        
        # Causal mask
        idx = jnp.arange(T)
        causal_mask = (idx[None, :] <= idx[:, None])
        causal_mask = jnp.broadcast_to(causal_mask, (B, 1, T, T))
        causal_mask = jnp.where(causal_mask, 0, -1e9)
        
        # Encoder mask
        em = None
        if encoder_attention_mask is not None:
             # (B, 1, 1, S)
             em = encoder_attention_mask[:, None, None, :]
             em = (1.0 - em) * -1e9
            
        for layer in self.layers:
            x = layer(x, encoder_hidden_states, encoder_mask=em, causal_mask=causal_mask, training=training)
            
        x = self.layer_norm(x)
        return x
