
import jax
import jax.numpy as jnp
from flax import nnx

# JIT-compatible Preprocessing

def normalize(x):
    # x: (T, N, 3)
    # Handle NaNs by masking.
    
    # Check for NaNs
    mask = ~jnp.isnan(x) # (T, N, 3)
    
    # We want to compute mean/std over valid points.
    # Flatten to simplify: (T*N, 3)
    # But strictly, if we want to match PyTorch `x[~torch.isnan(x)]`, it flattens everything.
    # PyTorch: x[mask].view(-1, 3).
    # This implies valid elements must be divisible by 3? 
    # Usually NaNs are per-point (all 3 coords are NaN or none).
    # If partial NaNs exist, PyTorch `.view(-1, 3)` might fail or mix coords if not aligned.
    # We assume NaNs are per-point for safety or just compute per-channel stats.
    
    # Independent channel stats (matches typical normalization):
    # Mean/Std per channel (0, 1, 2) over T and N.
    
    # Mask for valid values per channel
    valid_count = mask.sum(axis=(0, 1)) # (3,)
    valid_count = jnp.maximum(valid_count, 1.0)
    
    # Replace NaNs with 0 for sum
    x_zeroed = jnp.where(mask, x, 0.0)
    
    mean = x_zeroed.sum(axis=(0, 1)) / valid_count # (3,)
    
    # Std: sqrt( mean((x-mean)^2) )
    diff_sq = jnp.square(x_zeroed - mean)
    # Mask out invalids again (x_zeroed - mean might be non-zero where data was 0 if mean != 0)
    diff_sq = jnp.where(mask, diff_sq, 0.0)
    
    var = diff_sq.sum(axis=(0, 1)) / valid_count
    std = jnp.sqrt(var)
    
    x_out = (x - mean) / (std + 1e-6)
    
    # Restore NaNs? Or fill them?
    # PyTorch implementation fills NaNs with 0 via `torch.nan_to_num` later.
    # Here we return NaNs where input was NaN, or we can fill them now.
    # The original code `x = (x - mean) / std`. NaNs propagate.
    return x_out

def fill_nans(x):
    return jnp.nan_to_num(x, nan=0.0)

def resize_1d(data, new_len):
    # data: (T, C) -> (new_len, C)
    # Linear interpolation
    T = data.shape[0]
    C = data.shape[1]
    
    old_indices = jnp.arange(T)
    new_indices = jnp.linspace(0, T - 1, new_len)
    
    # Vectorized interp? jnp.interp works on 1D fp and 1D xp.
    # We need vmap over channels.
    def interp_channel(chan_data):
        return jnp.interp(new_indices, old_indices, chan_data)
        
    # vmap over C axis (1)
    data_new = jax.vmap(interp_channel, in_axes=1, out_axes=1)(data)
    return data_new

def interpolate_or_pad(data, max_len=384):
    # data: (T, N, 3)
    T = data.shape[0]
    N = data.shape[1]
    
    # Flatten N*3 for resizing
    data_flat = data.reshape(T, -1) # (T, N*3)
    
    # Branching for JIT: jax.lax.cond
    # If T < max_len: Pad
    # If T > max_len: Resize (Interpolate) -- Wait, original code said "diff <= 0" (T >= max_len) -> Resample.
    # If T < max_len: Pad.
    
    # But JIT requires static shapes. 
    # Current input `x` has static shape inside JIT?
    # Usually `x` is padded to `max_len` at dataset level? 
    # If `x` has variable `T` in JAX, we are tracing dynamic shapes.
    # However, for pure JAX model, inputs are usually fixed size batch.
    # If we are part of a `jax.jit` function, inputs must be static or we recompile.
    # If `x` comes from a dataset, it might be heavily padded already?
    # Original PyTorch dataset probably yields tensors of varying lengths if not collated?
    # Or collated to max_len?
    # preprocess.py normally takes RAW data (variable length).
    # JAX JIT *can* handle variable T if we recompile, OR we use padding + mask.
    # **Crucial**: "interpolate_or_pad" implies we output FIXED `max_len`.
    # So `x` input is variable `T`, output is `max_len`.
    
    # If we want FULL JIT compilation, we need `x` to have a max capacity padded 
    # or passed as a bounded shape.
    # But if this is run on CPU "dataloader" style, it's fine to be just JAX (no JIT) or JIT-per-shape.
    # The user asked for "pure jax code". We'll write JAX logic.
    
    # Case 1: Pad
    pad_len = max_len - T
    
    def do_pad(_):
        # Pad with zeros
        # jnp.pad needs static pad_width if we want static output shape?
        # Only if T is static.
        # If T is dynamic (Traced), we can use dynamic_slice or just concrete concatenation.
        # But wait, T is known at trace time usually if we jit a specific shape.
        
        # pad_width = ((0, pad_len), (0, 0))
        # return jnp.pad(data_flat, pad_width)
        # Using concatenate is safer for dynamic T if allow dynamic.
        padding = jnp.zeros((pad_len, data_flat.shape[1]))
        return jnp.concatenate([data_flat, padding], axis=0), jnp.concatenate([jnp.ones(T), jnp.zeros(pad_len)])

    # Case 2: Resize
    def do_resize(_):
        return resize_1d(data_flat, max_len), jnp.ones(max_len)

    # We use lax.cond. 
    # Note: T is dynamic potentially.
    # For lax.cond, both branches must return SAME shape? 
    # Yes, output shape is (max_len, C).
    
    # Problem: `jnp.concatenate` output shape depends on `pad_len` which depends on T.
    # If T is tracing variable, output shape is not static? 
    # JAX JIT output must be static.
    # So `interpolate_or_pad` can ONLY be JIT-ed if T is static (compile constant).
    # If we want a SINGLE JIT function that handles ANY T <= max_len, we need a bounded input (e.g. input is always max_len, with a true_len argument).
    
    # Assumption for this Preprocess:
    # 1. Either it runs eagerly (no JIT) -> fine.
    # 2. Or it runs JIT-ed but T is static per call -> fine.
    # 3. Or it takes a padded input + mask/len?
    
    # Original preprocess.py took raw `x`. logic: `T = data.shape[0]`.
    # We will implement it using standard JAX ops.
    # `jax.lax.cond` requires same output shape.
    # do_pad: (T+pad_len, C) = (max_len, C).
    # do_resize: (max_len, C).
    # Shapes match!
    
    # However, `jnp.zeros((pad_len, ...))` will fail in JIT if `pad_len` is dynamic.
    # If T changes, `pad_len` changes.
    # JIT requires re-compile for new T. This is standard JAX behavior.
    
    res, mask = jax.lax.cond(
        T <= max_len,
        do_pad,
        do_resize,
        operand=None
    )
    
    res = res.reshape(max_len, N, 3)
    return res, mask

class Preprocess(nnx.Module):
    # Helper module, no parameters usually.
    def __init__(self, max_len=384):
        self.max_len = max_len

    def __call__(self, x):
        # x: (T, 3*N) or (T, N, 3)
        # We handle flattening if needed.
        if x.ndim == 2:
            T, C = x.shape
            # PyTorch ds_2 logic: x.reshape(T, 3, -1).permute(0, 2, 1)
            # This implies input is Planar (x0..xN, y0..yN, z0..zN) per time step? 
            # Or just that PyTorch reshape behavior fills C first?
            # PyTorch reshape(T, 3, N) fills N dimension last.
            # So channel 0 gets first N elements.
            # JAX reshape(T, 3, N) does same.
            x = x.reshape(T, 3, -1).transpose(0, 2, 1) # (T, N, 3)
        
        # JAX arrays are immutable.
        x = normalize(x)
        x = fill_nans(x)
        x, mask = interpolate_or_pad(x, max_len=self.max_len)
        
        return x, mask

