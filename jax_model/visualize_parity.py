
import os
import sys
import torch
import math
import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import json
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not found, plotting will be skipped.")

# Add repo root to path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_root)
sys.path.append(os.path.join(repo_root, 'configs'))
sys.path.append(os.path.join(repo_root, 'data'))

# --- Patches ---
# Patch augmentations
try:
    import jax_model.compat_augmentations as compat_A
    sys.modules['augmentations'] = compat_A
    sys.modules['configs.augmentations'] = compat_A
except ImportError:
    print("Warning: compat_augmentations not found, skipping patch.")

# Patch LlamaRotaryEmbedding for CPU/Version compatibility
from transformers.models.llama import modeling_llama
class MockRotary(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=device, dtype=torch.get_default_dtype())
        
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype)[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype)[None, None, :, :], persistent=False)
        
    def forward(self, x, seq_len=None):
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]
modeling_llama.LlamaRotaryEmbedding = MockRotary

# --- Imports after path setup ---
from models import mdl_2_pt
from jax_model import model as jax_model_lib
from jax_model import config as jax_config_lib
from configs.cfg_2 import cfg as cfg_2
from data.ds_2 import Preprocessing, interpolate_or_pad
from jax_model.verify_parity import transfer_weights, load_data

def decode_text(token_ids, c2n):
    n2c = {v: k for k, v in c2n.items()}
    return "".join([n2c.get(t, '') for t in token_ids if t in n2c])

def jax_generate(model, x, encoder_mask, max_new_tokens=33, start_token_id=61, end_token_id=62):
    # Feature Extraction
    mask = encoder_mask
    if mask is None: mask = jnp.ones((x.shape[0], x.shape[1]), dtype=bool)
    
    # Copied from Net.__call__ for extraction
    def encode(x, mask, training=False):
        if mask is None: mask = jnp.ones((x.shape[0], x.shape[1]), dtype=bool)
        
        ranges = [(0, 21), (21, 42), (42, 54), (54, 130)]
        x_parts = []
        dropped_mask = (x[..., :2].sum(-1) == 0) 
        
        for start_idx, end_idx in ranges:
            part = x[:, :, start_idx:end_idx, :]
            mu = part.mean(axis=2, keepdims=True)
            std = part.std(axis=2, keepdims=True)
            part_norm = (part - mu) / std
            part_norm = jnp.nan_to_num(part_norm, nan=0.0, posinf=0.0, neginf=0.0)
            m_part = dropped_mask[:, :, start_idx:end_idx]
            part_norm = part_norm * (1.0 - m_part[..., None].astype(part_norm.dtype))
            x_parts.append(part_norm)
        
        x_lhand = model.feature_extractor_lhand(x_parts[0], mask, training)
        x_rhand = model.feature_extractor_rhand(x_parts[1], mask, training)
        x_pose = model.feature_extractor_pose(x_parts[2], mask, training)
        x_face = model.feature_extractor_face(x_parts[3], mask, training)
        x_global = model.feature_extractor(x, mask, training)
        x1 = jnp.concatenate([x_lhand, x_rhand, x_face, x_pose], axis=-1)
        x_combined = x_global + x1
        
        cos, sin = model.rotary_emb(x_combined, seq_len=x_combined.shape[1])
        enc_out = model.encoder(x_combined, cos, sin, mask=mask, training=training)
        return enc_out

    enc_out = encode(x, encoder_mask, training=False)
    
    # Decode Loop
    B = x.shape[0]
    decoder_input_ids = jnp.full((B, 1), start_token_id, dtype=jnp.int32)
    
    for i in range(max_new_tokens - 1):
        dec_out = model.decoder(decoder_input_ids, enc_out, encoder_attention_mask=mask, training=False)
        logits = model.fc(dec_out)
        next_token = jnp.argmax(logits[:, -1, :], axis=-1)[:, None]
        decoder_input_ids = jnp.concatenate([decoder_input_ids, next_token], axis=1)
        if (next_token == end_token_id).all():
            break
            
    return decoder_input_ids

def run_visual_comparison():
    # 1. Load Data
    batch = load_data()
    
    # 2. Load PyTorch
    print("Initializing PyTorch Model...")
    pt_model = mdl_2_pt.Net(cfg_2)
    ckpt_path = "datamount/weights/checkpoint_step_12800_seed444740.pth"
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt['model'] if 'model' in ckpt else ckpt
    state = {k.replace('module.', ''): v for k, v in state.items()}
    pt_model.load_state_dict(state, strict=False)
    pt_model.eval()
    
    # 3. Load JAX
    print("Initializing JAX Model...")
    jax_cfg = jax_config_lib.Config()
    jax_cfg.encoder_config.encoder_dim = cfg_2.encoder_config.encoder_dim
    jax_cfg.encoder_config.num_layers = cfg_2.encoder_config.num_layers
    jax_cfg.encoder_config.num_attention_heads = cfg_2.encoder_config.num_attention_heads
    jax_cfg.encoder_config.conv_kernel_size = cfg_2.encoder_config.conv_kernel_size
    jax_cfg.decoder_config.vocab_size = cfg_2.transformer_config.vocab_size
    jax_cfg.decoder_config.d_model = cfg_2.transformer_config.d_model
    jax_cfg.decoder_config.decoder_layers = cfg_2.transformer_config.decoder_layers
    jax_cfg.decoder_config.decoder_ffn_dim = cfg_2.transformer_config.decoder_ffn_dim
    jax_cfg.decoder_config.num_heads = cfg_2.transformer_config.num_attention_heads
    jax_cfg.decoder_config.pad_token_id = pt_model.decoder.decoder_pad_token_id
    if hasattr(cfg_2.transformer_config, 'max_target_positions'):
        jax_cfg.decoder_config.max_length = cfg_2.transformer_config.max_target_positions
    elif hasattr(cfg_2.transformer_config, 'max_position_embeddings'):
        jax_cfg.decoder_config.max_length = cfg_2.transformer_config.max_position_embeddings
    
    jax_model = jax_model_lib.Net(jax_cfg, rngs=nnx.Rngs(0))
    jax_model.eval()
    transfer_weights(pt_model, jax_model)
    
    # 4. Generate Text
    print("Generating Text...")
    with torch.no_grad():
        # Get encoder outputs using debug=True
        encoder_hidden_states = pt_model(batch, debug=True)
        # Generate using encoder outputs
        pt_gen_ids = pt_model.decoder.generate(encoder_hidden_states, max_new_tokens=33, encoder_attention_mask=batch['input_mask'].long())
        
    jax_x = jnp.array(batch['input'].numpy())
    jax_mask = jnp.array(batch['input_mask'].numpy())
    jax_gen_ids = jax_generate(jax_model, jax_x, jax_mask, max_new_tokens=33, 
                               start_token_id=pt_model.decoder.decoder_start_token_id, 
                               end_token_id=pt_model.decoder.decoder_end_token_id)
    
    # Decode
    with open('datamount/character_to_prediction_index.json') as f:
        c2n = json.load(f)
    n = len(c2n)
    c2n['P'] = n; c2n['S'] = n+1; c2n['E'] = n+2
    
    pt_text = decode_text(pt_gen_ids[0].tolist(), c2n)
    jax_text = decode_text(np.array(jax_gen_ids[0]).tolist(), c2n)
    
    print(f"PyTorch Predicted: {pt_text}")
    print(f"JAX Predicted:     {jax_text}")
    
    # 5. Teacher Forced Heatmaps
    print("Generating Heatmaps (Teacher Forced)...")
    
    # Capture Logits
    pt_logits = None
    def hook_fn(m, i, o):
        nonlocal pt_logits
        pt_logits = o.detach().cpu().numpy()
    h = pt_model.decoder.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        pt_out = pt_model(batch)
    h.remove()
    
    # JAX inputs for teacher forcing
    jax_ids = jnp.array(batch['token_ids'].numpy())
    start_id = pt_model.decoder.decoder_start_token_id
    jax_ids_shifted = jnp.zeros_like(jax_ids)
    jax_ids_shifted = jax_ids_shifted.at[:, 1:].set(jax_ids[:, :-1])
    jax_ids_shifted = jax_ids_shifted.at[:, 0].set(start_id)
    
    jax_logits_out, _ = jax_model(jax_x, mask=jax_mask, token_ids=jax_ids_shifted, training=False)
    
    # Plotting
    pt_log_map = pt_logits[0]
    jax_log_map = np.array(jax_logits_out[0])
    
    diff_map = np.abs(pt_log_map - jax_log_map)
    max_diff = diff_map.max()
    print(f"Max Diff Logits: {max_diff:.6f}")
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))
    
    def plot_heatmap(ax, data, title, cmap):
        im = ax.imshow(data.T, aspect='auto', cmap=cmap, origin='lower')
        ax.set_title(title)
        ax.set_ylabel("Vocab Index")
        ax.set_xlabel("Time Step")
        fig.colorbar(im, ax=ax)

    plot_heatmap(axes[0], pt_log_map, f"PyTorch Logits\nPred: {pt_text}", "viridis")
    plot_heatmap(axes[1], jax_log_map, f"JAX Logits\nPred: {jax_text}", "viridis")
    plot_heatmap(axes[2], diff_map, f"Difference (Max Diff: {max_diff:.6f})", "magma")
    
    plt.tight_layout()
    plt.savefig("parity_comparison.png")
    print("Saved parity_comparison.png")

if __name__ == "__main__":
    run_visual_comparison()
