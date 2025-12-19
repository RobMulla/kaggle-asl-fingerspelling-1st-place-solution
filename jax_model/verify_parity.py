
import os
import sys
import torch
import math
import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import json
from argparse import Namespace
import importlib

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

def transfer_weights(pt_model, jax_model):
    print("Transferring weights...")
    
    def transfer(pt_tensor, jax_var, transpose=False, permute_conv=False):
        if pt_tensor is None: return False
        np_arr = pt_tensor.detach().cpu().numpy()
        if transpose: np_arr = np_arr.T
        if permute_conv: np_arr = np.transpose(np_arr, (2, 1, 0)) # Conv1d (Out, In, K) -> (K, In, Out)
        
        if np_arr.shape != jax_var.value.shape:
            # Try handling reshape for specific mismatch if obvious
            if np_arr.size == jax_var.value.size:
                np_arr = np_arr.reshape(jax_var.value.shape)
            else:
                print(f"Shape mismatch! PT: {np_arr.shape}, JAX: {jax_var.value.shape}")
                return False
                
        jax_var.value = jnp.array(np_arr)
        return True

    # --- Feature Extractor ---
    # Global
    transfer(pt_model.feature_extractor.stem_linear.weight, jax_model.feature_extractor.stem_linear.kernel, transpose=True)
    transfer(pt_model.feature_extractor.stem_bn.weight, jax_model.feature_extractor.stem_bn.scale)
    transfer(pt_model.feature_extractor.stem_bn.bias, jax_model.feature_extractor.stem_bn.bias)
    transfer(pt_model.feature_extractor.stem_bn.running_mean, jax_model.feature_extractor.stem_bn.mean)
    transfer(pt_model.feature_extractor.stem_bn.running_var, jax_model.feature_extractor.stem_bn.var)
    
    w = pt_model.feature_extractor.conv_stem.weight.detach().cpu().numpy() # (Out, In, H, W) -> (32, 3, 3, 3) 
    # JAX Conv: (H, W, In, Out) -> (3, 3, 3, 32)
    w = np.transpose(w, (2, 3, 1, 0))
    jax_model.feature_extractor.conv_stem.kernel.value = jnp.array(w)
    
    transfer(pt_model.feature_extractor.bn_conv.weight, jax_model.feature_extractor.bn_conv.scale)
    transfer(pt_model.feature_extractor.bn_conv.bias, jax_model.feature_extractor.bn_conv.bias)
    transfer(pt_model.feature_extractor.bn_conv.running_mean, jax_model.feature_extractor.bn_conv.mean)
    transfer(pt_model.feature_extractor.bn_conv.running_var, jax_model.feature_extractor.bn_conv.var)

    # Branches
    parts = [
        (pt_model.feature_extractor_lhand, jax_model.feature_extractor_lhand),
        (pt_model.feature_extractor_rhand, jax_model.feature_extractor_rhand),
        (pt_model.feature_extractor_face, jax_model.feature_extractor_face),
        (pt_model.feature_extractor_pose, jax_model.feature_extractor_pose),
    ]
    for pt_fe, jax_fe in parts:
        transfer(pt_fe.stem_linear.weight, jax_fe.stem_linear.kernel, transpose=True)
        transfer(pt_fe.stem_bn.weight, jax_fe.stem_bn.scale)
        transfer(pt_fe.stem_bn.bias, jax_fe.stem_bn.bias)
        transfer(pt_fe.stem_bn.running_mean, jax_fe.stem_bn.mean)
        transfer(pt_fe.stem_bn.running_var, jax_fe.stem_bn.var)
        
        w = pt_fe.conv_stem.weight.detach().cpu().numpy()
        w = np.transpose(w, (2, 3, 1, 0))
        jax_fe.conv_stem.kernel.value = jnp.array(w)
        
        transfer(pt_fe.bn_conv.weight, jax_fe.bn_conv.scale)
        transfer(pt_fe.bn_conv.bias, jax_fe.bn_conv.bias)
        transfer(pt_fe.bn_conv.running_mean, jax_fe.bn_conv.mean)
        transfer(pt_fe.bn_conv.running_var, jax_fe.bn_conv.var)

    # --- Encoder ---
    for i in range(len(pt_model.encoder.blocks)):
        pt_b = pt_model.encoder.blocks[i]
        jax_b = jax_model.encoder.blocks[i]
        
        # RoPE MHSA
        transfer(pt_b.mhsa_llama.q_proj.weight, jax_b.mhsa_llama.q_proj.kernel, transpose=True)
        transfer(pt_b.mhsa_llama.k_proj.weight, jax_b.mhsa_llama.k_proj.kernel, transpose=True)
        transfer(pt_b.mhsa_llama.v_proj.weight, jax_b.mhsa_llama.v_proj.kernel, transpose=True)
        transfer(pt_b.mhsa_llama.o_proj.weight, jax_b.mhsa_llama.o_proj.kernel, transpose=True)
        transfer(pt_b.ln_mhsa.weight, jax_b.ln_mhsa.scale)
        transfer(pt_b.ln_mhsa.bias, jax_b.ln_mhsa.bias)
        
        # FF MHSA
        transfer(pt_b.ff_mhsa.ffn1.weight, jax_b.ff_mhsa.ffn1.kernel, transpose=True)
        transfer(pt_b.ff_mhsa.ffn1.bias, jax_b.ff_mhsa.ffn1.bias)
        transfer(pt_b.ff_mhsa.ffn2.weight, jax_b.ff_mhsa.ffn2.kernel, transpose=True)
        transfer(pt_b.ff_mhsa.ffn2.bias, jax_b.ff_mhsa.ffn2.bias)
        transfer(pt_b.ln_ff_mhsa.weight, jax_b.ln_ff_mhsa.scale)
        transfer(pt_b.ln_ff_mhsa.bias, jax_b.ln_ff_mhsa.bias)
        
        # Conv
        transfer(pt_b.conv.pw_conv_1.conv.weight, jax_b.conv.pw_conv_1.conv.kernel, permute_conv=True)
        transfer(pt_b.conv.pw_conv_1.conv.bias, jax_b.conv.pw_conv_1.conv.bias)
        transfer(pt_b.conv.dw_conv.conv.weight, jax_b.conv.dw_conv.conv.kernel, permute_conv=True)
        transfer(pt_b.conv.bn.weight, jax_b.conv.bn.scale)
        transfer(pt_b.conv.bn.bias, jax_b.conv.bn.bias)
        transfer(pt_b.conv.bn.running_mean, jax_b.conv.bn.running_mean)
        transfer(pt_b.conv.bn.running_var, jax_b.conv.bn.running_var)
        transfer(pt_b.conv.pw_conv_2.conv.weight, jax_b.conv.pw_conv_2.conv.kernel, permute_conv=True)
        transfer(pt_b.conv.pw_conv_2.conv.bias, jax_b.conv.pw_conv_2.conv.bias)
        transfer(pt_b.ln_conv.weight, jax_b.ln_conv.scale)
        transfer(pt_b.ln_conv.bias, jax_b.ln_conv.bias)
        
        # FF Conv
        transfer(pt_b.ff_conv.ffn1.weight, jax_b.ff_conv.ffn1.kernel, transpose=True)
        transfer(pt_b.ff_conv.ffn1.bias, jax_b.ff_conv.ffn1.bias)
        transfer(pt_b.ff_conv.ffn2.weight, jax_b.ff_conv.ffn2.kernel, transpose=True)
        transfer(pt_b.ff_conv.ffn2.bias, jax_b.ff_conv.ffn2.bias)
        transfer(pt_b.ln_ff_conv.weight, jax_b.ln_ff_conv.scale)
        transfer(pt_b.ln_ff_conv.bias, jax_b.ln_ff_conv.bias)
        
        # Scales
        transfer(pt_b.scale_mhsa, jax_b.scale_mhsa)
        transfer(pt_b.bias_mhsa, jax_b.bias_mhsa)
        transfer(pt_b.scale_ff_mhsa, jax_b.scale_ff_mhsa)
        transfer(pt_b.bias_ff_mhsa, jax_b.bias_ff_mhsa)
        transfer(pt_b.scale_conv, jax_b.scale_conv)
        transfer(pt_b.bias_conv, jax_b.bias_conv)
        transfer(pt_b.scale_ff_conv, jax_b.scale_ff_conv)
        transfer(pt_b.bias_ff_conv, jax_b.bias_ff_conv)

    # --- Decoder ---
    # Access layers
    if hasattr(pt_model.decoder.decoder, 'model'):
        pt_dec_layers = pt_model.decoder.decoder.model.decoder.layers
        pt_embed = pt_model.decoder.decoder.model.decoder.embed_tokens
        pt_ln = pt_model.decoder.decoder.model.decoder.layer_norm
    else:
        pt_dec_layers = pt_model.decoder.decoder.layers
        pt_embed = pt_model.decoder.decoder.embed_tokens
        pt_ln = pt_model.decoder.decoder.layer_norm

    for i in range(len(pt_dec_layers)):
        pt_l = pt_dec_layers[i]
        jax_l = jax_model.decoder.layers[i]
        
        # Self Attn
        transfer(pt_l.self_attn.k_proj.weight, jax_l.self_attn.k_proj.kernel, transpose=True)
        transfer(pt_l.self_attn.k_proj.bias, jax_l.self_attn.k_proj.bias)
        transfer(pt_l.self_attn.v_proj.weight, jax_l.self_attn.v_proj.kernel, transpose=True)
        transfer(pt_l.self_attn.v_proj.bias, jax_l.self_attn.v_proj.bias)
        transfer(pt_l.self_attn.q_proj.weight, jax_l.self_attn.q_proj.kernel, transpose=True)
        transfer(pt_l.self_attn.q_proj.bias, jax_l.self_attn.q_proj.bias)
        transfer(pt_l.self_attn.out_proj.weight, jax_l.self_attn.o_proj.kernel, transpose=True)
        transfer(pt_l.self_attn.out_proj.bias, jax_l.self_attn.o_proj.bias)
        transfer(pt_l.self_attn_layer_norm.weight, jax_l.self_attn_layer_norm.scale)
        transfer(pt_l.self_attn_layer_norm.bias, jax_l.self_attn_layer_norm.bias)
        
        # Cross Attn
        transfer(pt_l.encoder_attn.k_proj.weight, jax_l.encoder_attn.k_proj.kernel, transpose=True)
        transfer(pt_l.encoder_attn.k_proj.bias, jax_l.encoder_attn.k_proj.bias)
        transfer(pt_l.encoder_attn.v_proj.weight, jax_l.encoder_attn.v_proj.kernel, transpose=True)
        transfer(pt_l.encoder_attn.v_proj.bias, jax_l.encoder_attn.v_proj.bias)
        transfer(pt_l.encoder_attn.q_proj.weight, jax_l.encoder_attn.q_proj.kernel, transpose=True)
        transfer(pt_l.encoder_attn.q_proj.bias, jax_l.encoder_attn.q_proj.bias)
        transfer(pt_l.encoder_attn.out_proj.weight, jax_l.encoder_attn.o_proj.kernel, transpose=True)
        transfer(pt_l.encoder_attn.out_proj.bias, jax_l.encoder_attn.o_proj.bias)
        transfer(pt_l.encoder_attn_layer_norm.weight, jax_l.encoder_attn_layer_norm.scale)
        transfer(pt_l.encoder_attn_layer_norm.bias, jax_l.encoder_attn_layer_norm.bias)
        
        # FF
        transfer(pt_l.fc1.weight, jax_l.fc1.kernel, transpose=True)
        transfer(pt_l.fc1.bias, jax_l.fc1.bias)
        transfer(pt_l.fc2.weight, jax_l.fc2.kernel, transpose=True)
        transfer(pt_l.fc2.bias, jax_l.fc2.bias)
        transfer(pt_l.final_layer_norm.weight, jax_l.final_layer_norm.scale)
        transfer(pt_l.final_layer_norm.bias, jax_l.final_layer_norm.bias)

    # Embeddings
    transfer(pt_embed.weight, jax_model.decoder.embed_tokens.embedding)
    transfer(pt_ln.weight, jax_model.decoder.layer_norm.scale)
    transfer(pt_ln.bias, jax_model.decoder.layer_norm.bias)
    
    # Heads
    transfer(pt_model.decoder.lm_head.weight, jax_model.fc.kernel, transpose=True)
    transfer(pt_model.decoder2.lm_head.weight, jax_model.fc_bwd.kernel, transpose=True)
    transfer(pt_model.aux_fc.weight, jax_model.aux_fc.kernel, transpose=True)
    transfer(pt_model.aux_fc.bias, jax_model.aux_fc.bias)

def load_data():
    print("Loading data...")
    npy_path = "datamount/train_landmarks_npy/542995127.npy"
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"{npy_path} not found")
        
    data = np.load(npy_path)
    print(f"Loaded data shape: {data.shape}")
    
    # Setup args for Preprocessing
    # We need to write a temp inference_args.json for manual usage if we rely on it,
    # OR we can just mock what we need. 
    # ds_2 uses inference_args.json to filter columns.
    temp_dir = "temp_data_verify"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Logic to fix interleaved columns for ds_2
    orig_args = 'datamount/train_landmarks_npy/inference_args.json'
    with open(orig_args) as f:
        orig_cols = json.load(f)['selected_columns']
        
    print(f"Loaded orig_cols: {len(orig_cols)} items")
    print(f"First 10: {orig_cols[:10]}")

    cols_by_type = {'left_hand': [], 'right_hand': [], 'pose': [], 'face': []}
    for c in orig_cols:
        for t in cols_by_type:
            if t in c:
                cols_by_type[t].append(c)
                break
                
    # Fix counts (Face 76, Pose 12)
    # Check what we have first
    for t in cols_by_type:
        print(f"Original {t}: {len(cols_by_type[t])} columns")

    target_pose_count = 36 # 12 * 3
    excess_pose = cols_by_type['pose'][target_pose_count:]
    cols_by_type['pose'] = cols_by_type['pose'][:target_pose_count]
    cols_by_type['face'].extend([c.replace('pose', 'face_remapped') for c in excess_pose])
    
    print(f"After Remap - Pose: {len(cols_by_type['pose'])}, Face: {len(cols_by_type['face'])}")

    # Blocked ordering: ALL X, then ALL Y, then ALL Z
    final_xs = []
    final_ys = []
    final_zs = []
    
    for t in ['left_hand', 'right_hand', 'pose', 'face']:
         c_list = cols_by_type[t]
         xs = [c for c in c_list if 'x_' in c]
         ys = [c for c in c_list if 'y_' in c]
         zs = [c for c in c_list if 'z_' in c]
         print(f"Type {t}: X={len(xs)}, Y={len(ys)}, Z={len(zs)}")
         final_xs.extend(xs)
         final_ys.extend(ys)
         final_zs.extend(zs)
         
    all_cols = final_xs + final_ys + final_zs
         
    print(f"Total Columns: {len(all_cols)}")
    with open(f"{temp_dir}/inference_args.json", "w") as f:
        json.dump({"selected_columns": all_cols}, f)
        
    cfg_2.data_folder = f"{temp_dir}/"
    
    # process
    processor = Preprocessing()
    data_t = torch.from_numpy(data)
    data_t = processor(data_t)
    data_t, mask = interpolate_or_pad(data_t, max_len=384)
    
    # Tokenize dummy phrase
    # +63-3001-130-160-0570
    phrase = "+63-3001-130-160-0570"
    with open('datamount/character_to_prediction_index.json') as f:
        c2n = json.load(f)
    n = len(c2n)
    c2n['P'] = n; c2n['S'] = n+1; c2n['E'] = n+2
    
    ids = [c2n[c] for c in phrase] + [c2n['E']]
    max_phrase = 33
    if len(ids) > max_phrase: ids = ids[:max_phrase]
    pads = [c2n['P']] * (max_phrase - len(ids))
    ids += pads
    mask_ids = [1]*(len(ids)-len(pads)) + [0]*len(pads)
    
    return {
        'input': data_t.unsqueeze(0),
        'input_mask': mask.unsqueeze(0),
        'token_ids': torch.tensor(ids).long().unsqueeze(0),
        'attention_mask': torch.tensor(mask_ids).long().unsqueeze(0),
        'score': torch.tensor([1.0]),
        'seq_len': torch.tensor([len(data)])
    }

def verify():
    # 1. Load Data
    batch = load_data()
    
    # 2. PyTorch Model
    print("Initializing PyTorch Model...")
    pt_model = mdl_2_pt.Net(cfg_2)
    ckpt_path = "datamount/weights/checkpoint_step_12800_seed444740.pth"
    print(f"Loading weights from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt['model'] if 'model' in ckpt else ckpt
    state = {k.replace('module.', ''): v for k, v in state.items()}
    pt_model.load_state_dict(state, strict=False)
    pt_model.eval()
    
    # 3. JAX Model
    print("Initializing JAX Model...")
    jax_cfg = jax_config_lib.Config()
    # Sync params
    jax_cfg.encoder_config.encoder_dim = cfg_2.encoder_config.encoder_dim
    jax_cfg.encoder_config.num_layers = cfg_2.encoder_config.num_layers
    jax_cfg.encoder_config.num_attention_heads = cfg_2.encoder_config.num_attention_heads
    jax_cfg.encoder_config.conv_kernel_size = cfg_2.encoder_config.conv_kernel_size
    
    jax_cfg.decoder_config.vocab_size = cfg_2.transformer_config.vocab_size
    jax_cfg.decoder_config.d_model = cfg_2.transformer_config.d_model
    jax_cfg.decoder_config.decoder_layers = cfg_2.transformer_config.decoder_layers
    jax_cfg.decoder_config.decoder_ffn_dim = cfg_2.transformer_config.decoder_ffn_dim
    jax_cfg.decoder_config.num_heads = cfg_2.transformer_config.num_attention_heads
    
    # Pad Token ID
    jax_cfg.decoder_config.pad_token_id = pt_model.decoder.decoder_pad_token_id
    if hasattr(cfg_2.transformer_config, 'max_target_positions'):
        jax_cfg.decoder_config.max_length = cfg_2.transformer_config.max_target_positions
    elif hasattr(cfg_2.transformer_config, 'max_position_embeddings'):
        jax_cfg.decoder_config.max_length = cfg_2.transformer_config.max_position_embeddings
    
    jax_model = jax_model_lib.Net(jax_cfg, rngs=nnx.Rngs(0))
    jax_model.eval()
    
    # 4. Transfer
    transfer_weights(pt_model, jax_model)
    
    # 5. Run & Compare
    print("Running Inference...")
    
    # Capture Logits
    pt_logits = None
    def hook_fn(m, i, o):
        nonlocal pt_logits
        pt_logits = o.detach().cpu().numpy()
    h = pt_model.decoder.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        pt_out = pt_model(batch)
    h.remove()
    
    # JAX inputs
    jax_x = jnp.array(batch['input'].numpy())
    jax_mask = jnp.array(batch['input_mask'].numpy())
    jax_ids = jnp.array(batch['token_ids'].numpy())
    
    # Shift logic for decoder input
    start_id = pt_model.decoder.decoder_start_token_id
    jax_ids_shifted = jnp.zeros_like(jax_ids)
    jax_ids_shifted = jax_ids_shifted.at[:, 1:].set(jax_ids[:, :-1])
    jax_ids_shifted = jax_ids_shifted.at[:, 0].set(start_id)
    
    # Valid Mask
    jax_logits_out, jax_aux_out = jax_model(jax_x, mask=jax_mask, token_ids=jax_ids_shifted, training=False)
    
    # Mask out padding for comparison
    # jax_ids_shifted has the input ids to decoder.
    # We should use the target mask or valid length
    # batch['attention_mask'] (1 for valid, 0 for pad)
    # But batch['attention_mask'] length is 33? (Token IDs length).
    # Logits shape: (1, 33, 63).
    # We need to mask (1, 33).
    
    valid_mask = batch['attention_mask'].numpy().astype(bool) # (1, 33)
    
    # Aux
    pt_aux = pt_out['aux_logits'].detach().cpu().numpy()
    # Mask aux? Aux is (B, 1)? No, Aux is (B, T, 1)?
    # mdl_2_pt.Net: aux_logits = self.aux_fc(x) -> (B, T, 1)?
    # Wait, encoder output (B, T_enc, Dim).
    # Aux FC is on encoder output. Mask is from input_mask (Landmarks).
    enc_mask = batch['input_mask'].numpy().astype(bool) # (1, 384)
    if pt_aux.shape[1] == enc_mask.shape[1]:
         aux_diff = np.abs(pt_aux[enc_mask] - np.array(jax_aux_out)[enc_mask]).max()
    else:
         aux_diff = np.abs(pt_aux - np.array(jax_aux_out)).max()
         
    print(f"Aux Logits Max Diff (Masked): {aux_diff:.6f}")
    
    # Logits
    # pt_logits collected via hook is (B, T_dec, Vocab)
    if pt_logits is not None:
        logits_diff = np.abs(pt_logits[valid_mask] - np.array(jax_logits_out)[valid_mask]).max()
    else:
        logits_diff = 999.0
        
    print(f"Logits Max Diff (Masked): {logits_diff:.6f}")
    
    if logits_diff < 1e-3:
        print("VERIFICATION SUCCESSFUL")
    else:
        print("VERIFICATION FAILED (High Diff)")

if __name__ == "__main__":
    verify()
