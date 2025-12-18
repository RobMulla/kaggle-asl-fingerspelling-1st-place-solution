
import os
import sys
import torch
import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import json
from argparse import Namespace

# Add repo root to path
repo_root = "/Users/robmulla/Repos/kaggle-asl-fingerspelling-1st-place-solution"
sys.path.append(repo_root)

# Patching for compatibility (Reuse from audit_script)
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
        # Reshape to (1, 1, seq_len, dim) to match slicing in mdl_2_pt.py
        self.register_buffer("cos_cached", emb.cos().to(dtype)[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype)[None, None, :, :], persistent=False)
    def forward(self, x, seq_len=None):
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]
modeling_llama.LlamaRotaryEmbedding = MockRotary

from models import mdl_2_pt
from jax_model import model as jax_model_lib
from jax_model import config as jax_config_lib
from transformers import Speech2TextConfig

def run_verification():
    print("Initializing models...")
    
    # 1. Init PyTorch
    dummy_data_dir = "/tmp/asl_audit/"
    os.makedirs(dummy_data_dir, exist_ok=True)
    if not os.path.exists(f"{dummy_data_dir}/inference_args.json"):
        landmarks = []
        for i in range(21): landmarks.append(f"left_hand_{i}")
        for i in range(21): landmarks.append(f"right_hand_{i}")
        for i in range(33): landmarks.append(f"pose_{i}")
        for i in range(55): landmarks.append(f"face_{i}")
        columns = [f"x_{lm}" for lm in landmarks] + [f"y_{lm}" for lm in landmarks] + [f"z_{lm}" for lm in landmarks]
        with open(f"{dummy_data_dir}/inference_args.json", "w") as f:
            json.dump({"selected_columns": columns}, f)

    class PTConfig:
        def __init__(self):
            self.data_folder = dummy_data_dir
            self.max_phrase = 33
            self.n_landmarks = 130
            self.ce_ignore_index = -100
            self.label_smoothing = 0.0
            self.val_mode = 'padded'
            self.batch_size = 64
            self.max_len = 384
            self.aux_loss_weight = 0.0
            self.return_aux_logits = True
            self.bwd_loss_weight = 0.0
            self.decoder_mask_aug = 0.1
            self.encoder_config = Namespace(
                input_dim=144,
                encoder_dim=208,
                num_layers=14,
                num_attention_heads=8,
                feed_forward_expansion_factor=1,
                conv_expansion_factor=2,
                input_dropout_p=0.1,
                feed_forward_dropout_p=0.1,
                attention_dropout_p=0.1,
                conv_dropout_p=0.1,
                conv_kernel_size=51,
            )
            self.transformer_config = Speech2TextConfig(
                vocab_size=63, d_model=208, decoder_layers=2, decoder_ffn_dim=512,
                num_attention_heads=4, attention_dropout=0.2, pad_token_id=60,
                bos_token_id=61, eos_token_id=62, decoder_start_token_id=61,
                max_length=33, num_hidden_layers=2, decoder_attention_heads=4,
                max_position_embeddings=2048, scale_embedding=True, use_cache=False
            )

    pt_cfg = PTConfig()
    pt_model = mdl_2_pt.Net(pt_cfg)
    pt_model.eval()

    # 2. Init JAX
    jax_cfg = jax_config_lib.Config()
    # Override defaults to match cfg_2 (208)
    jax_cfg.decoder_config.d_model = 208
    jax_cfg.encoder_config.encoder_dim = 208 # Already default but explicit is good
    
    jax_model = jax_model_lib.Net(jax_cfg, rngs=nnx.Rngs(0))
    jax_model.eval() # Set eval mode

    print("Models initialized. Starting weight transfer...")

    # Helper to transfer
    def transfer(pt_tensor, jax_var, transpose=False, permute_conv=False):
        # pt_tensor: torch.Tensor
        # jax_var: nnx.Variable
        np_arr = pt_tensor.detach().cpu().numpy()
        if transpose:
            np_arr = np_arr.T
        if permute_conv:
            # PT: (C_out, C_in, K) -> JAX: (K, C_in, C_out)
            np_arr = np.transpose(np_arr, (2, 1, 0))
            
        if np_arr.shape != jax_var.value.shape:
            print(f"Shape mismatch! PT: {np_arr.shape}, JAX: {jax_var.value.shape}")
            return False
            
        jax_var.value = jnp.array(np_arr)
        return True

    count = 0
    
    # --- Feature Extractor ---
    # Global
    transfer(pt_model.feature_extractor.stem_linear.weight, jax_model.feature_extractor.stem_linear.kernel, transpose=True)
    transfer(pt_model.feature_extractor.stem_bn.weight, jax_model.feature_extractor.stem_bn.scale)
    transfer(pt_model.feature_extractor.stem_bn.bias, jax_model.feature_extractor.stem_bn.bias)
    # nnx.BatchNorm uses 'mean' and 'var'
    transfer(pt_model.feature_extractor.stem_bn.running_mean, jax_model.feature_extractor.stem_bn.mean)
    transfer(pt_model.feature_extractor.stem_bn.running_var, jax_model.feature_extractor.stem_bn.var)
    
    # Conv Stem 
    # PT: Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1), bias=False)
    # JAX: Conv(3, 32, (3,3), (1,2))
    # PT Weight: (32, 3, 3, 3) -> (Out, In, H, W)
    # JAX Weight: (3, 3, 3, 32) -> (H, W, In, Out)
    w = pt_model.feature_extractor.conv_stem.weight.detach().cpu().numpy()
    w = np.transpose(w, (2, 3, 1, 0))
    jax_model.feature_extractor.conv_stem.kernel.value = jnp.array(w)
    
    # BN Conv (BatchNormAct2d likely has attributes directly)
    transfer(pt_model.feature_extractor.bn_conv.weight, jax_model.feature_extractor.bn_conv.scale)
    transfer(pt_model.feature_extractor.bn_conv.bias, jax_model.feature_extractor.bn_conv.bias)
    transfer(pt_model.feature_extractor.bn_conv.running_mean, jax_model.feature_extractor.bn_conv.mean)
    transfer(pt_model.feature_extractor.bn_conv.running_var, jax_model.feature_extractor.bn_conv.var)

    # --- Feature Extractors (Hands/Face/Pose) ---
    # Assuming similar structure, loop?
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
        
        # Conv Stem
        w = pt_fe.conv_stem.weight.detach().cpu().numpy()
        w = np.transpose(w, (2, 3, 1, 0))
        jax_fe.conv_stem.kernel.value = jnp.array(w)
        
        
        # BN Conv (PT is BatchNormAct2d, JAX is nnx.BatchNorm)
        transfer(pt_fe.bn_conv.weight, jax_fe.bn_conv.scale)
        transfer(pt_fe.bn_conv.bias, jax_fe.bn_conv.bias)
        transfer(pt_fe.bn_conv.running_mean, jax_fe.bn_conv.mean)
        transfer(pt_fe.bn_conv.running_var, jax_fe.bn_conv.var)

    # --- Encoder ---
    for i in range(len(pt_model.encoder.blocks)):
        pt_b = pt_model.encoder.blocks[i]
        jax_b = jax_model.encoder.blocks[i]
        
        # MHSA
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
        # PT ConvModule: pw_conv_1, dw_conv, bn, pw_conv_2
        # PT pw_conv_1 is PointwiseConv1d -> Conv1d(k=1)
        transfer(pt_b.conv.pw_conv_1.conv.weight, jax_b.conv.pw_conv_1.conv.kernel, permute_conv=True)
        transfer(pt_b.conv.pw_conv_1.conv.bias, jax_b.conv.pw_conv_1.conv.bias)
        
        # DW Conv
        transfer(pt_b.conv.dw_conv.conv.weight, jax_b.conv.dw_conv.conv.kernel, permute_conv=True)
        # No bias in DW usually? Check code. 
        # layers.DepthwiseConv1d: bias=False default. 
        # Check pt definitions.
        
        # BN
        transfer(pt_b.conv.bn.weight, jax_b.conv.bn.scale)
        transfer(pt_b.conv.bn.bias, jax_b.conv.bn.bias)
        transfer(pt_b.conv.bn.running_mean, jax_b.conv.bn.running_mean)
        transfer(pt_b.conv.bn.running_var, jax_b.conv.bn.running_var)
        
        # PW 2
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
        
        # Scales/Biases
        transfer(pt_b.scale_mhsa, jax_b.scale_mhsa)
        transfer(pt_b.bias_mhsa, jax_b.bias_mhsa)
        transfer(pt_b.scale_ff_mhsa, jax_b.scale_ff_mhsa)
        transfer(pt_b.bias_ff_mhsa, jax_b.bias_ff_mhsa)
        transfer(pt_b.scale_conv, jax_b.scale_conv)
        transfer(pt_b.bias_conv, jax_b.bias_conv)
        transfer(pt_b.scale_ff_conv, jax_b.scale_ff_conv)
        transfer(pt_b.bias_ff_conv, jax_b.bias_ff_conv)

    # --- Decoder ---
    # PT: likelihood is pt_model.decoder.decoder (Speech2TextDecoder)
    # Check if it has 'model' or just 'layers'
    print(f"PT Decoder keys: {pt_model.decoder.decoder.__dict__.keys()}")
    # It seems to be directly 'layers' maybe? or 'model.decoder.layers'
    # Actually Speech2TextDecoder is the standalone decoder model.
    # It might conform to MBartDecoder or similar.
    # Often it is: .layers
    
    if hasattr(pt_model.decoder.decoder, 'layers'):
         pt_dec_layers = pt_model.decoder.decoder.layers
    elif hasattr(pt_model.decoder.decoder, 'model'):
         pt_dec_layers = pt_model.decoder.decoder.model.decoder.layers
    else:
         print(f"WARNING: Could not find PT Decoder layers. Keys: {pt_model.decoder.decoder.__dict__.keys()}")
         pt_dec_layers = []
         
    jax_dec_layers = jax_model.decoder.layers
    
    for i in range(len(pt_dec_layers)):
        pt_l = pt_dec_layers[i] # Speech2TextDecoderLayer
        jax_l = jax_dec_layers[i]
        
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
        
        # Encoder Attn
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
        
        # FC
        transfer(pt_l.fc1.weight, jax_l.fc1.kernel, transpose=True)
        transfer(pt_l.fc1.bias, jax_l.fc1.bias)
        transfer(pt_l.fc2.weight, jax_l.fc2.kernel, transpose=True)
        transfer(pt_l.fc2.bias, jax_l.fc2.bias)
        transfer(pt_l.final_layer_norm.weight, jax_l.final_layer_norm.scale)
        transfer(pt_l.final_layer_norm.bias, jax_l.final_layer_norm.bias)

    # Embeddings (Tokens)
    if hasattr(pt_model.decoder.decoder, 'embed_tokens'):
        transfer(pt_model.decoder.decoder.embed_tokens.weight, jax_model.decoder.embed_tokens.embedding)
    elif hasattr(pt_model.decoder.decoder, 'model'):
        transfer(pt_model.decoder.decoder.model.decoder.embed_tokens.weight, jax_model.decoder.embed_tokens.embedding)
    
    # Positions (Sinusoidal - No weights to transfer, handled by logic)
    
    # Layer Norm
    if hasattr(pt_model.decoder.decoder, 'layer_norm'):
        transfer(pt_model.decoder.decoder.layer_norm.weight, jax_model.decoder.layer_norm.scale)
        transfer(pt_model.decoder.decoder.layer_norm.bias, jax_model.decoder.layer_norm.bias)
    elif hasattr(pt_model.decoder.decoder, 'model') and hasattr(pt_model.decoder.decoder.model.decoder, 'layer_norm'):
        transfer(pt_model.decoder.decoder.model.decoder.layer_norm.weight, jax_model.decoder.layer_norm.scale)
        transfer(pt_model.decoder.decoder.model.decoder.layer_norm.bias, jax_model.decoder.layer_norm.bias)
    else:
        print("WARNING: PT Decoder has no finals layer_norm?")

    # FC Heads
    # PT: decoder.lm_head
    transfer(pt_model.decoder.lm_head.weight, jax_model.fc.kernel, transpose=True)
    # No bias in lm_head usually for Speech2Text?
    # pt_model.decoder.lm_head is nn.Linear.
    # Check if it has bias.
    if pt_model.decoder.lm_head.bias is not None:
         print("WARNING: PT LM Head has bias!")
         
    transfer(pt_model.aux_fc.weight, jax_model.aux_fc.kernel, transpose=True)
    transfer(pt_model.aux_fc.bias, jax_model.aux_fc.bias)
    
    print("Weight transfer complete.")
    
    # --- Verify ---
    print("\nRunning Verification Forward Pass...")
    
    # Create random input
    B, T, L = 1, 100, 130
    x_np = np.random.randn(B, T, L, 3).astype(np.float32)
    mask_np = np.ones((B, T)).astype(np.float32)
    token_ids_np = np.random.randint(0, 63, (B, 33)).astype(np.longlong)
    
    # PT Forward
    pt_input = {
        'input': torch.tensor(x_np),
        'input_mask': torch.tensor(mask_np),
        'token_ids': torch.tensor(token_ids_np),
        'attention_mask': torch.ones(B, 33).long(), # Dummy
        'score': torch.zeros(B), # Dummy score
        'seq_len': torch.full((B,), 33).long() # Dummy seq_len
    }
    
    # --- Verify Feature Extractor ---
    print("\nVerifying Feature Extractor...")
    with torch.no_grad():
        pt_fe_out = pt_model.feature_extractor(pt_input['input'], pt_input['input_mask'])
    
    jax_fe_out = jax_model.feature_extractor(jnp.array(x_np), jnp.array(mask_np), training=False)
    
    fe_diff = np.abs(pt_fe_out.detach().cpu().numpy() - np.array(jax_fe_out)).max()
    print(f"Feature Extractor Max Diff: {fe_diff}")
    
    if fe_diff > 1e-4:
        print("FAILURE: Feature Extractor mismatch!")
        # return # Optionally stop
    else:
        print("SUCCESS: Feature Extractor matched!")
        
    # --- Verify Encoder Block 0 ---
    print("\nVerifying Encoder Block 0...")
    
    # PT Inputs
    # We need cos, sin
    # pt_fe_out is (B, T, D) (after linear)
    # Actually wait. pt_fe_out from FeatureExtractor?
    # FeatureExtractor output: (B, T, D).
    # Net.forward logic:
    # x = feature_extractor(...)
    # x = x + feature_extractor_lhand(...) ...
    # Wait, we only verified GLOBAL feature extractor!
    # There are 5 branches!
    # If One branch is wrong, the sum is wrong.
    # But Global branch matched. Others are same class.
    # We should assume others are likely fine OR verify sum?
    # Let's verify SUM first.
    
    # Actually, let's verify Block 0 with KNOWN input (from FE output or random).
    # To isolate Block 0, we should feed SAME random input to Block 0.
    
    # Create random hidden state for Block 0 input
    hidden_dim = 208
    T = 100 # From x_np shape
    # Subsampled? FE stride (1,2) on (T, N).
    # T is preserved? No?
    # FeatureExtractor PyTorch: 
    # xc = data.permute(0,3,1,2). conv(3,3) stride(1,2).
    # Stride (1,2) -> Time stride 1. Landmarks stride 2.
    # So T is preserved.
    # So T=100.
    
    block_input_np = np.random.randn(B, T, hidden_dim).astype(np.float32)
    pt_block_input = torch.tensor(block_input_np)
    jax_block_input = jnp.array(block_input_np)
    
    # RoPE
    # pt_model has self.cos, self.sin (Parameters)
    # They are (1, 1, MaxLen, D) likely.
    # We need to slice to T.
    pt_cos = pt_model.cos[:, :, :T, :]
    pt_sin = pt_model.sin[:, :, :T, :]
    
    jax_cos, jax_sin = jax_model.rotary_emb(jax_block_input, seq_len=T)
    # Verify RoPE?
    # Checking cos diff
    rope_diff = np.abs(pt_cos.detach().cpu().numpy() - np.array(jax_cos)).max()
    print(f"RoPE Cos Diff: {rope_diff}")
    
    # Block 0
    with torch.no_grad():
        # Block signature: x, cos, sin, mask
        # Mask: pt_input['input_mask'] (B, T)
        pt_b0_out = pt_model.encoder.blocks[0](pt_block_input, pt_cos, pt_sin, pt_input['input_mask'])
        
    jax_b0_out = jax_model.encoder.blocks[0](jax_block_input, jax_cos, jax_sin, mask=jnp.array(mask_np), training=False)
    
    b0_diff = np.abs(pt_b0_out.detach().cpu().numpy() - np.array(jax_b0_out)).max()
    print(f"Block 0 Max Diff: {b0_diff}")
    
    if b0_diff > 1e-4:
        print("FAILURE: Block 0 mismatch!")
    else:
        print("SUCCESS: Block 0 matched!")
        
    # --- Verify Encoder Block 1 ---
    print("\nVerifying Encoder Block 1...")
    # Using same random input for convenience, ensuring we check weights/logic of Block 1 isolated
    with torch.no_grad():
        pt_b1_out = pt_model.encoder.blocks[1](pt_block_input, pt_cos, pt_sin, pt_input['input_mask'])
        
    jax_b1_out = jax_model.encoder.blocks[1](jax_block_input, jax_cos, jax_sin, mask=jnp.array(mask_np), training=False)
    
    b1_diff = np.abs(pt_b1_out.detach().cpu().numpy() - np.array(jax_b1_out)).max()
    print(f"Block 1 Max Diff: {b1_diff}")
    
    if b1_diff > 1e-4:
        print("FAILURE: Block 1 mismatch!")
    else:
        print("SUCCESS: Block 1 matched!")

    # --- Verify Encoder Block 13 (Last Block) ---
    print("\nVerifying Encoder Block 13...")
    with torch.no_grad():
        pt_b13_out = pt_model.encoder.blocks[13](pt_block_input, pt_cos, pt_sin, pt_input['input_mask'])
        
    jax_b13_out = jax_model.encoder.blocks[13](jax_block_input, jax_cos, jax_sin, mask=jnp.array(mask_np), training=False)
    
    b13_diff = np.abs(pt_b13_out.detach().cpu().numpy() - np.array(jax_b13_out)).max()
    print(f"Block 13 Max Diff: {b13_diff}")
    
    if b13_diff > 1e-4:
        print("FAILURE: Block 13 mismatch!")
    else:
        print("SUCCESS: Block 13 matched!")
        
    # --- Verify AuxFC ---
    print("\nVerifying AuxFC...")
    # Input (B, D)
    aux_in_np = np.random.randn(B, hidden_dim).astype(np.float32)
    with torch.no_grad():
        pt_aux = pt_model.aux_fc(torch.tensor(aux_in_np))
        
    jax_aux = jax_model.aux_fc(jnp.array(aux_in_np))
    
    aux_diff = np.abs(pt_aux.detach().cpu().numpy() - np.array(jax_aux)).max()
    print(f"AuxFC Max Diff: {aux_diff}")
    
    if aux_diff > 1e-4:
        print("FAILURE: AuxFC mismatch!")
    else:
        print("SUCCESS: AuxFC matched!")

    with torch.no_grad():
        pt_out = pt_model(pt_input, debug=False)
        # pt_out is dict with keys: 'loss', 'generated_ids', 'aux_logits'
        
    # JAX Forward
    # Need to verify call signature: x, mask, token_ids
    # x: (B, T, N, 3)
    logits, aux_logits = jax_model(
        jnp.array(x_np), 
        mask=jnp.array(mask_np), 
        token_ids=jnp.array(token_ids_np),
        training=False
    )
    
    # Compare Aux Logits
    pt_aux = pt_out['aux_logits'].detach().cpu().numpy()
    jax_aux = np.array(aux_logits)
    diff = np.abs(pt_aux - jax_aux).max()
    print(f"Aux Logits Max Diff: {diff}")
    
    if diff < 1e-4:
        print("SUCCESS: Aux Logits matched!")
    else:
        print("FAILURE: Aux Logits mismatch.")

    # Compare Logits (Harder because PT returns loss or generated ids, but we can verify intermediate or hack PT output)
    # pt_out might not contain raw logits if not modified.
    # mdl_2_pt.py returns output dict.
    # We can inspect pt_model to return logits if needed.
    
    # But wait, audit_script checks parity.
    # For now, let's just create this script and try running to see if aux logits match.
    # That proves Encoder parity at least.

if __name__ == "__main__":
    run_verification()
