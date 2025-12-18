
import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import os
import sys

# Add path
# sys.path.append(os.path.dirname(__file__))

from .model import Net, FeatureExtractor, SqueezeformerEncoder, Decoder
from .config import Config, EncoderConfig, DecoderConfig
from .layers import LlamaRotaryEmbedding

def load_dump(name):
    path = f'dump_checks/{name}.npy'
    if not os.path.exists(path):
        print(f"Skipping {name}, not found.")
        return None
    return np.load(path)

def verify():
    print("Verifying Model 2 JAX Port...")
    
    # Load inputs
    x_np = load_dump('input')
    input_mask_np = load_dump('input_mask')
    token_ids_np = load_dump('token_ids')
    
    if x_np is None:
        print("Input dump not found! Run dump_mdl2_intermediates.py first.")
        return

    # Initialize JAX model
    cfg = Config()
    # Ensure config matches Model 2
    cfg.encoder_config.encoder_dim = 208
    cfg.encoder_config.input_dim = 208
    cfg.encoder_config.num_layers = 14
    cfg.encoder_config.num_attention_heads = 8
    cfg.encoder_config.conv_kernel_size = 51
    
    # Decoder Config alignment
    cfg.decoder_config.d_model = 208
    cfg.decoder_config.num_heads = 4
    # Speech2TextConfig "s2t-small" has num_heads=4.
    # d_model=256 default. 
    # If we set d_model=208, 208/4 = 52.
    
    # We must ensure defaults in verify match cfg_2.
    # cfg_2: dim=208, num_attention_heads=4 (for encoder? No encoder has 8? Wait)
    # cfg_2.py: encoder_config.num_attention_heads = 4.
    # In verify.py I set it to 8 above?
    # Let me check cfg_2.py again.
    # cfg_2.py line 108: encoder_config.num_attention_heads= 4.
    # So I should set encoder heads to 4 too if I want to match exactly.
    # But verify.py had 8.
    
    # Aligning with cfg_2 exactly:
    cfg.encoder_config.num_attention_heads = 4
    cfg.decoder_config.num_heads = 4
    # If JAX config defaults are updated, we assume they match.
    
    rngs = nnx.Rngs(0)
    # print(f"DEBUG: cfg.encoder_config.encoder_dim = {cfg.encoder_config.encoder_dim}")
    # print(f"DEBUG: cfg.decoder_config.d_model = {cfg.decoder_config.d_model}") 
    
    model = Net(cfg, rngs=rngs)
    
    # print(f"DEBUG: type(model.decoder.layers) = {type(model.decoder.layers)}")
    if hasattr(model.decoder, 'layers'):
        pass
        # print(f"DEBUG: model.decoder.layers = {model.decoder.layers}")
    
    # We strictly need to verify layer by layer.
    # But initializing weights to match PyTorch is hard without a converter.
    # We can only verify SHAPES and Graph connectivity, OR we modify verification to use dummy weights or zero weights?
    # Actually, the user asked for "Verification test harness (JAX)". 
    # Usually this implies checking if we can Load weights or if forward pass runs.
    # To check numerical equality, we need to load weights.
    # I haven't written a weight converter. 
    # The Implementation Plan says: "Verify Feature Extractors (expect np.allclose)".
    # This implies we should be able to match outputs given SAME inputs + weights.
    # Since we can't easily load PyTorch weights yet (naming mismatch), 
    # checking shapes is the first step. 
    # Or we can verify logical components by mocking inputs to them and asserting non-nan outputs.
    
    # Let's verify Forward Pass Shapes first.
    print("Running Forward Pass (JAX)...")
    
    import inspect
    print(f"DEBUG: Net.__call__ signature: {inspect.signature(model.__call__)}")
    
    x_jax = jnp.array(x_np)
    mask_jax = jnp.array(input_mask_np)
    tokens_jax = jnp.array(token_ids_np)
    
    # Forward
    logits, aux_logits = model(x_jax, mask=mask_jax, token_ids=tokens_jax, training=False)
    
    print(f"Logits shape: {logits.shape}")
    print(f"Aux logits shape: {aux_logits.shape}")
    
    # Load PyTorch outputs
    logits_pt = load_dump('logits')
    
    if logits_pt is not None:
        print(f"PyTorch Logits shape: {logits_pt.shape}")
        if logits.shape == logits_pt.shape:
             print("Shapes MATCH!")
        else:
             print("Shapes MISMATCH!")
             
    # Component-wise verifications
    # 1. Feature Extractor
    # We can manually run JAX extractors and compare shapes.
    
    print("\nVerifying Components...")
    # Manual extraction in JAX (reuse model logic)
    # x_jax: (B, T, 130, 3)
    # LHand: 0-21
    x_lhand = model.feature_extractor_lhand(x_jax[:, :, 0:21, :], mask=mask_jax, training=False)
    print(f"LHand shape: {x_lhand.shape}")
    
    x_lhand_pt = load_dump('x_lhand')
    if x_lhand_pt is not None:
         print(f"PT LHand shape: {x_lhand_pt.shape}")
         assert x_lhand.shape == x_lhand_pt.shape
         
    # Rotation
    # Verify RoPE cache
    print("\nVerifying RoPE...")
    head_dim = cfg.encoder_config.encoder_dim // cfg.encoder_config.num_attention_heads # 208 // 8 = 26
    rot_emb = LlamaRotaryEmbedding(head_dim, max_position_embeddings=384, rngs=rngs)
    cos, sin = rot_emb(x_jax, seq_len=50)
    print(f"Cos shape: {cos.shape}") # Should be (1, 1, 50, 26) or (50, 26) depending on impl
    
    # Preprocessing Verification
    print("\nVerifying Preprocessing...")
    from .preprocess import Preprocess
    
    # Try to import ds_2 for comparison
    try:
        from data.ds_2 import Preprocessing as PTPreprocessing
        has_pt_ds = True
    except ImportError:
        print("Could not import data.ds_2, skipping PyTorch comparison.")
        has_pt_ds = False
        
    # Generate random raw input
    # (T, 3*N)
    T_raw = 50
    N_raw = 130
    x_raw = np.random.randn(T_raw, 3*N_raw).astype(np.float32)
    
    # JAX Preprocess
    jax_prep = Preprocess(max_len=384)
    x_jax_p, mask_jax_p = jax_prep(x_raw)
    
    print(f"JAX Preprocessed shape: {x_jax_p.shape}")
    print(f"JAX Mask shape: {mask_jax_p.shape}")
    
    if has_pt_ds:
        import torch
        pt_prep = PTPreprocessing()
        x_raw_torch = torch.from_numpy(x_raw)
        
        # ds_2 pipeline: Preprocessing() matches normalizing part.
        # interpolate_or_pad matches resizing.
        # We need to call them sequentially to match 'Preprocess' class in JAX.
        
        # 1. Normalize/Fill NaNs
        x_pt_1 = pt_prep(x_raw_torch) # (T, N, 3) ? No, ds_2 forward does permute.
        # ds_2.py forward: 
        # x.reshape(x.shape[0],3,-1).permute(0,2,1) -> (T, N, 3)
        # So x_pt_1 is (T, N, 3)
        
        # 2. Interpolate/Pad
        from data.ds_2 import interpolate_or_pad
        x_pt_final, mask_pt_final = interpolate_or_pad(x_pt_1.permute(1,2,0), max_len=384) # API differs?
        # In ds_2.py: interpolate_or_pad(data, max_len=100, mode="start")
        # data expected: (3, T) or (N, 3, T)?
        # ds_2 line 56: F.interpolate(data.permute(1,2,0), max_len).permute(2,0,1)
        # imply input data is (N, 3, T) usually? 
        # Wait, ds_2 preprocess forward returns (T, N, 3).
        # In __getitem__:
        # data = self.processor(data) -> (T, N, 3)
        # ...
        # data, mask = interpolate_or_pad(data, ...)
        # So input to interpolate_or_pad is (T, N, 3).
        
        # Inside interpolate_or_pad:
        # diff = max_len - data.shape[0] -> checks T.
        # If crop: data = F.interpolate(data.permute(1,2,0), max_len).permute(2,0,1)
        # (T, N, 3) permute(1,2,0) -> (N, 3, T).
        # interpolate on last dim (T).
        # permute(2,0,1) -> (max_len, N, 3).
        # matches.
        
        x_pt_final, mask_pt_final = interpolate_or_pad(x_pt_1, max_len=384)
        
        # Check shapes
        # PyTorch returns tensor, JAX returns numpy
        print(f"PT Preprocessed shape: {x_pt_final.shape}")
        
        # Compare
        # Note: interpolation numerical differences might exist (linear vs simple).
        # We check means/stds or strict equality for padding case.
        
        # Case 1: Padding (T=50 < 384)
        # Should be exact/close for first 50 elements.
        x_jax_p_valid = x_jax_p[:50]
        x_pt_final_valid = x_pt_final[:50].numpy()
        
        diff = np.abs(x_jax_p_valid - x_pt_final_valid).mean()
        print(f"Mean Difference (Valid Part): {diff}")
        
        if diff < 1e-5:
            print("Preprocessing MATCHES!")
        else:
            print("Preprocessing MISMATCH (High diff)!")
            
    print("\nVerification Complete.")

if __name__ == '__main__':
    verify()
