
import os
import sys
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import torch
import numpy as np
from tqdm import tqdm
import argparse
import importlib

# Add project root and subdirs to path
sys.path.append('.')
sys.path.append('configs')
sys.path.append('data') 

# Import config
import cfg_2 as cfg_ch_38 # Alias to keep code working or just rename usage
from jax_model import config as jax_config
from jax_model.model import Net
from data.ds_2 import CustomDataset, tr_collate_fn, val_collate_fn

def create_train_state(model, learning_rate, total_steps, num_epochs, warmup_steps):
    # Optax optimizer
    # Cosine schedule with warmup
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=0.0 # Or small epsilon
    )
    
    optimizer = optax.adamw(learning_rate=schedule, weight_decay=0.05) # Check default cfg weight decay
    
    # NNX Optimizer integration
    # nnx.Optimizer is not yet standard in all versions, we often use optax directly with nnx.split
    # We will use functional style for optimization step
    return optimizer, schedule

# @nnx.jit (Disabled for verification/compat reasons)
def train_step(model, optimizer, opt_state, batch, loss_config):
    # Batch Unpack
    # x: (B, T, N, 3)
    # mask: (B, T)
    # token_ids: (B, L)
    # token_ids_bwd: (B, L)
    # labels: (B, L) (shifted for loss)
    # labels_bwd: (B, L)
    
    x = batch['input']
    mask = batch['input_mask'] # (B, T)
    
    # Forward Labels (Targets)
    labels = batch['token_ids']
    # Backward Labels (Targets)
    labels_bwd = batch['token_ids_bwd']
    
    # Inputs to decoder (Shifted right)
    # We assume token_ids in 'batch' are targets. 
    # We need to shift them for decoder input.
    # PAD = 59? Start = 60? End = 61?
    # We need config values.
    # Let's extract them from config.
    pad_token_id = loss_config['pad_token_id']
    start_token_id = loss_config['start_token_id']
    
    def shift_right(x):
        # Shift right by 1, prepend start_token_id
        # x: (B, L)
        shifted = jnp.roll(x, 1, axis=1)
        shifted = shifted.at[:, 0].set(start_token_id)
        return shifted

    dec_input_ids = shift_right(labels)
    dec_input_ids_bwd = shift_right(labels_bwd) # Start token is same?
    
    def loss_fn(model):
        outputs = model(
            x, 
            mask=mask, 
            token_ids=dec_input_ids, 
            token_ids_bwd=dec_input_ids_bwd, 
            training=True
        )
        
        logits = outputs['logits']
        logits_bwd = outputs.get('logits_bwd', None)
        aux_logits = outputs['aux_logits']
        
        # CE Loss
        # logits: (B, L, V)
        # labels: (B, L)
        # Ignore index? usually pad_token_id
        # Label smoothing?
        
        one_hot = jax.nn.one_hot(labels, logits.shape[-1])
        # Smooth
        confidence = 1.0 - loss_config['label_smoothing']
        one_hot = one_hot * confidence + (1.0 - confidence) / logits.shape[-1]
        
        loss_ce = optax.softmax_cross_entropy(logits, one_hot)
        # Mask padding
        loss_mask = (labels != pad_token_id)
        loss_ce = (loss_ce * loss_mask).sum() / loss_mask.sum()
        
        loss_ce_bwd = 0.0
        if logits_bwd is not None:
            one_hot_bwd = jax.nn.one_hot(labels_bwd, logits_bwd.shape[-1])
            one_hot_bwd = one_hot_bwd * confidence + (1.0 - confidence) / logits_bwd.shape[-1]
            loss_ce_bwd = optax.softmax_cross_entropy(logits_bwd, one_hot_bwd)
            loss_mask_bwd = (labels_bwd != pad_token_id)
            loss_ce_bwd = (loss_ce_bwd * loss_mask_bwd).sum() / loss_mask_bwd.sum()
            
        # Aux Loss (BCE)
        # scores: (B,)
        scores = batch['score']
        loss_aux = optax.sigmoid_binary_cross_entropy(aux_logits, scores).mean()
        
        # Weighted Sum
        # Weights from config (approx)
        aux_wt = loss_config['aux_loss_weight']
        bwd_wt = loss_config['bwd_loss_weight']
        
        total_loss = (1 - aux_wt) * (loss_ce * (1 - bwd_wt) + bwd_wt * loss_ce_bwd) + aux_wt * loss_aux
        
        return total_loss, (loss_ce, loss_ce_bwd, loss_aux)
        
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, (l_ce, l_bwd, l_aux)), grads = grad_fn(model)
    
    updates, opt_state = optimizer.update(grads, opt_state, model)
    nnx.update(model, updates)
    
    return loss, l_ce, l_bwd, l_aux, opt_state

def make_bwd_labels(token_ids, lengths, pad_id=59):
    # token_ids: np array (B, L)
    # lengths: np array (B,)
    # Reverse the valid part of the sequence.
    # PyTorch logic: 
    # lbl_bwd = [torch.cat((torch.flip(i[:-1], [0]), i[-1:])) for i in lbl_bwd]
    # i[:-1] means all except last token (EOS is usually last valid).
    # so we flip everything BEFORE EOS.
    # And keep EOS at end.
    
    # We do this on CPU via numpy
    B, L = token_ids.shape
    bwd_ids = np.full_like(token_ids, pad_id)
    
    for i in range(B):
        l = lengths[i] # This is sequence length of INPUT audio? No, this is phrase length usually?
        # Dataset returns 'phrase' ids. token_ids include EOS and PAD.
        # We need actual length of tokens including EOS?
        # PyTorch dataset:
        # phrase_ids = ... + [EOS]
        # then padded.
        # So valid tokens are until first PAD.
        
        # Find first PAD or use L if no pad
        # Valid tokens: ids[i][ids[i] != pad_id]
        valid = token_ids[i][token_ids[i] != pad_id]
        if len(valid) == 0: continue
        
        # valid includes EOS at end.
        # Flip everything except last (EOS)
        if len(valid) > 1:
            flipped = np.concatenate([valid[:-1][::-1], valid[-1:]])
        else:
            flipped = valid # Just EOS
            
        bwd_ids[i, :len(flipped)] = flipped
        
    return bwd_ids

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args(args)
    
    # Config
    cfg = cfg_ch_38.cfg
    cfg.batch_size = args.batch_size
    
    # Disable Augmentations for JAX Verification to avoid upstream augmentation lib issues
    # (Parity check should be done on clean data anyway, and pipeline verification doesn't need complex augs)
    cfg.train_aug = None
    
    # Data
    print("Loading Data...")
    import pandas as pd
    df = pd.read_csv(cfg.train_df)
    train_df = df[df["fold"] != cfg.fold].copy()
    if args.debug:
        train_df = train_df.iloc[:100]
        
    ds = CustomDataset(train_df, cfg, aug=cfg.train_aug, mode="train")
    if args.debug:
        print("DEBUG MODE: Using Synthetic Data for Verification")
        class SyntheticLoader:
            def __init__(self, batch_size, num_batches=5):
                self.batch_size = batch_size
                self.num_batches = num_batches
                
            def __len__(self):
                return self.num_batches
                
            def __iter__(self):
                for _ in range(self.num_batches):
                    # Yield dummy batch matching ds_2 structure
                    # input: (B, T, N, 3)
                    # token_ids: (B, L)
                    # score: (B,)
                    B = self.batch_size
                    T = 100
                    N = 130
                    L = 20
                    yield {
                        'input': torch.randn(B, T, N, 3),
                        'input_mask': torch.ones(B, T),
                        'token_ids': torch.randint(0, 60, (B, L)), # 0-60 range
                        'score': torch.rand(B),
                        'seq_len': torch.full((B,), T)
                    }
        dl = SyntheticLoader(cfg.batch_size)
    else:
        dl = torch.utils.data.DataLoader(
            ds, 
            batch_size=cfg.batch_size, 
            shuffle=True, 
            num_workers=0, 
            collate_fn=tr_collate_fn,
            drop_last=True
        )
    
    # Model
    print("Initializing Model...")
    # Map Config
    jcfg = jax_config.Config(
        n_landmarks=cfg.n_landmarks,
        max_len=cfg.max_len,
        max_phrase=cfg.max_phrase,
        encoder_config=jax_config.EncoderConfig(
            input_dim=cfg.encoder_config.input_dim,
            encoder_dim=cfg.encoder_config.encoder_dim,
            num_layers=cfg.encoder_config.num_layers,
            num_attention_heads=cfg.encoder_config.num_attention_heads,
            feed_forward_expansion_factor=cfg.encoder_config.feed_forward_expansion_factor,
            conv_expansion_factor=cfg.encoder_config.conv_expansion_factor,
            conv_kernel_size=cfg.encoder_config.conv_kernel_size,
            # dropouts
            input_dropout_p=cfg.encoder_config.input_dropout_p,
            feed_forward_dropout_p=cfg.encoder_config.feed_forward_dropout_p,
            attention_dropout_p=cfg.encoder_config.attention_dropout_p,
            conv_dropout_p=cfg.encoder_config.conv_dropout_p,
        ),
        decoder_config=jax_config.DecoderConfig(
            vocab_size=len(ds.char_to_num), # + special
            d_model=cfg.transformer_config.d_model,
            num_heads=cfg.transformer_config.num_attention_heads, # HF config usually has this
            decoder_layers=cfg.transformer_config.decoder_layers,
            decoder_ffn_dim=cfg.transformer_config.decoder_ffn_dim,
            max_length=cfg.max_phrase,
            pad_token_id=ds.pad_token_id
        ),
    )
    
    # Check pad/start/end id
    jcfg.decoder_config.pad_token_id = ds.pad_token_id
    start_token_id = ds.start_token_id
    
    model = Net(jcfg, rngs=nnx.Rngs(0))
    
    # Optimizer
    total_steps = len(dl) * args.epochs
    warmup_steps = int(cfg.warmup * len(dl))
    if warmup_steps >= total_steps:
        print(f"Warning: Warmup steps {warmup_steps} >= Total steps {total_steps}. Clamping.")
        warmup_steps = max(0, total_steps - 1)
        
    optimizer, schedule = create_train_state(model, cfg.lr, total_steps, args.epochs, warmup_steps)
    opt_state = optimizer.init(model)
    
    # Loop
    print("Starting Training...")
    
    # Loss Config
    loss_config = {
        'pad_token_id': ds.pad_token_id,
        'start_token_id': ds.start_token_id,
        'label_smoothing': cfg.label_smoothing,
        'aux_loss_weight': cfg.aux_loss_weight,
        'bwd_loss_weight': cfg.bwd_loss_weight,
    }

    for epoch in range(args.epochs):
        pbar = tqdm(dl, desc=f"Epoch {epoch}")
        avg_loss = 0.0
        
        for i, batch in enumerate(pbar):
            # Convert to numpy
            batch_np = {k: v.numpy() for k, v in batch.items()}
            
            # Add bwd labels
            batch_np['token_ids_bwd'] = make_bwd_labels(batch_np['token_ids'], batch_np.get('seq_len', np.zeros(len(batch_np))), pad_id=ds.pad_token_id)
            # Actually passed 'seq_len' is audio seq len. We need phrase lengths.
            # We can deduce phrase len from token_ids padding.
            
            # Step
            loss, l_ce, l_bwd, l_aux, opt_state = train_step(model, optimizer, opt_state, batch_np, loss_config)
            loss.block_until_ready()
            
            avg_loss += loss.item()
            pbar.set_postfix(loss=loss.item(), ce=l_ce.item(), bwd=l_bwd.item(), aux=l_aux.item())
            
            if args.debug and i > 5:
                break
                
        print(f"Epoch {epoch} Loss: {avg_loss / len(pbar)}")
        
if __name__ == "__main__":
    main()
