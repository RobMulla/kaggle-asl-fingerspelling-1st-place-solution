import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
from jax_model.model import Net
from jax_model.config import Config

def verify():
    print("Initializing JAX Config...")
    config = Config()
    config.batch_size = 2
    config.max_len = 64
    config.n_landmarks = 130
    
    # NNX Rngs
    rngs = nnx.Rngs(0)
    
    print("Initializing Model...")
    model = Net(config, rngs=rngs)
    
    # Dummy Input
    # (B, T, N, 3)
    B = 2
    T = 64
    N = 130
    
    x = jnp.ones((B, T, N, 3))
    input_mask = jnp.ones((B, T)) # All valid
    decoder_ids = jnp.ones((B, 10), dtype=jnp.int32) # Dummy targets
    
    print("Running Forward Pass...")
    try:
        logits = model(x, input_mask, decoder_ids, training=True)
        print("Forward Pass Successful!")
        print("Logits Shape:", logits.shape)
        # Expected: (B, T_dec, vocab)
        # T_dec = 10
        # vocab = 63
        assert logits.shape == (B, 10, 63)
        print("Shape Verification Passed: (2, 10, 63)")
    except Exception as e:
        print("Forward Pass Failed!")
        print(e)
        raise e

if __name__ == "__main__":
    verify()
