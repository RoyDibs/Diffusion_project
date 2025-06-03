import jax
import jax.numpy as jnp
from torch.utils import data
from functools import partial

class DataGenerator(data.Dataset):
    def __init__(self, s_in, cord, s_out, batch_size, gen_key):
        self.s_in = s_in
        self.cord = cord
        self.s_out = s_out
        self.N = s_in.shape[0]
        self.batch_size = batch_size
        self.key = gen_key

    def __getitem__(self, index):
        """Generate one batch of data"""
        self.key, subkey = jax.random.split(self.key)
        inputs, outputs = self.__data_generation(subkey)
        return inputs, outputs

    @partial(jax.jit, static_argnums=(0,))
    def __data_generation(self, key_i):
        """Generates data containing batch_size samples"""
        idx = jax.random.choice(key_i, self.N, (self.batch_size,), replace=False)
        s_out = self.s_out[idx, :, :, :]
        cord = self.cord[:, :]
        s_in = self.s_in[idx, :, :, :]
        # Construct batch
        inputs = (s_in, cord)
        outputs = s_out
        return inputs, outputs