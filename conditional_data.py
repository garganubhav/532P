import jax
import jax.numpy as jnp
import haiku as hk

import configlib

class ConditionalData(hk.Module):
    def __init__(self, c: configlib.Config, n_batches=None):
        super().__init__()
        self.c = c

    def __call__(self, noise, labels, batches, is_training):
        init = hk.initializers.RandomNormal(mean=0, stddev=0.02)

        data = hk.get_parameter("data",
            shape=[self.c.n_classes, self.c.images_per_class, self.c.im_dim, self.c.im_dim, self.c.im_chan],
            dtype=jnp.float32,
            init=init
        )

        candidates = jnp.take(data, labels, axis=0)
        noise = jnp.expand_dims(noise, axis=(1, 2, 3, 4))
        selected = jnp.take_along_axis(candidates, noise, axis=1)
        selected = jnp.squeeze(selected, axis=1)
        return selected

