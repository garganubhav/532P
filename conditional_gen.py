import jax
import jax.numpy as jnp
import haiku as hk

import configlib

class ConditionalGenerator(hk.Module):
    def __init__(self, c: configlib.Config, n_batches=None):
        super().__init__()
        self.c = c
        self.n_batches = n_batches
        self.hidden_dim = c.z_dim + c.n_classes

    def __call__(self, noise, labels, batches, is_training):
        init = hk.initializers.RandomNormal(mean=0, stddev=0.02)

        # reshape inputs for convolution
        noise  = jnp.expand_dims(noise, axis=(1, 2))
        labels = jnp.expand_dims(labels, axis=(1, 2))

        if self.c.conditional_label_encoding == "embedding":
            encoded_labels = hk.Embed(vocab_size=self.c.n_classes, embed_dim=self.c.n_classes)(labels)
        elif self.c.conditional_label_encoding == "one_hot":
            encoded_labels = jax.nn.one_hot(labels, self.c.n_classes)
        else:
            raise NotImplementedError
        x = jnp.concatenate((noise, encoded_labels), axis=-1)

        if self.c.batch_conditioning:
            batches = jnp.expand_dims(batches, axis=(1, 2))
            encoded_batch = hk.Embed(vocab_size=self.n_batches, embed_dim=self.c.batch_embedding_size)(batches)
            x = jnp.concatenate((x, encoded_batch), axis=-1)

        # Image gen block 1
        im = hk.Conv2DTranspose(self.hidden_dim * 4, kernel_shape=3, stride=2, padding="VALID", w_init=init)(x)
        im = hk.BatchNorm(create_offset=True, create_scale=True, decay_rate=0.1)(im, is_training=is_training)
        im = jax.nn.leaky_relu(im)

        # Image gen block 2
        im = hk.Conv2DTranspose(self.hidden_dim * 2, kernel_shape=4, stride=1, padding="VALID", w_init=init)(im)
        im = hk.BatchNorm(create_offset=True, create_scale=True, decay_rate=0.1)(im, is_training=is_training)
        im = jax.nn.leaky_relu(im)

        # Image gen block 3
        im = hk.Conv2DTranspose(self.hidden_dim, kernel_shape=3, stride=2, padding="VALID", w_init=init)(im)
        im = hk.BatchNorm(create_offset=True, create_scale=True, decay_rate=0.1)(im, is_training=is_training)
        im = jax.nn.leaky_relu(im)

        # Image gen block final
        im = hk.Conv2DTranspose(self.c.im_chan, kernel_shape=4, stride=2, padding="VALID", w_init=init)(im)
        im = jax.nn.tanh(im)

        #a = im.reshape(im.shape[0], -1)
        #return im, probs(im)
        return im

if __name__ == "__main__":
    print("testing conditional_generator.py")
    test_batch = 8
    n_classes = 10
    z_dim = 64
    z_n_labels_dim = z_dim + n_classes # noise + 1-hot-labels
    im_chan = 1
    gen_hidden_dim = 64
    model_g = hk.transform_with_state(forward_generator)
    g_prng_seq = hk.PRNGSequence(0)
    dummy_noise_g = get_noise((test_batch, z_dim), next(g_prng_seq))
    fake_labels = jax.random.randint(next(g_prng_seq), (test_batch,), 0, n_classes)
    one_hot_labels = jax.nn.one_hot(fake_labels, n_classes)
    dummy_inp_g = jax.numpy.concatenate((dummy_noise_g, one_hot_labels), axis=1)
    print(dummy_inp_g.shape)
    print(z_n_labels_dim)
    params_g, state_g = model_g.init(rng=next(g_prng_seq),
                                     x=dummy_inp_g,
                                     z_dim=z_n_labels_dim,
                                     im_chan=im_chan,
                                     hidden_dim=gen_hidden_dim,
                                     is_training=True)
    out = model_g.apply(params_g, state_g, next(g_prng_seq), dummy_inp_g, z_n_labels_dim, im_chan, gen_hidden_dim, False)
    print(out[0].shape)
