import jax
import jax.numpy as jnp
import haiku as hk

import configlib

def unsqueeze_noise(noise, z_dim):
    return jnp.reshape(noise, (len(noise), 1, 1, z_dim))

def get_noise(shape, prng_key):
    return jax.random.normal(prng_key, shape)

class ConvTransposeBlock(hk.Module):
    def __init__(self, out_c, kernel_size, stride, padding):
        super().__init__()
        self.out_c = out_c
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def __call__(self, x, is_training):
        init = hk.initializers.RandomNormal(mean=0, stddev=0.02)
        x = hk.Conv2DTranspose(
                self.out_c, kernel_shape=self.kernel_size, stride=self.stride, padding=self.padding, w_init=init
            )(x)
        x = hk.BatchNorm(
                create_offset=True, create_scale=True, decay_rate=0.1, scale_init=init
            )(x, is_training=is_training)
        x = jax.nn.leaky_relu(x, negative_slope=0.2)
        return x


class ResidualTransposeBlock(hk.Module):
    def __init__(self, out_c, kernel_size, stride, padding):
        super().__init__()
        self.out_c = out_c
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def __call__(self, x, is_training):
        init = hk.initializers.RandomNormal(mean=0, stddev=0.02)
        residual = hk.Conv2DTranspose(
                self.out_c, kernel_shape=self.kernel_size, stride=self.stride, padding=self.padding, w_init=init
            )(x)
        residual = hk.BatchNorm(
                create_offset=True, create_scale=True, decay_rate=0.1, scale_init=init
            )(residual, is_training=is_training)

        x = hk.Conv2DTranspose(
                self.out_c, kernel_shape=self.kernel_size, stride=self.stride, padding=self.padding,
                with_bias=False, w_init=init
            )(x)
        x = hk.BatchNorm(
                create_offset=True, create_scale=True, decay_rate=0.1, scale_init=init
            )(x, is_training=is_training)
        x = jax.nn.leaky_relu(x, negative_slope=0.2)
        x = hk.Conv2DTranspose(
                self.out_c, kernel_shape=3, stride=1, padding="SAME", with_bias=False, w_init=init
            )(x)
        x = hk.BatchNorm(
                create_offset=True, create_scale=True, decay_rate=0.1, scale_init=init
            )(x, is_training=is_training)

        return jax.nn.leaky_relu(x + residual, negative_slope=0.2)

class ConditionalResidualGenerator(hk.Module):
    def __init__(self, c: configlib.Config, n_batches=None):
        super().__init__()
        self.c = c
        self.n_batches = n_batches
        self.hidden_dim = c.z_dim + c.n_classes

    '''
    This function computes a forward pass on the network. Similar to Pytorch's forward. 
    VALID padding does no padding.
    '''

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

        # input is Z, going into a first convolution
        x = ConvTransposeBlock(out_c=self.hidden_dim*4, kernel_size=4, stride=2, padding="VALID")(
                x, is_training
        )

        padding = ((2,2), (2,2))
        # state size of x: (self.hidden_dim * 2) x 4 x 4
        x = ResidualTransposeBlock(out_c=self.hidden_dim*2, kernel_size=4, stride=2, padding=padding)(
                x, is_training
        )
        # state size of x: (self.hidden_dim * 2) x 8 x 8
        x = ResidualTransposeBlock(out_c=self.hidden_dim, kernel_size=4, stride=2, padding=padding)(
                x, is_training
        )
        # state size of x: (self.hidden_dim) x 16 x 16
        #  x = ResidualTransposeBlock(out_c=self.hidden_dim, kernel_size=4, stride=2, padding=1)(
                #  x, is_training
        #  )
        #  # state size of x: (self.hidden_dim) x 32 x 32

        clip = (32 - self.c.im_dim) // 2
        padding = ((2-clip,2-clip), (2-clip,2-clip))
        # Image gen block final
        x = hk.Conv2DTranspose(
                self.c.im_chan, kernel_shape=4, stride=2, padding=padding, with_bias=False, w_init=init)(x)
        x = jax.nn.tanh(x)
        # state size of x: (self.im_chaan) x 32 x 32

        #a = im.reshape(im.shape[0], -1)
        #return im, probs(im)
        return x


'''
Forward wrapper used to initialize params and do forward pass in the model
'''

def forward_generator(x, z_dim, im_chan, hidden_dim, is_training):
    gen = Generator(z_dim, im_chan, hidden_dim)
    target = gen(x, is_training)
    return target

def get_forward_gen(im_dim):
    def forward_gen(x, z_dim, im_chan, hidden_dim, is_training):
        gen = Generator(z_dim, im_chan, hidden_dim, im_dim=im_dim)
        target = gen(x, is_training)
        return target
    return forward_gen

if __name__ == "__main__":
    print("testing conditional_generator.py")
    test_batch = 8
    n_classes = 10
    z_dim = 64
    z_n_labels_dim = z_dim + n_classes # noise + 1-hot-labels
    im_chan = 1
    gen_hidden_dim = 64
    model_g = hk.without_apply_rng(hk.transform_with_state(forward_generator))
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
    out = model_g.apply(params_g, state_g, dummy_inp_g, z_n_labels_dim, im_chan, gen_hidden_dim, False)
    print(out[0].shape)
