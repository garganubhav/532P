import jax
import utils
import haiku as hk


class Generator(hk.Module):
    def __init__(self, z_dim, im_chan, hidden_dim):
        super().__init__()
        self.z_dim = z_dim
        self.im_chan = im_chan
        self.hidden_dim = hidden_dim

    '''
    This function computes a forward pass on the network. Similar to Pytorch's forward. 
    VALID padding does no padding.
    '''

    def __call__(self, noise, is_training):
        init = hk.initializers.RandomNormal(mean=0, stddev=0.02)
        x = utils.unsqueeze_noise(noise, self.z_dim)

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
        im = hk.Conv2DTranspose(self.im_chan, kernel_shape=4, stride=2, padding="VALID", w_init=init)(im)
        im = jax.nn.tanh(im)

        #a = im.reshape(im.shape[0], -1)
        return im, probs(im)


'''
Forward wrapper used to initialize params and do forward pass in the model
'''


def forward_generator(x, z_dim, im_chan, hidden_dim, is_training):
    gen = Generator(z_dim, im_chan, hidden_dim)
    im, target = gen(x, is_training)
    return im, target


'''
Applies a convolution on the generated image, followed by fully connected layers.
Outputs the probability for each class.
'''


def probs(x):
    x = x.reshape(-1, 28 * 28 * 1)
    x = fc(x)
    return x


'''
A fully connected layer that returns the probabilities of each class.
'''


def fc(inp):
    fc_layer = hk.Sequential([
        hk.Linear(500),
        jax.nn.leaky_relu,
        hk.Linear(100),
        jax.nn.leaky_relu,
        hk.Linear(10)
    ])
    return fc_layer(inp)


'''
A convolutional layer with leaky relu and max pool.
'''


def conv(inp, c_out, kernel_shape):
    conv_layer = hk.Sequential([
        hk.Conv2D(c_out, kernel_shape=kernel_shape, stride=2),
        jax.nn.leaky_relu,
        hk.MaxPool(window_shape=2, strides=1, padding="VALID")
    ])

    return conv_layer(inp)
