import jax
import jax.numpy as jnp
import losses
import optax


'''
Accumulates gradients from a list of gradients and takes
their average
'''


def accumulate_grads(grads):
    accum = jax.tree_map(lambda x: x / len(grads), grads[0])
    for grad in grads:
        accum = jax.tree_map(lambda x, y: x + (y / len(grads)), accum, grad)

    return accum


'''
Adds DP noise to flattened gradient
'''


def noise_grads(grads, sigma, c, prng_key):
    noise_multiplier = sigma * c
    grads_flat, grads_treedef = jax.tree_flatten(grads)
    (*rngs,) = jax.random.split(prng_key, len(grads_flat))
    noised = jax.tree_multimap(lambda g, r:
        g + noise_multiplier * jax.random.normal(r, g.shape, g.dtype),
        grads_flat, rngs)
    return jax.tree_unflatten(grads_treedef, noised)


'''
Clips gradient from each training example and accumulates clipped grads 
'''


def clip_grads(grads, c):
    grads_norm = jnp.sqrt(jax.tree_util.tree_reduce(
        lambda agg, x: agg + jnp.sum(jnp.reshape(jnp.square(x), (x.shape[0], -1)), axis=-1),
        grads,
        0
    ))

    # Apply clipping norm
    grads = jax.tree_map(
        lambda x: x / jnp.maximum(
            jnp.expand_dims(grads_norm, axis=tuple(range(1, len(x.shape)))),
            c
        ),
        grads
    )

    return grads


'''
Returns a function that computes the DP classifier loss on the real data.
'''


def get_c_loss_real(model_c, c_out):
    @jax.jit
    def c_loss_real(params, img_real, img_label):
        pred = model_c.apply(params, jnp.expand_dims(img_real, axis=0), c_out)
        loss = losses.cross_entropy(pred, jnp.expand_dims(img_label, axis=0))
        return loss

    return c_loss_real

def get_accuracy(model_c, c_out):
    @jax.jit
    def accuracy(params, img_real, img_label):
        pred = model_c.apply(params, img_real, c_out)
        pred = jnp.expand_dims(jnp.argmax(pred, axis=1), axis=-1)
        correct = jnp.sum(pred == img_label)
        batch_size = img_real.shape[0]
        return correct, batch_size

    return accuracy


'''
Gets dp gradients from classifier. Feeds real data and returns gradients with dp noise.
'''


def get_dp_grads_c(loss_fn):
    @jax.jit
    def dp_grads_c(params_c, real, label, sigma, c, prng_key):
        batch_size = real.shape[0]

        vmap_grads = jax.vmap(jax.value_and_grad(loss_fn), in_axes=(None, 0, 0))
        loss, grads = vmap_grads(params_c, real, jnp.expand_dims(label, axis=1))
        clipped_grads = clip_grads(grads, c)

        # Sum gradients and add noise
        clipped_grads = jax.tree_map(lambda x: jnp.sum(x, axis=0), clipped_grads)
        noisy_grads = noise_grads(clipped_grads, sigma, c, prng_key)

        # Average gradients
        avg_grads = jax.tree_map(lambda x: x/batch_size, noisy_grads)
        return avg_grads, jnp.mean(loss)

    return dp_grads_c

def get_clipped_grads_c(loss_fn):
    @jax.jit
    def clipped_grads_c(params_c, inputs, labels, c):
        batch_size = inputs.shape[0]

        vmap_grads = jax.vmap(jax.value_and_grad(loss_fn), in_axes=(None, 0, 0))
        loss, grads = vmap_grads(params_c, jnp.expand_dims(inputs, axis=1), labels)
        clipped_grads = clip_grads(grads, c)
        # Avg gradients
        avg_grads = jax.tree_map(lambda x: jnp.sum(x, axis=0)/batch_size, clipped_grads)
        return avg_grads

    return clipped_grads_c


'''
Applies gradient updates to classifier and updates optimizer state.
'''


def update_c(opt_c, opt_c_state, grad_c_dp, params_c):
    updates_c, opt_c_state = opt_c.update(grad_c_dp, opt_c_state)
    c_params = optax.apply_updates(params_c, updates_c)
    return c_params, opt_c_state


'''
Returns a function that computes the classifier loss on the fake data.
'''


def get_c_loss_fake(model_c, c_out):
    @jax.jit
    def c_loss_fake(params, fake_img, fake_img_label):
        pred = model_c.apply(params, fake_img, c_out)
        loss = losses.soft_x_ent(pred, jnp.expand_dims(fake_img_label, axis=0))
        return loss

    return c_loss_fake
