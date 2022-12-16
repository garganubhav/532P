import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

'''
Soft cross entropy loss, where label is a probability distribution.
'''


def soft_x_ent(pred, target):
    logprobs = jax.nn.log_softmax(pred)
    return jnp.mean(-jnp.sum(jax.nn.softmax(target) * logprobs))


'''
Soft cross entropy loss, where label is a probability distribution.
'''


def soft_x_ent_novmap(pred, target):
    logprobs = jax.nn.log_softmax(pred)
    return jnp.mean(-jnp.sum(jax.nn.softmax(target) * logprobs, axis=1))


'''
Cross entropy loss that takes class indices as the target value
'''


def cross_entropy(pred, target):
    """target has to be of dimension (batch_size, 1)"""
    loss = -jnp.take_along_axis(pred, target, axis=1) + logsumexp(pred, axis=1, keepdims=True)
    return jnp.mean(loss)

def bce(pred, y):
    log_sigmoid_logit = jax.nn.log_sigmoid(pred)
    loss = -(
        y * log_sigmoid_logit + (1-y) * (log_sigmoid_logit - pred)
    )
    return jnp.mean(loss)


'''
Binary cross entropy loss with logits
'''


def bce_w_logits(pred, target):
    max_val = jnp.clip(pred, 0, None)
    loss = pred - pred * target + max_val + jnp.log(jnp.exp(-max_val) + jnp.exp((-pred - max_val)))

    return jnp.mean(loss)
