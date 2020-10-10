import jax
from jax import numpy as jnp, random

from flax import nn, optim

from . import common


class MLPClassifier(nn.Module):
    def apply(self, x, hidden_layers, hidden_dim, n_classes, categorical=False):
        x = jnp.reshape(x, (x.shape[0], -1))
        for layer in range(hidden_layers):
            x = nn.Dense(x, hidden_dim, name=f'fc{layer}')
            x = nn.relu(x)
        preds = nn.Dense(x, n_classes, name=f'fc{hidden_layers}')
        if categorical:
            preds = nn.log_softmax(preds)
        return preds


def make_algorithm(input_shape, n_classes, n_layers=2, h_dim=512):
    def init_fn(seed):
        rng = random.PRNGKey(seed)
        classifier = MLPClassifier.partial(hidden_layers=n_layers,
                                           hidden_dim=h_dim,
                                           n_classes=n_classes)
        _, initial_params = classifier.init_by_shape(rng, [(128, *input_shape)])
        initial_model = nn.Model(classifier, initial_params)
        optimizer = optim.Adam(1e-4).create(initial_model)
        return optimizer

    @jax.jit
    def train_step_fn(optimizer, batch):
        batch = common.batch_to_jax(batch)
        loss, grad = common.loss_and_grad_fn(optimizer.target, batch)
        optimizer = optimizer.apply_gradient(grad)
        return optimizer, loss

    @jax.jit
    def eval_fn(optimizer, batch):
        batch = common.batch_to_jax(batch)
        return common.loss_fn(optimizer.target, batch)

    return init_fn, train_step_fn, eval_fn
