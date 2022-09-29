"""Module nag_gs_jax implements NAG-GS optimizer in JAX/Optax. There are two
implementations. One of them keeps gamma constant and another one performs full
NAG-GS step.
"""

import chex
import jax
import jax.numpy as jnp

from functools import partial
from typing import Any, NamedTuple, Optional, Union

from optax import (GradientTransformation, Params, Schedule, Updates,
                   safe_int32_increment)

__all__ = ('nag4', 'nag_gs')

ScalarOrSchedule = Union[float, Schedule]


class NAG4State(NamedTuple):

    count: chex.Array

    trace: Params


def scale_by_nag4(alpha: ScalarOrSchedule, mu: float, gamma: float,
                  dtype: Optional[Any] = None) -> GradientTransformation:

    def fn(alpha, state, param, grad):
        # Calculate coefficients for state, params, and grads.
        norm = alpha * mu + gamma
        coefs = gamma / norm, alpha * mu / norm, -alpha / norm
        # Apply linear transformation to state, params, and grads.
        output = state * coefs[0] + param * coefs[1] + grad * coefs[2]
        return output.astype(dtype)

    def gn(alpha, state, param):
        coef = alpha / (1 + alpha)
        return state * coef - param * coef

    def init_fn(params):
        clone = partial(jnp.array, dtype=dtype, copy=True)
        trace = jax.tree_map(clone, params)
        return NAG4State(count=jnp.zeros((), jnp.int32), trace=trace)

    def update_fn(grads: Updates, state: NAG4State, params: Params):
        if callable(alpha):
            alpha_ = alpha(state.count)
        else:
            alpha_ = alpha
        # Update optimizer state.
        trace = jax.tree_map(partial(fn, alpha_), state.trace, params, grads)
        grads = jax.tree_map(partial(gn, alpha_), trace, params)
        count = safe_int32_increment(state.count)
        return grads, NAG4State(count=count, trace=trace)

    return GradientTransformation(init_fn, update_fn)


def nag4(alpha: ScalarOrSchedule = 0.05, mu: float = 1.0, gamma: float = 1.5,
         dtype: Optional[Any] = None) -> GradientTransformation:
    return scale_by_nag4(alpha, mu, gamma, dtype)


class NAGGSState(NamedTuple):

    count: chex.Array

    gamma: chex.Array

    trace: Params


def update_trace(state, params, grads, alpha, gamma, mu, dtype=None):

    def fn(state, param, grad):
        # Calculate coefficients for state, params, and grads.
        norm = alpha * mu + gamma
        coefs = gamma / norm, alpha * mu / norm, -alpha / norm
        # Apply linear transformation to state, params, and grads.
        output = state * coefs[0] + param * coefs[1] + grad * coefs[2]
        return output.astype(dtype)

    return jax.tree_map(fn, state, params, grads)


def update_grads(state, params, grads, alpha):
    def fn(state, param):
        coef = alpha / (1 + alpha)
        return state * coef - param * coef

    return jax.tree_map(fn, state, params)


def scale_by_nag_gs(alpha: ScalarOrSchedule, mu: float, gamma: float,
                    dtype: Optional[Any] = None) -> GradientTransformation:

    update_trace_fn = partial(update_trace, mu=mu, dtype=dtype)
    update_grads_fn = update_grads

    def init_fn(params):
        clone = partial(jnp.array, dtype=dtype, copy=True)
        trace = jax.tree_map(clone, params)
        return NAGGSState(count=jnp.zeros((), jnp.int32),
                          gamma=jnp.array(gamma, jnp.float32),
                          trace=trace)

    def update_fn(grads: Updates, state: NAGGSState, params: Params):
        # Apply schedule to alpha if there is any.
        if callable(alpha):
            alpha_ = alpha(state.count)
        else:
            alpha_ = alpha

        # Gauss-Siedel (GS) iteration splitted over two consecutive steps of
        # optimizer. So, this update is the end of GS iteration.
        trace = state.trace
        if state.count > 0:
            trace = update_trace_fn(trace, params, grads, alpha_, state.gamma)

        # This is the beginning of the GS iterations. Update the reset of
        # optimizer state.
        gamma = (alpha_ * mu + state.gamma) / (1 + alpha_)
        grads = update_grads_fn(trace, params, grads, alpha_)
        count = safe_int32_increment(state.count)

        # Combine all together.
        return grads, NAGGSState(count=count, gamma=gamma, trace=trace)

    return GradientTransformation(init_fn, update_fn)


def nag_gs(alpha: ScalarOrSchedule = 0.05, mu: float = 1.0, gamma: float = 1.5,
           dtype: Optional[Any] = None) -> GradientTransformation:
    """Create NAG-GS optimizer.

    Args:
        alpha: Learning rate.
        mu: Momentum mu.
        gamma: Gamma factor for integration along time.
        dtype: Optional `dtype` to be used for the first order accumulator; if
               `None` then the `dtype` is inferred from `params` and `updates`.
    Returns:
        The corresponding `GradientTransformation`.
    """
    return scale_by_nag_gs(alpha, mu, gamma, dtype)
