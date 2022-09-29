"""Package nag_gs provides an implementation of NAG-GS optimizer written in JAX
and PyTorch.
"""

backends = []

try:
    from .nag_gs_jax import nag4, nag_gs
    backends.append('jax')
except ImportError:
    pass

try:
    from .nag_gs_pytorch import NAG4
    backends.append('pytorch')
except ImportError:
    pass

if not backends:
    import warnings
    warnings.warn('It seems that there is no neither JAX/Optax nor PyTorch.',
                  RuntimeWarning)
    del warnings
