"""Package nag_gs provides an implementation of NAG-GS optimizer written in JAX
and PyTorch.
"""

__all__ = ()
__backend__ = []

try:
    from .nag_gs_jax import nag4, nag_gs
    __all__ += ('nag4', 'nag_gs')
    __backend__.append('jax')
except ImportError:
    pass

try:
    from .nag_gs_pytorch import NAG4, NAGGS
    __all__ += ('NAG4', 'NAGGS')
    __backend__.append('pytorch')
except ImportError:
    pass

if not __backend__:
    import warnings
    warnings.warn('It seems that there is no neither JAX/Optax nor PyTorch.',
                  RuntimeWarning)
    del warnings
