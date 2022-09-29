import jax
import jax.numpy as jnp
import numpy as np
import pytest

from numpy.testing import assert_array_almost_equal
from optax import apply_updates, constant_schedule

from nag_gs.nag_gs_jax import nag4, nag_gs


class TestNAGOptimizer:

    @pytest.mark.parametrize('nag', (nag4, nag_gs))
    def test_sanity(self, nag):
        params = jnp.ones(8)
        opt = nag()
        opt_state = opt.init(params)
        assert_array_almost_equal(opt_state.trace, params)

    @pytest.mark.parametrize('nag', (nag4, nag_gs))
    def test_schedule(self, nag):
        alpha = constant_schedule(0.05)
        params = jnp.ones(8)
        opt = nag(alpha)
        opt_state = opt.init(params)
        assert_array_almost_equal(opt_state.trace, params)

    @pytest.mark.parametrize('nag', (nag4, nag_gs))
    def test_quadratic3d(self, nag):

        @jax.value_and_grad
        def fn(xs):
            return xs @ xs

        eps = 1e-3
        params = jnp.ones(3)
        opt = nag(alpha=0.05, mu=1.0, gamma=1.5)
        opt_state = opt.init(params)
        for _ in range(200):
            loss, grads = fn(params)
            if abs(loss) < eps:
                break
            grads, opt_state = opt.update(grads, opt_state, params)
            params = apply_updates(params, grads)
        else:
            pytest.fail(False, 'No convergence.')

    @pytest.mark.skip
    def test_against_reference(self):
        import torch as T
        from nag_gs.nag_gs_pytorch import NAG4
        inp = T.ones(42).requires_grad_()
        out = inp @ inp
        out.backward()
        params = np.array(inp.detach().numpy())
        grads = np.array(inp.grad.detach().numpy())

        opt_kwargs = {'alpha': 0.05, 'mu': 1.0, 'gamma': 1.5}
        opt = nag4(**opt_kwargs)
        opt_state = opt.init(params)

        ref_kwargs = {('lr' if k == 'alpha' else k): v
                      for k, v in opt_kwargs.items()}
        ref = NAG4([inp], **ref_kwargs)
        ref.step()
        ref_state = ref.state_dict()['state']
        ref_trace = np.array(ref_state[0]['v'].detach().numpy())

        grads, opt_state = opt.update(grads, opt_state, params)
        params = apply_updates(params, grads)
        assert_array_almost_equal(ref_trace, opt_state.trace)
