import pytest
import torch as T

from nag_gs.nag_gs_pytorch import NAG4, NAGGS


class TestNAGOptimizer:

    @pytest.mark.parametrize('opt_ty', (NAG4, NAGGS))
    def test_sanity(self, opt_ty):
        params = T.ones(8)
        _ = opt_ty([params])

    @pytest.mark.parametrize('opt_ty', (NAG4, NAGGS))
    def test_quadratic3d(self, opt_ty):

        def fn(xs):
            return xs @ xs

        eps = 1e-3
        params = T.ones(3, requires_grad=True)
        opt = opt_ty([params], lr=0.05, mu=1.0, gamma=1.5)
        for _ in range(200):
            loss = fn(params)
            loss.backward()
            if abs(loss) < eps:
                break
            opt.step()
            opt.zero_grad()
        else:
            pytest.fail(False, 'No convergence.')
