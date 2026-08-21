"""Focused regression tests for the Pol orientation-field construction."""

import numpy as np
import pytest

import grcwa
from grcwa.fft_funs import Epsilon_fft_pol, _compute_tangent_field_pol


try:
    from autograd import grad
    AG_AVAILABLE = True
except ImportError:
    AG_AVAILABLE = False


@pytest.fixture(autouse=True)
def _numpy_backend():
    """Keep these tests independent of the global backend left by other tests."""
    grcwa.set_backend("numpy")
    yield
    try:
        grcwa.set_backend("autograd")
    except ValueError:
        pass


def _one_dimensional_bar(n):
    eps = np.ones((n, 1), dtype=float)
    eps[n // 4:3 * n // 4, 0] = 12.0
    return eps


def test_opposite_bar_interfaces_do_not_cancel_orientation():
    """Opposite normals describe one orientation and must reinforce each other."""
    projections = _compute_tangent_field_pol(
        _one_dimensional_bar(96), pol_sigma=1.0 / 12.0, pol_niter=20
    )
    p_xx, p_xy, p_yx, p_yy = [np.asarray(p) for p in projections]

    # An x-normal interface has a y-directed tangent in grcwa's eps2 basis.
    # The projection must remain defined between both pairs of periodic edges;
    # blurring signed tangent vectors used to create exact zero bands there.
    np.testing.assert_allclose(p_xx, 0.0, atol=1e-6)
    np.testing.assert_allclose(p_xy, 0.0, atol=1e-6)
    np.testing.assert_allclose(p_yx, 0.0, atol=1e-6)
    assert np.min(p_yy) > 1.0 - 1e-6


@pytest.mark.parametrize("inside_eps", [-1.0 + 0.0j, 0.0 + 1.0j])
def test_equal_magnitude_complex_contrast_still_defines_an_interface(inside_eps):
    """Phase/sign contrast must not disappear merely because |eps| is equal."""
    eps = np.ones((96, 1), dtype=complex)
    eps[24:72, 0] = inside_eps

    p_xx, p_xy, p_yx, p_yy = _compute_tangent_field_pol(eps)

    np.testing.assert_allclose(p_xx, 0.0, atol=1e-6)
    np.testing.assert_allclose(p_xy, 0.0, atol=1e-6)
    np.testing.assert_allclose(p_yx, 0.0, atol=1e-6)
    np.testing.assert_allclose(p_yy, 1.0, atol=1e-6)


@pytest.mark.parametrize(
    "normal, expected",
    [
        ((1, 0), (0.0, 0.0, 1.0)),
        ((1, 1), (0.5, -0.5, 0.5)),
        ((0, 1), (1.0, 0.0, 0.0)),
    ],
)
def test_projection_for_axis_aligned_and_diagonal_normals(normal, expected):
    """The doubled-angle field reconstructs the tangent outer product."""
    n = 32
    ix, iy = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    phase = 2.0 * np.pi * (normal[0] * ix + normal[1] * iy) / n
    eps = 2.0 + 0.25 * np.cos(phase)

    p_xx, p_xy, p_yx, p_yy = _compute_tangent_field_pol(
        eps, pol_sigma=1.0 / 12.0, pol_niter=0
    )
    expected_xx, expected_xy, expected_yy = expected

    np.testing.assert_allclose(p_xx, expected_xx, atol=5e-7)
    np.testing.assert_allclose(p_xy, expected_xy, atol=5e-7)
    np.testing.assert_allclose(p_yx, expected_xy, atol=5e-7)
    np.testing.assert_allclose(p_yy, expected_yy, atol=5e-7)


def test_fractional_sigma_has_resolution_independent_physical_support():
    """A fixed fractional sigma occupies the same part of N and 2N grids."""
    coverages = []
    for n in (64, 128):
        p_xx, _, _, p_yy = _compute_tangent_field_pol(
            _one_dimensional_bar(n), pol_sigma=0.05, pol_niter=0
        )
        projection_trace = np.asarray(p_xx) + np.asarray(p_yy)
        coverages.append(np.mean(projection_trace > 0.5))

    # This also guards the native-1D (N, 1) case: using min(Nx, Ny) would
    # leave sigma at a fraction of one pixel and only the two edge pixels valid.
    assert min(coverages) > 0.90
    assert abs(coverages[0] - coverages[1]) < 0.03


@pytest.mark.skipif(not AG_AVAILABLE, reason="autograd is not installed")
def test_pol_epsilon_matrix_autograd_matches_centered_difference():
    """The orientation blur remains differentiable through Epsilon_fft_pol."""
    nx = ny = 8
    base = np.ones((nx, ny), dtype=complex) * (1.0 + 0.02j)
    base[2:6, 2:6] = 7.0 + 0.3j

    direction = np.zeros((nx, ny), dtype=float)
    direction[2, 3] = 0.7
    direction[3, 2] = -0.4
    direction[5, 4] = 0.2

    g_vectors = np.array(
        [[0, 0], [1, 0], [-1, 0], [0, 1]], dtype=int
    )
    matrix_weight = np.linspace(0.2, 1.0, 64).reshape((8, 8))
    inverse_weight = np.linspace(-0.3, 0.4, 16).reshape((4, 4))

    def objective(alpha):
        epsinv, eps2 = Epsilon_fft_pol(
            1.0 / (nx * ny),
            base + alpha * direction,
            g_vectors,
            pol_sigma=1.0 / 12.0,
        )
        value = (
            grcwa.backend.sum(eps2 * matrix_weight)
            + grcwa.backend.sum(epsinv * inverse_weight)
        )
        return grcwa.backend.real(value)

    alpha = 0.15
    step = 1e-5

    grcwa.set_backend("autograd")
    derivative_ad = float(grad(objective)(alpha))

    grcwa.set_backend("numpy")
    derivative_fd = float(
        (objective(alpha + step) - objective(alpha - step)) / (2.0 * step)
    )

    assert np.isfinite(derivative_ad)
    assert np.isfinite(derivative_fd)
    np.testing.assert_allclose(derivative_ad, derivative_fd, rtol=5e-4, atol=1e-6)
