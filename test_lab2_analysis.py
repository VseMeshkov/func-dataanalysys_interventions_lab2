"""Содержательные проверки SMC, потери идентифицируемости и графа."""
import numpy as np
import pytest
from scipy.stats import norm, kstest
from scipy.optimize import linprog
import lab2_analysis as A
import oscillators as L


def test_zeroing_is_nested_and_does_not_mutate():
    X = np.arange(60, dtype=float).reshape(20, 3) + 1
    original = X.copy()
    half = A.zero_input(X, 0, .5)
    full = A.zero_input(X, 0)
    np.testing.assert_array_equal(X, original)
    assert (half[:,0] == 0).sum() == 10
    assert np.all(full[:,0] == 0)
    np.testing.assert_array_equal(full[:,1:], X[:,1:])
    with pytest.raises(ValueError):
        A.zero_input(X, 2)


def test_smc_matches_exact_posterior_and_unidentified_prior():
    data = A.synthetic(n=180)
    X, y = data.X, data.y
    scale = A.prior_scale(X, y)
    for j in (None, 1):
        design = X if j is None else A.zero_input(X, j)
        samples, hist = A.smc(design, y, .08, scale, n_particles=4096, seed=11)
        mean, cov = A.gaussian_posterior(design, y, .08, scale)
        assert np.max(np.abs(samples.mean(axis=0)-mean)/np.sqrt(np.diag(cov))) < .12
        np.testing.assert_allclose(samples.std(axis=0), np.sqrt(np.diag(cov)), rtol=.12)
        assert hist.beta.iloc[-1] == 1
        assert np.all(np.diff(hist.beta) > 0)
        assert hist.ess.min() >= .799 * len(samples)
        if j is not None:
            assert kstest(samples[:,j], norm(scale=scale[j]).cdf).statistic < .05


def test_laplace_smc_recovers_signal():
    data = A.synthetic(n=150)
    scale = A.prior_scale(data.X, data.y)
    samples, hist = A.smc(data.X, data.y, .08, scale, loss="MAE", seed=13)
    # Независимый LAD-оптимум той же шумной выборки: истина не обязана
    # совпадать с оценкой на конечной реализации шума.
    n = len(data.y)
    optimum = linprog(np.r_[np.zeros(3), np.ones(n)],
        A_ub=np.block([[data.X, -np.eye(n)],[-data.X, -np.eye(n)]]),
        b_ub=np.r_[data.y,-data.y], bounds=[(None,None)]*3+[(0,None)]*n,
        method="highs")
    assert optimum.success
    assert np.max(np.abs(samples.mean(axis=0)-optimum.x[:3])/samples.std(axis=0)) < 2
    assert hist.beta.iloc[-1] == 1


def test_reused_graph_matches_original_equation():
    graph = A.expression_graph()
    values = dict(k=6.25, c=.18, b=0., x=.4, v=-.3)
    acceleration = L.evaluate_graph(graph, values)["a"]
    assert acceleration == pytest.approx(L.rhs_spring(0, np.array([.4,-.3]), 2.5, .18/5)[1])


def test_ks_distance_known_cases():
    a = np.arange(100).reshape(-1,1)
    assert A.ks_distance(a,a)[0] == 0
    assert A.ks_distance(a,a+100)[0] == 1
