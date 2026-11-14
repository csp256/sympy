"""Tests for sparse multivariate Hensel lifting (MTSHL). """
from __future__ import annotations

from random import Random

from sympy.core import random as sympy_random
from sympy.ntheory.generate import nextprime
from sympy.polys import polyconfig as config
from sympy.polys.densebasic import dmp_LC
from sympy.polys.domains import ZZ
from sympy.polys.factortools import (
    dmp_zz_factor, dmp_zz_mignotte_bound, dmp_zz_wang,
    dmp_zz_wang_lead_coeffs, dmp_zz_wang_test_points, dup_zz_factor_sqf,
)
from sympy.polys.polyerrors import EvaluationFailed, ExtraneousFactors
from sympy.polys.rings import ring
from sympy.polys.sparsehensel import dmp_zz_sparse_hensel_lifting
from sympy.polys.specialpolys import w_polys
from sympy.testing.pytest import raises


def _wang_config(f, A, u):
    """The inputs of the lifting step of Wang's algorithm for ``f`` at the
    evaluation point ``A``."""
    K = ZZ
    A = [K(a) for a in A]
    ct, T = dmp_zz_factor(dmp_LC(f, K), u - 1, K)
    cs, s, E = dmp_zz_wang_test_points(f, T, ct, A, u, K)
    _, H = dup_zz_factor_sqf(s, K)
    f, H, LC = dmp_zz_wang_lead_coeffs(f, T, cs, E, H, A, u, K)
    p = K(nextprime(dmp_zz_mignotte_bound(f, u, K)))
    return f, H, LC, A, p


def _factors(f):
    """Factors of ``f`` (a ring element) as a set, normalized."""
    coeff, factors = f.factor_list()
    return coeff, {(g, k) for g, k in factors}


def _normal(g):
    g = g.primitive()[1]
    return -g if g.LC < 0 else g


def test_dmp_zz_sparse_hensel_lifting():
    w_1 = w_polys()[0]
    R = w_1.ring
    UV, _x = ring("x", ZZ)

    p = ZZ(nextprime(R.dmp_zz_mignotte_bound(w_1)))
    A = [ZZ(-14), ZZ(3)]
    H = [44*_x**2 + 42*_x + 1, 126*_x**2 - 9*_x + 28, 187*_x**2 - 23]
    LC = [lc.drop(R.gens[0]) for lc in [-4*R.gens[1] - 4*R.gens[2],
                                       -R.gens[1]*R.gens[2]**2,
                                       R.gens[1]**2 - R.gens[2]**2]]

    factors = R.dmp_zz_sparse_hensel_lifting(w_1, H, LC, A, p)
    assert factors == R.dmp_zz_wang_hensel_lifting(w_1, H, LC, A, p)
    assert R.dmp_expand(factors) == w_1

    # the result does not depend on the seed
    for seed in range(5):
        assert R.dmp_zz_sparse_hensel_lifting(w_1, H, LC, A, p, seed=seed) == factors


def test_dmp_zz_sparse_hensel_lifting_global_rng():
    # The random choices come from a private generator.
    w_1 = w_polys()[0]
    R = w_1.ring
    config = _wang_config(R.to_dense(w_1), [-14, 3], 2)
    state = sympy_random.rng.getstate()
    dmp_zz_sparse_hensel_lifting(*config, u=2, K=ZZ)
    assert sympy_random.rng.getstate() == state


def test_dmp_zz_sparse_hensel_lifting_zero_point():
    # With zero in the evaluation point the supports of consecutive Taylor
    # coefficients are unrelated, so the strong SHL assumption fails and the
    # MDPs are solved with Zippel's interpolation (MDSolver).
    R, x, y, z, t = ring("x,y,z,t", ZZ)
    F = [x**2*(y + 2) + z*t + 3*y + 1, x**2*(t - 3) - y*z**2 + 2*z + 5,
         x*(z + 5) + y*t**2 + 7]
    f = R.to_dense(F[0]*F[1]*F[2])
    lifted = 0
    points = [[a, b, c] for a in range(-3, 4) for b in range(-3, 4)
              for c in range(-3, 4) if 0 in (a, b, c)]
    for A in points:
        try:
            config = _wang_config(f, A, 3)
        except EvaluationFailed:
            continue
        lifted += 1
        factors = dmp_zz_sparse_hensel_lifting(*config, u=3, K=ZZ)
        assert {_normal(R.from_dense(g)) for g in factors} == set(F)
        if lifted == 5:
            break
    assert lifted == 5


def test_dmp_zz_sparse_hensel_lifting_extraneous():
    # An irreducible polynomial in 7 variables whose univariate image at A
    # splits. Wang's lifting took seconds to fail (about 40 seconds for
    # the evaluation points of the original report); sparse lifting finds
    # quickly that the diophantine equations have no polynomial solution.
    R, x0, x1, x2, x3, x4, x5, x6 = ring("x0:7", ZZ)
    g = x2 + x0*x1*x3 + x0**2*x1**2*x4 + x0**3*x1**3*x5 + x0**4*x1**4*x6
    config = _wang_config(R.to_dense(g), [-2, -1, -3, 3, -2, 3], 6)
    assert len(config[1]) == 2
    raises(ExtraneousFactors,
           lambda: dmp_zz_sparse_hensel_lifting(*config, u=6, K=ZZ))

    # A polynomial from simplify() of a Kerr metric component. Wang's lifting
    # ran for minutes at this evaluation point.
    k = _kerr_polynomial(R)
    config = _wang_config(R.to_dense(k), [-1, -2, -2, 1, -1, -1], 6)
    assert len(config[1]) == 2
    raises(ExtraneousFactors,
           lambda: dmp_zz_sparse_hensel_lifting(*config, u=6, K=ZZ))


def _kerr_polynomial(R):
    x0, x1, x2, x3, x4, x5, x6 = R.gens
    return (40*x0**8*x4*x6 + 16*x0**8*x5 - 16*x0**7*x2*x4*x6 - 16*x0**7*x2*x5
        + 12*x0**6*x1**2*x3*x5 + 72*x0**6*x1**2*x4**3*x6
        + 16*x0**6*x1**2*x4**2*x5 + 32*x0**6*x1**2*x4*x6**3
        + 8*x0**6*x1**2*x4*x6 + 28*x0**6*x1**2*x5
        - 16*x0**5*x1**2*x2*x4**3*x6 - 32*x0**5*x1**2*x2*x4**2*x5
        + 6*x0**4*x1**4*x3**2*x5 + 24*x0**4*x1**4*x3*x5
        + 24*x0**4*x1**4*x4**5*x6 + 8*x0**4*x1**4*x4**4*x5
        + 64*x0**4*x1**4*x4**3*x6**3 + 8*x0**4*x1**4*x4**3*x6
        + 16*x0**4*x1**4*x4**2*x5 + 32*x0**4*x1**4*x4*x6**3
        - 32*x0**4*x1**4*x4*x6 + 18*x0**4*x1**4*x5
        + 16*x0**3*x1**4*x2*x4**5*x6 - 16*x0**3*x1**4*x2*x4**4*x5
        + x0**2*x1**6*x3**3*x5 + 9*x0**2*x1**6*x3**2*x5
        + 15*x0**2*x1**6*x3*x5 - 8*x0**2*x1**6*x4**7*x6
        + 32*x0**2*x1**6*x4**5*x6**3 - 8*x0**2*x1**6*x4**5*x6
        + 8*x0**2*x1**6*x4**4*x5 + 64*x0**2*x1**6*x4**3*x6**3
        - 64*x0**2*x1**6*x4**3*x6 + 7*x0**2*x1**6*x5
        + 16*x0*x1**6*x2*x4**7*x6 + x1**8*x3**3*x5 + 3*x1**8*x3**2*x5
        + 3*x1**8*x3*x5 - 8*x1**8*x4**7*x6 + 32*x1**8*x4**5*x6**3
        - 32*x1**8*x4**5*x6 + x1**8*x5)


def test_dmp_zz_wang_spurious_seeds():
    R, x0, x1, x2, x3, x4, x5, x6 = ring("x0:7", ZZ)
    g = x2 + x0*x1*x3 + x0**2*x1**2*x4 + x0**3*x1**3*x5 + x0**4*x1**4*x6
    f = R.to_dense(g)
    # Seeds 4 and 16 took about 40 seconds in the original report. With the
    # current random module seed 1020 is one that tries to lift.
    for seed in list(range(31)) + [1020]:
        assert dmp_zz_wang(f, 6, ZZ, seed=seed) == [f]


def test_dmp_zz_wang_kerr():
    # Seeds 13 and 16 lift spurious univariate factors of this irreducible
    # polynomial; Wang's lifting took minutes on them.
    R = ring("x0:7", ZZ)[0]
    f = R.to_dense(_kerr_polynomial(R))
    for seed in [13, 16]:
        assert dmp_zz_wang(f, 6, ZZ, seed=seed) == [f]

    # What trigsimp factors for another Kerr component; Wang's lifting did
    # not finish within minutes for seeds 5 and 7.
    R, x0, x1, x2, x3, x4 = ring("x0:5", ZZ)
    p = (9*x0**8 - 6*x0**7*x2 - 15*x0**6*x1**2*x4**2 + 24*x0**6*x1**2
         + 10*x0**5*x1**2*x2*x4**2 - 10*x0**5*x1**2*x2 + 7*x0**4*x1**4*x3**4
         - 4*x0**4*x1**4*x4**4 - 7*x0**4*x1**4*x4**2 + 11*x0**4*x1**4
         - 2*x0**3*x1**4*x2*x3**4 + 5*x0**2*x1**6*x3**4*x4**2
         + 2*x0**2*x1**6*x3**4 - 2*x0**2*x1**6*x4**6 + 2*x0**2*x1**6*x4**4
         + 2*x0**2*x1**6*x4**2 - 2*x0**2*x1**6 + 2*x0*x1**6*x2*x3**6
         + 5*x1**8*x3**4*x4**2 - 5*x1**8*x3**4 - 2*x1**8*x4**6
         + 6*x1**8*x4**4 - 6*x1**8*x4**2 + 2*x1**8)
    f = R.to_dense(p)
    for seed in range(20):
        assert dmp_zz_wang(f, 4, ZZ, seed=seed) == [f]


def _random_sparse(R, rnd, nterms, deg):
    f = R.zero
    while len(f) < nterms:
        e = [0]*R.ngens
        for _ in range(rnd.randint(0, deg)):
            e[rnd.randrange(R.ngens)] += 1
        f += rnd.choice([-1, 1])*rnd.randint(1, 30)*R({tuple(e): 1})
    return f


def _random_irreducible(R, rnd, nterms, deg):
    """A random sparse polynomial that is irreducible because it has degree
    one in some variable and is primitive with respect to it."""
    while True:
        j = rnd.randrange(1, R.ngens)
        xj = R.gens[j]
        a = _drop(R, _random_sparse(R, rnd, nterms, deg), j)
        b = _drop(R, _random_sparse(R, rnd, nterms, deg), j)
        if a.degree(R.gens[0]) <= 0 and b.degree(R.gens[0]) <= 0:
            continue
        if a.gcd(b) != 1:
            continue
        return _normal(a*xj + b)


def _drop(R, f, j):
    """Remove the variable ``x_j`` from ``f`` by setting its exponent to
    zero."""
    g = R.zero
    for m, c in f.terms():
        m = list(m)
        m[j] = 0
        g += c*R({tuple(m): 1})
    return g


def test_sparse_hensel_random_products():
    # Products of random sparse irreducible factors in 3 to 8 variables.
    rnd = Random(2025)
    for nvars in range(3, 9):
        R = ring(",".join("x%d" % i for i in range(nvars)), ZZ)[0]
        for _ in range(2):
            F = [_random_irreducible(R, rnd, 4, 4) for _ in range(rnd.randint(2, 3))]
            f = 1
            for g in F:
                f *= g
            coeff, factors = _factors(f)
            expected = {}
            for g in F:
                expected[g] = expected.get(g, 0) + 1
            assert factors == set(expected.items())
            assert coeff in (1, -1)


def test_sparse_hensel_edge_cases():
    R, x, y, z, t, s = ring("x,y,z,t,s", ZZ)
    cases = [
        # non-monic leading coefficients in x
        [x**2*y*z + 2*y + t + 1, 3*x**3*z*s - y + z**2 + 5, x*t**2 + s - 4],
        # several factors of the same degree in x
        [x**2 + y*z + 1, x**2 + t*s - 2, x**2 + y + s + 3, x**2 - z*t + 4],
        # a factor that splits for many evaluation points
        [x**2 - y**2*z + 3*y, x*y + z*t + 1, x - s],
        # factors with a common leading coefficient
        [y*t*x**2 + z + 1, y*t*x**2 + s - 1],
    ]
    for F in cases:
        f = 1
        for g in F:
            f *= g
        for content in [1, 6]:
            coeff, factors = _factors(content*f)
            assert coeff == content
            assert factors == {(_normal(g), 1) for g in F}


def test_sparse_hensel_agrees_with_wang():
    R, x, y, z, t = ring("x,y,z,t", ZZ)
    rnd = Random(5)
    for _ in range(6):
        F = [_random_sparse(R, rnd, 5, 3) + x**rnd.randint(1, 3) for _ in range(2)]
        f = F[0]*F[1]
        results = []
        for method in ['wang', 'sparse']:
            sympy_random.rng.seed(1)
            with config.using(EEZ_HENSEL_LIFTING=method):
                results.append(f.factor_list())
        assert results[0] == results[1]


def test_sparse_hensel_mdp_solvers():
    # The three solvers of the multivariate diophantine problem agree.
    from sympy.polys.sparsehensel import (
        _Context, _MDPSolver, _sp_add, _sp_mul, _NoSolution)
    rnd = Random(3)
    P = ZZ(2**61 - 1)
    for m in [2, 3, 4]:
        ctx = _Context(m, 8, [6]*m, P, ZZ, Random(m))

        def poly(nterms, d0, lead):
            f = {}
            for _ in range(nterms):
                e = [rnd.randint(0, d0 - 1)] + [rnd.randint(0, 2) for _ in range(m - 1)]
                f[ctx.pack(e)] = ZZ(rnd.randint(1, 50))
            if lead:
                f[ctx.pack([d0] + [0]*(m - 1))] = ZZ(1)
            return f

        F = [poly(4, 2, True), poly(4, 3, True), poly(3, 1, True)]
        sigma = [poly(3, 2, False), poly(3, 3, False), poly(2, 1, False)]
        c = {}
        for i, s in enumerate(sigma):
            b = s
            for l, f in enumerate(F):
                if l != i:
                    b = _sp_mul(b, f, P)
            c = _sp_add(c, b, P)
        A = [ZZ(1)]*m
        assert _MDPSolver(ctx, F, m, A).sparse_interpolate(c, sigma) == sigma
        assert _MDPSolver(ctx, F, m, A).mdsolver(c) == sigma
        assert _MDPSolver(ctx, F, m, A).dense(c) == sigma
        assert _MDPSolver(ctx, F, m, A).solve(c, [F]) == sigma
        # Adding a monomial free of x0 to c gives a problem whose unique
        # solution has denominators (resultants of the F in x0), so it has
        # no polynomial solution.
        bad = _sp_add(c, {ctx.pack([0] + [3]*(m - 1)): ZZ(1)}, P)
        raises(_NoSolution, lambda: _MDPSolver(ctx, F, m, A).mdsolver(bad))
