"""Sparse multivariate Hensel lifting over ``ZZ`` (Monagan and Tuncer's MTSHL).

Wang's multivariate Hensel lifting (:func:`~.dmp_zz_wang_hensel_lifting`)
recovers the variables `x_2, \\dots, x_n` of the factors one at a time. In
the `j`-th step it expands the factors as Taylor series in `x_j - \\alpha_j`
and finds the Taylor coefficients by solving multivariate polynomial
diophantine problems (MDPs). Wang solves each MDP with the same ideal-adic
method, recursively and with dense polynomials, which costs time exponential
in the number of variables when the evaluation point is nonzero and the
factors are sparse.

The functions here keep the outer structure of Wang's lifting (the same
evaluation point, the same leading coefficient correction and the same
Taylor expansion) and solve the MDPs by sparse interpolation instead:

* The solution `\\sigma_k` of the `k`-th MDP of a step is interpolated
  with the support of `\\sigma_{k-1}` as its form. This is the *strong SHL
  assumption* of Algorithm 5 (MTSHL) of [MonaganTuncer20]_, which holds
  with high probability when `\\alpha_j` is random (Lemma 1 there). The
  coefficients of the form are found from univariate images at powers of a
  random point by solving transposed Vandermonde systems ([Zippel90]_;
  Algorithm 4, SparseInterpolate, of [MonaganTuncer20]_). Monagan and
  Tuncer use bivariate images and a dense bivariate solver to need fewer
  images; univariate images keep the code simple.
* If that fails, the support of `\\sigma_0` (the *weak SHL assumption* of
  [Kaltofen85]_, Section 3.1 of [MonaganTuncer20]_) is tried, and then
  Zippel's sparse interpolation with a form taken from one random image
  (Algorithm 3, MDSolver, of [MonaganTuncer20]_), which assumes nothing
  about `\\alpha_j`; this is the usual case when `\\alpha_j = 0`. All
  factors are interpolated together, as suggested in Section 9 of
  [MonaganTuncer20]_ and in [MonaganTuncer18]_.
* Only if the random choices keep being unlucky is Wang's dense solver
  :func:`~.dmp_zz_diophantine` used for that MDP.

Every candidate solution is checked at a random point before it is used,
each step checks the lifted factorization at a random point, and the final
factors are multiplied out over the integers and compared with ``f``. The
randomness affects only the running time, never the result: the factors
returned are the unique factors that Wang's lifting computes.

Polynomials are stored sparsely as dicts mapping packed monomials (the
exponent of `x_i` occupies a fixed bit field of an integer, so monomial
multiplication is integer addition) to coefficients in `[0, P)`.

References
==========

[MonaganTuncer20]_, [MonaganTuncer16]_, [MonaganTuncer18]_, [Zippel90]_,
[Kaltofen85]_ (see the bibliography of the polys module).

"""

from __future__ import annotations

from random import Random

from sympy.polys.densebasic import (
    dmp_degree_list, dmp_from_dict, dmp_to_dict,
)
from sympy.polys.polyerrors import ExtraneousFactors


# A prime used for lifting when the prime chosen by Wang's algorithm is
# small. Sparse interpolation needs a large prime: the probability that
# random evaluations are unlucky is proportional to 1/P (Proposition 3 of
# [MonaganTuncer20]).
_MERSENNE_61 = 2**61 - 1

# How many times a solver redraws its random evaluation points before it
# gives up and lets the caller fall back to a more robust method.
_RETRIES = 3


class _Unlucky(Exception):
    """A random choice was unlucky (e.g. an evaluation made two univariate
    images share a factor, or two monomials evaluate to the same value)."""


class _FormFailure(Exception):
    """The assumed support (form) of a solution was wrong."""


class _NoSolution(Exception):
    """The diophantine problem has no polynomial solution within the degree
    bounds, so the univariate images were extraneous."""


class _Context:
    """Shared data of one lifting: the prime, monomial packing, degree
    bounds and the random number generator."""

    def __init__(self, nvars, bits, bounds, P, K, rng):
        self.nvars = nvars
        self.bits = bits
        self.mask = (1 << bits) - 1
        self.bounds = bounds
        self.P = P
        self.K = K
        self.rng = rng

    def pack(self, exps):
        m, bits = 0, self.bits
        for i, e in enumerate(exps):
            m |= e << (i*bits)
        return m

    def unpack(self, m):
        bits, mask = self.bits, self.mask
        return tuple((m >> (i*bits)) & mask for i in range(self.nvars))

    def degree(self, m, i):
        return (m >> (i*self.bits)) & self.mask

    def nonzero(self):
        """A random nonzero element of `Z_P`."""
        return self.K(self.rng.randrange(1, int(self.P)))


def _sp_add(f, g, P):
    h = dict(f)
    for m, c in g.items():
        c = (h.get(m, 0) + c) % P
        if c:
            h[m] = c
        else:
            h.pop(m, None)
    return h


def _sp_sub(f, g, P):
    h = dict(f)
    for m, c in g.items():
        c = (h.get(m, 0) - c) % P
        if c:
            h[m] = c
        else:
            h.pop(m, None)
    return h


def _sp_mul(f, g, P):
    """Multiply sparse polynomials, reducing modulo ``P`` (unless ``P`` is
    None)."""
    if len(f) > len(g):
        f, g = g, f
    h = {}
    get = h.get
    gitems = list(g.items())
    for ma, ca in f.items():
        for mb, cb in gitems:
            m = ma + mb
            h[m] = get(m, 0) + ca*cb
    if P is None:
        return {m: c for m, c in h.items() if c}
    h = {m: c % P for m, c in h.items()}
    return {m: c for m, c in h.items() if c}


def _sp_scale(f, a, P):
    h = {m: c*a % P for m, c in f.items()}
    return {m: c for m, c in h.items() if c}


def _sp_shift(f, m0):
    """Multiply by the monomial ``m0``."""
    return {m + m0: c for m, c in f.items()}


def _sp_degree(ctx, f, i):
    return max((ctx.degree(m, i) for m in f), default=-1)


def _sp_eval_var(ctx, f, i, a):
    """Substitute `x_i = a`."""
    P, shift = ctx.P, i*ctx.bits
    mask = ctx.mask << shift
    pw = [ctx.K.one]
    h = {}
    for m, c in f.items():
        e = (m & mask) >> shift
        while len(pw) <= e:
            pw.append(pw[-1]*a % P)
        m -= e << shift
        h[m] = h.get(m, 0) + c*pw[e]
    h = {m: c % P for m, c in h.items()}
    return {m: c for m, c in h.items() if c}


def _binomials(n, P):
    """Rows ``0..n`` of Pascal's triangle modulo ``P``."""
    rows = [[1]]
    for k in range(1, n + 1):
        prev = rows[-1]
        rows.append([1] + [(prev[t - 1] + prev[t]) % P for t in range(1, k)] + [1])
    return rows


def _sp_taylor(ctx, f, i, a):
    """Taylor coefficients of ``f`` in `x_i - a`, as a list of polynomials
    free of `x_i`."""
    P, K, shift = ctx.P, ctx.K, i*ctx.bits
    mask = ctx.mask << shift
    n = _sp_degree(ctx, f, i)
    if n < 0:
        return []
    binom = _binomials(n, P)
    pw = [K.one]
    for _ in range(n):
        pw.append(pw[-1]*a % P)
    T = [{} for _ in range(n + 1)]
    for m, c in f.items():
        e = (m & mask) >> shift
        m -= e << shift
        row = binom[e]
        for k in range(e + 1):
            Tk = T[k]
            Tk[m] = Tk.get(m, 0) + c*row[k]*pw[e - k]
    result = []
    for Tk in T:
        Tk = {m: c % P for m, c in Tk.items()}
        result.append({m: c for m, c in Tk.items() if c})
    return result


def _sp_untaylor(ctx, T, i, a):
    """Inverse of :func:`_sp_taylor`: compute ``sum(T[k]*(x_i - a)**k)``."""
    P, K, shift = ctx.P, ctx.K, i*ctx.bits
    n = len(T) - 1
    binom = _binomials(max(n, 0), P)
    pw = [K.one]
    for _ in range(n):
        pw.append(-pw[-1]*a % P)
    h = {}
    for k, Tk in enumerate(T):
        row = binom[k]
        for t in range(k + 1):
            b = row[t]*pw[k - t] % P
            if not b:
                continue
            e = t << shift
            for m, c in Tk.items():
                h[m + e] = h.get(m + e, 0) + c*b
    h = {m: c % P for m, c in h.items()}
    return {m: c for m, c in h.items() if c}


def _power_tables(ctx, point, nvars):
    """Tables of powers of ``point[v]`` up to the degree bound of `x_v`."""
    P, K = ctx.P, ctx.K
    tables = []
    for v in range(nvars):
        b, row = point[v], [K.one]
        for _ in range(max(ctx.bounds[v], 1)):
            row.append(row[-1]*b % P)
        tables.append(row)
    return tables


def _monomial_value(ctx, m, tables, first, nvars):
    """Value of the monomial ``m`` in the variables ``first..nvars-1``."""
    P, bits, mask = ctx.P, ctx.bits, ctx.mask
    val = 1
    for v in range(first, nvars):
        e = (m >> (v*bits)) & mask
        if e:
            row = tables[v]
            val = val*(row[e] if e < len(row) else pow(row[1], e, P)) % P
    return val


def _sp_eval_point(ctx, f, tables, nvars):
    """Evaluate ``f`` at a point in all of its variables."""
    P = ctx.P
    return sum(c*_monomial_value(ctx, m, tables, 0, nvars) for m, c in f.items()) % P


def _sp_eval_x0(ctx, f, tables, nvars):
    """Evaluate the variables `x_1, \\dots` of ``f``, giving a dense
    univariate polynomial in `x_0`."""
    P, mask = ctx.P, ctx.mask
    acc = {}
    for m, c in f.items():
        e = m & mask
        acc[e] = acc.get(e, 0) + c*_monomial_value(ctx, m, tables, 1, nvars)
    return _dup_from_dict(acc, P, ctx.K)


def _dup_from_dict(acc, P, K):
    if not acc:
        return []
    n = max(acc)
    f = [0]*(n + 1)
    for e, c in acc.items():
        f[n - e] = c % P
    return _dup_strip(f)


def _dup_strip(f):
    i = 0
    while i < len(f) and not f[i]:
        i += 1
    return f[i:] if i else f


def _dup_coeff(f, e):
    n = len(f) - 1 - e
    return f[n] if 0 <= n < len(f) else 0


def _inverse(a, P):
    """Inverse of ``a`` modulo the prime ``P``."""
    return pow(int(a), -1, int(P))


# Dense univariate arithmetic in Z_P[x_0] (lists, highest degree first). The
# helpers in galoistools would do, but these avoid a modular inversion per
# division, which dominates on small inputs.

def _dup_mul(f, g, P):
    if not f or not g:
        return []
    h = [0]*(len(f) + len(g) - 1)
    for i, a in enumerate(f):
        if a:
            for j, b in enumerate(g):
                h[i + j] += a*b
    return [c % P for c in h]


def _dup_divmod(f, g, P, ginv=None):
    dg = len(g) - 1
    if len(f) <= dg:
        return [], f
    if ginv is None:
        ginv = _inverse(g[0], P)
    r = list(f)
    q = []
    for i in range(len(f) - dg):
        c = r[i]*ginv % P
        q.append(c)
        if c:
            for k in range(1, dg + 1):
                r[i + k] = (r[i + k] - c*g[k]) % P
    return q, _dup_strip(r[len(f) - dg:])


def _dup_rem(f, g, P, ginv=None):
    return _dup_divmod(f, g, P, ginv)[1]


def _dup_sub(f, g, P):
    n = max(len(f), len(g))
    f = [0]*(n - len(f)) + list(f)
    g = [0]*(n - len(g)) + list(g)
    return _dup_strip([(a - b) % P for a, b in zip(f, g)])


def _dup_invmod(a, f, P):
    """Inverse of ``a`` modulo ``f`` in `Z_P[x]` by the extended Euclidean
    algorithm. Raises ``_Unlucky`` if they are not coprime."""
    r0, r1 = f, _dup_rem(a, f, P)
    s0, s1 = [], [1]
    while r1:
        q, r = _dup_divmod(r0, r1, P)
        r0, r1 = r1, r
        s0, s1 = s1, _dup_sub(s0, _dup_mul(q, s1, P), P)
    if len(r0) != 1:
        raise _Unlucky
    inv = _inverse(r0[0], P)
    return [c*inv % P for c in s0]


def _uni_inverses(F, P, K):
    """For univariate ``F[i]``, compute `w_i` with
    `w_i \\prod_{l \\ne i} F_l \\equiv 1 \\pmod{F_i}`, and the inverses
    of the leading coefficients of ``F``."""
    W = []
    finv = [_inverse(f[0], P) for f in F]
    for i, f in enumerate(F):
        g = [1]
        for l, h in enumerate(F):
            if l != i:
                g = _dup_rem(_dup_mul(g, h, P), f, P, finv[i])
        W.append(_dup_invmod(g, f, P))
    return W, finv


def _uni_solve(F, W, c, P, K):
    """Solve `\\sum_i \\sigma_i \\prod_{l \\ne i} F_l = c` in `Z_P[x_0]`
    with `\\deg \\sigma_i < \\deg F_i`, given the inverses ``W`` of
    :func:`_uni_inverses`. Since `\\deg c < \\sum_i \\deg F_i`, this is the
    unique solution by the Chinese remainder theorem."""
    W, finv = W
    return [_dup_rem(_dup_mul(_dup_rem(c, f, P, fi), w, P), f, P, fi)
            for f, w, fi in zip(F, W, finv)]


def _vandermonde(ms, vs, P, K):
    """Solve the transposed Vandermonde system
    `\\sum_l c_l m_l^n = v_n` for `n = 1, \\dots, t` in `O(t^2)`
    (Zippel, [Zippel90]_; Step 15 of Algorithm 4 of [MonaganTuncer20]_)."""
    t = len(ms)
    # master polynomial prod(z - m_l), low degree first
    M = [K.one]
    for m in ms:
        nxt = [K.zero]*(len(M) + 1)
        for k, a in enumerate(M):
            nxt[k + 1] = (nxt[k + 1] + a) % P
            nxt[k] = (nxt[k] - a*m) % P
        M = nxt
    cs = []
    for m in ms:
        # q = M/(z - m) by synthetic division, low degree first
        q = [K.zero]*t
        acc = K.zero
        for k in range(t, 0, -1):
            acc = (M[k] + acc*m) % P
            q[k - 1] = acc
        num = sum(qk*vk for qk, vk in zip(q, vs)) % P
        den = K.zero
        for qk in reversed(q):
            den = (den*m + qk) % P
        den = den*m % P
        if not den:
            raise _Unlucky
        cs.append(num*_inverse(den, P) % P)
    return cs


class _MDPSolver:
    """Solve `\\sum_i \\sigma_i \\prod_{l \\ne i} F_l = c` in
    `Z_P[x_0, \\dots, x_{m-1}]` with `\\deg_{x_0} \\sigma_i <
    \\deg_{x_0} F_i` for a fixed list of factors ``F``.

    Random evaluation points and the univariate images of ``F`` at them are
    cached, because Hensel lifting solves many problems with the same
    ``F``.
    """

    def __init__(self, ctx, F, m, A):
        self.ctx = ctx
        self.F = F
        self.m = m
        self.A = A
        self.degs = [_sp_degree(ctx, f, 0) for f in F]
        self._images = None
        self._check = None
        self._subsolvers = None
        if m == 1:
            P, K = ctx.P, ctx.K
            self.uF = [_sp_eval_x0(ctx, f, None, 1) for f in F]
            self.uW = _uni_inverses(self.uF, P, K)

    # -- evaluation at random points -------------------------------------

    def _univariate_images(self, point):
        """Univariate images of ``F`` at ``point`` and their inverses."""
        ctx = self.ctx
        tables = _power_tables(ctx, point, self.m)
        uF = [_sp_eval_x0(ctx, f, tables, self.m) for f in self.F]
        if any(len(u) - 1 != d for u, d in zip(uF, self.degs)):
            raise _Unlucky
        return tables, uF, _uni_inverses(uF, ctx.P, ctx.K)

    def _checker(self):
        """A random point (cached) at which candidate solutions are
        checked."""
        ctx = self.ctx
        for _ in range(_RETRIES):
            if self._check is not None:
                return self._check
            point = [ctx.nonzero() for _ in range(self.m)]
            try:
                self._check = self._univariate_images(point)
            except _Unlucky:
                pass
        raise _Unlucky

    def verify(self, c, sigma):
        """Check a candidate solution at a random point."""
        ctx = self.ctx
        P, K = ctx.P, ctx.K
        tables, uF, uW = self._checker()
        uc = _sp_eval_x0(ctx, c, tables, self.m)
        expected = _uni_solve(uF, uW, uc, P, K)
        got = [_sp_eval_x0(ctx, s, tables, self.m) for s in sigma]
        return expected == got

    # -- the solvers -------------------------------------------------------

    def solve(self, c, forms=None):
        """Solve the problem with right hand side ``c``. ``forms`` are
        guesses for the supports of the solutions, tried in order."""
        ctx = self.ctx
        if self.m == 1:
            uc = _sp_eval_x0(ctx, c, None, 1)
            sol = _uni_solve(self.uF, self.uW, uc, ctx.P, ctx.K)
            return [_sp_from_dup(s) for s in sol]
        for form in (forms or ())[:1 if self.m == 2 else None]:
            try:
                return self.sparse_interpolate(c, form)
            except _FormFailure:
                pass
        # In two variables Wang's solver is not exponential (it solves one
        # univariate problem per power of x_1 - a_1) and is cheaper than
        # MDSolver.
        for _ in range(_RETRIES if self.m > 2 else 0):
            try:
                return self.mdsolver(c)
            except (_Unlucky, _FormFailure):
                self._subsolvers = None
        sigma = self.dense(c)
        if not self.verify(c, sigma):
            # Wang's solver is exact when a polynomial solution exists
            raise _NoSolution
        return sigma

    def sparse_interpolate(self, c, form):
        """Interpolate the solutions assuming their supports are contained
        in those of ``form`` (Algorithm 4, SparseInterpolate, of [MonaganTuncer20]_,
        with univariate images and all factors at once)."""
        for _ in range(_RETRIES):
            try:
                sigma = self._sparse_interpolate(c, form)
            except _Unlucky:
                self._images = None
                continue
            if self.verify(c, sigma):
                return sigma
            raise _FormFailure
        raise _FormFailure

    def _image(self, n):
        """The ``n``-th univariate images (``n >= 1``) of ``F`` at the
        powers of the cached random point (Section 2.4 of [MonaganTuncer20]_)."""
        ctx = self.ctx
        P, K = ctx.P, ctx.K
        if self._images is None:
            point = [ctx.nonzero() for _ in range(self.m)]
            tables = _power_tables(ctx, point, self.m)
            self._images = (tables, [])
        tables, imgs = self._images
        while len(imgs) < n:
            k = len(imgs) + 1
            uF = [_sp_eval_x0(ctx, f, _powered(tables, k, P), self.m)
                  for f in self.F]
            if any(len(u) - 1 != d for u, d in zip(uF, self.degs)):
                raise _Unlucky
            imgs.append((uF, _uni_inverses(uF, P, K)))
        return imgs[n - 1]

    def _sparse_interpolate(self, c, form):
        ctx = self.ctx
        P, K, mask = ctx.P, ctx.K, ctx.mask
        groups = []
        s = 0
        for f, d in zip(form, self.degs):
            g = {}
            for mono in f:
                e = mono & mask
                if e < d:
                    g.setdefault(e, []).append(mono)
            groups.append(g)
            s = max(s, max((len(v) for v in g.values()), default=0))
        if not s:
            raise _FormFailure
        self._image(s)
        tables = self._images[0]
        # monomial values; they must be distinct within a group
        values = []
        for g in groups:
            vg = {}
            for e, monos in g.items():
                vals = [_monomial_value(ctx, mono, tables, 1, self.m)
                        for mono in monos]
                if len(set(vals)) != len(vals):
                    raise _Unlucky
                vg[e] = vals
            values.append(vg)
        # univariate images of c and of the solutions
        cimgs = _images_of(ctx, c, tables, s, self.m)
        sols = []
        for n in range(1, s + 1):
            uF, uW = self._image(n)
            sols.append(_uni_solve(uF, uW, cimgs[n - 1], P, K))
        sigma = []
        for i, (g, vg) in enumerate(zip(groups, values)):
            sig = {}
            for n in range(s):
                sol = sols[n][i]
                for e in range(len(sol)):
                    if sol[len(sol) - 1 - e] and e not in g:
                        raise _FormFailure
            for e, monos in g.items():
                vals = vg[e]
                t = len(monos)
                vs = [_dup_coeff(sols[n][i], e) for n in range(s)]
                coeffs = _vandermonde(vals, vs[:t], P, K)
                # the remaining images are a free consistency check
                for n in range(t, s):
                    if sum(cl*pow(ml, n + 1, P) for cl, ml in zip(coeffs, vals)) % P != vs[n] % P:
                        raise _FormFailure
                for mono, cl in zip(monos, coeffs):
                    if cl:
                        sig[mono] = cl
            sigma.append(sig)
        return sigma

    def mdsolver(self, c):
        """Zippel's sparse interpolation of the solutions with a form taken
        from one random image (Algorithm 3, MDSolver, of [MonaganTuncer20]_). The last
        variable is interpolated densely, the others recursively."""
        ctx = self.ctx
        P, K = ctx.P, ctx.K
        v = self.m - 1
        bound = ctx.bounds[v]
        shift = v*ctx.bits
        if self._subsolvers is None:
            self._subsolvers = []
        sigma = None
        q = [K.one]  # prod (x_v - beta), low degree first
        form = None
        for k in range(bound + 2):
            beta, sub = self._subsolver(k)
            cb = _sp_eval_var(ctx, c, v, beta)
            if form is None:
                image = form = sub.solve(cb, ())
            else:
                image = sub.solve_with_form(cb, form)
            if sigma is None:
                sigma = list(image)
            else:
                current = [_sp_eval_var(ctx, s, v, beta) for s in sigma]
                if current == image:
                    # the interpolant has stabilized: check it
                    if self.verify(c, sigma):
                        return sigma
                else:
                    qb = K.zero
                    for a in reversed(q):
                        qb = (qb*beta + a) % P
                    inv = _inverse(qb, P)
                    for i, (cur, img) in enumerate(zip(current, image)):
                        diff = _sp_scale(_sp_sub(img, cur, P), inv, P)
                        if not diff:
                            continue
                        s = sigma[i]
                        for e, a in enumerate(q):
                            if a:
                                s = _sp_add(s, _sp_shift(_sp_scale(diff, a, P), e << shift), P)
                        sigma[i] = s
            nxt = [K.zero]*(len(q) + 1)
            for e, a in enumerate(q):
                nxt[e + 1] = (nxt[e + 1] + a) % P
                nxt[e] = (nxt[e] - a*beta) % P
            q = nxt
        if self.verify(c, sigma):
            return sigma
        raise _NoSolution

    def _subsolver(self, k):
        """The ``k``-th random value of the last variable and a solver for
        the problem with the last variable evaluated there (cached)."""
        ctx = self.ctx
        subs = self._subsolvers
        v = self.m - 1
        while len(subs) <= k:
            used = {b for b, _ in subs}
            for _ in range(_RETRIES):
                beta = ctx.nonzero()
                if beta in used:
                    continue
                Fb = [_sp_eval_var(ctx, f, v, beta) for f in self.F]
                if any(_sp_degree(ctx, f, 0) != d for f, d in zip(Fb, self.degs)):
                    continue
                try:
                    sub = _MDPSolver(ctx, Fb, v, self.A)
                except _Unlucky:
                    continue
                subs.append((beta, sub))
                break
            else:
                raise _Unlucky
        return subs[k]

    def solve_with_form(self, c, form):
        """Solve with a known form; a wrong form here means an unlucky
        evaluation point in the caller."""
        if self.m == 1:
            return self.solve(c)
        return self.sparse_interpolate(c, form)

    def dense(self, c):
        """Wang's dense ideal-adic solver (Algorithm 2 of [MonaganTuncer20]_)."""
        from sympy.polys.factortools import dmp_zz_diophantine
        ctx = self.ctx
        P, K = ctx.P, ctx.K
        u = self.m - 1
        F = [_sp_to_dense(ctx, f, u) for f in self.F]
        cd = _sp_to_dense(ctx, c, u)
        d = max(ctx.bounds[1:])
        S = dmp_zz_diophantine(F, cd, self.A[:u], d, P, u, K)
        return [_sp_from_dense(ctx, s, u, P) for s in S]


def _powered(tables, k, P):
    """Power tables of the ``k``-th power of the point of ``tables``."""
    out = []
    for row in tables:
        b = row[1] if len(row) > 1 else 1
        bk = pow(b, k, P)
        new = [row[0]]
        for _ in range(len(row) - 1):
            new.append(new[-1]*bk % P)
        out.append(new)
    return out


def _images_of(ctx, f, tables, s, m):
    """Univariate images of ``f`` at the powers ``1..s`` of the point of
    ``tables``, updating one coefficient array (Section 2.4 of [MonaganTuncer20]_)."""
    P, K, mask = ctx.P, ctx.K, ctx.mask
    es, cs, ths = [], [], []
    for mono, c in f.items():
        es.append(mono & mask)
        cs.append(c)
        ths.append(_monomial_value(ctx, mono, tables, 1, m))
    imgs = []
    for _ in range(s):
        acc = {}
        for t in range(len(cs)):
            ct = cs[t]*ths[t] % P
            cs[t] = ct
            e = es[t]
            acc[e] = acc.get(e, 0) + ct
        imgs.append(_dup_from_dict(acc, P, K))
    return imgs


def _sp_from_dup(f):
    n = len(f) - 1
    return {n - k: c for k, c in enumerate(f) if c}


def _sp_to_dense(ctx, f, u):
    K = ctx.K
    d = {ctx.unpack(m)[:u + 1]: K(c) for m, c in f.items()}
    return dmp_from_dict(d, u, ctx.K)


def _sp_from_dense(ctx, f, u, P):
    pad = (0,)*(ctx.nvars - u - 1)
    h = {}
    for exps, c in dmp_to_dict(f, u, ctx.K).items():
        c = c % P
        if c:
            h[ctx.pack(exps + pad)] = c
    return h


def _prefix_coeff(ctx, pre, H, l, k):
    """Coefficient of `y^k` in ``pre[l-1]*H[l]`` (both lists of Taylor
    coefficients)."""
    P = ctx.P
    acc = {}
    prev, h = pre[l - 1], H[l]
    for t in range(k + 1):
        a, b = prev[t], h[k - t]
        if a and b:
            acc = _sp_add(acc, _sp_mul(a, b, P), P)
    return acc


def _random_check(ctx, A_T, H_T, nvars):
    """Check ``prod(H_T) == A_T`` (lists of Taylor coefficients) at a random
    point."""
    P = ctx.P
    point = [ctx.nonzero() for _ in range(nvars)]
    tables = _power_tables(ctx, point, nvars)
    y = ctx.nonzero()

    def value(T):
        acc = 0
        for f in reversed(T):
            acc = (acc*y + _sp_eval_point(ctx, f, tables, nvars)) % P
        return acc

    prod = 1
    for T in H_T:
        prod = prod*value(T) % P
    return prod == value(A_T)


def dmp_zz_sparse_hensel_lifting(f, H, LC, A, p, u, K, seed=None):
    r"""
    Sparse parallel Hensel lifting (MTSHL).

    Takes the same arguments as :func:`~.dmp_zz_wang_hensel_lifting` and
    returns the same factors: ``f`` in `Z[x_0, \dots, x_u]`, its univariate
    factors ``H`` at `x_i = A_{i-1}`, the leading coefficients ``LC`` of the
    true factors (polynomials in `x_1, \dots, x_u`) and a prime ``p``.

    The Taylor coefficients of the factors in each step are found by sparse
    interpolation (Monagan and Tuncer, [MonaganTuncer20]_), which takes time
    polynomial in the number of terms of the factors instead of Wang's
    time, exponential in the number of variables.

    ``seed`` seeds a private random number generator used to choose
    evaluation points; it affects the running time, not the result.
    Raises ``ExtraneousFactors`` when ``H`` do not lift to factors of ``f``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> from sympy.polys.sparsehensel import dmp_zz_sparse_hensel_lifting
    >>> R, x, y, z = ring("x,y,z", ZZ)
    >>> f = R.to_dense((x*y + z + 1)*(x + y*z - 2))
    >>> H = [[ZZ(2), ZZ(7)], [ZZ(1), ZZ(10)]]  # 2*x + 7 and x + 10 at y=2, z=6
    >>> LC = [[[ZZ(1)], []], [[ZZ(1)]]]  # y and 1
    >>> F = dmp_zz_sparse_hensel_lifting(f, H, LC, [ZZ(2), ZZ(6)], ZZ(101), 2, K=ZZ)
    >>> [R.from_dense(g) for g in F]
    [x*y + z + 1, x + y*z - 2]

    References
    ==========

    .. [1] [MonaganTuncer20]_
    .. [2] [MonaganTuncer16]_

    """
    rng = Random(seed if isinstance(seed, int) else 0)
    degs = dmp_degree_list(f, u)
    r = len(H)
    bits = ((r + 2)*(max(degs) + 1)*(len(degs) + 2)).bit_length() + 1

    P = K(p) if p > _MERSENNE_61 else K(_MERSENNE_61)
    ctx = _Context(u + 1, bits, list(degs), P, K, rng)

    fd = {ctx.pack(e): K(c) for e, c in dmp_to_dict(f, u, K).items()}
    A0, A = A, [K(a) % P for a in A]

    # images of f and of the leading coefficients with x_{j+1}, ... evaluated
    fP = {m: c % P for m, c in fd.items()}
    S = [fP]
    for j in range(u, 1, -1):
        S.insert(0, _sp_eval_var(ctx, S[0], j, A[j - 1]))
    S.insert(0, None)

    LCd = []
    for lc in LC:
        d = {}
        for e, c in dmp_to_dict(lc, u - 1, K).items():
            c = K(c) % P
            if c:
                d[ctx.pack((0,) + e)] = c
        Ls = [d]
        for j in range(u, 1, -1):
            Ls.insert(0, _sp_eval_var(ctx, Ls[0], j, A[j - 1]))
        LCd.append([None] + Ls)

    Hd = []
    for h in H:
        n = len(h) - 1
        Hd.append({ctx.pack((n - k,) + (0,)*u): K(c) % P
                   for k, c in enumerate(h) if c % P})
    hdeg = [len(h) - 1 for h in H]

    try:
        for j in range(1, u + 1):
            Hd = _lift_step(ctx, S[j], Hd, [L[j] for L in LCd], hdeg, j, A)
    except _NoSolution:
        raise ExtraneousFactors
    except _Unlucky:
        # The univariate factors are not coprime modulo P (P divides their
        # resultant). This is very unlikely; use Wang's lifting modulo p.
        from sympy.polys.factortools import dmp_zz_wang_hensel_lifting
        return dmp_zz_wang_hensel_lifting(f, H, LC, A0, p, u, K)

    # The result must be exact: multiply the factors over the integers.
    half = P // 2
    factors = [{m: (c - P if c > half else c) for m, c in h.items()} for h in Hd]
    prod = factors[0]
    for g in factors[1:]:
        prod = _sp_mul(prod, g, None)
    if prod != {m: c for m, c in fd.items() if c}:
        raise ExtraneousFactors

    return [dmp_from_dict({ctx.unpack(m): K(c) for m, c in g.items()}, u, K)
            for g in factors]


def _lift_step(ctx, a, H, L, hdeg, j, A):
    """The `j`-th step of MTSHL (Algorithm 5 of [MonaganTuncer20]_): lift the factors
    ``H`` of ``a`` at `x_j = \\alpha_j` to factors of ``a``, given the true
    leading coefficients ``L``."""
    P = ctx.P
    alpha = A[j - 1]
    r = len(H)

    At = _sp_taylor(ctx, a, j, alpha)
    dj = len(At) - 1

    Hc = []
    for h, lc, d in zip(H, L, hdeg):
        lead = d  # packed x_0**d
        lower = {m: c for m, c in h.items() if (m & ctx.mask) < d}
        Lt = _sp_taylor(ctx, lc, j, alpha)
        T = [_sp_shift(t, lead) for t in Lt]
        T += [{} for _ in range(dj + 1 - len(T))]
        T[0] = _sp_add(T[0], lower, P)
        Hc.append(T)

    solver = _MDPSolver(ctx, [T[0] for T in Hc], j, A)

    pre = [Hc[0]]
    for l in range(1, r):
        pre.append([_sp_mul(pre[l - 1][0], Hc[l][0], P)] + [None]*dj)

    prev = [{m: c for m, c in T[0].items() if (m & ctx.mask) < d}
            for T, d in zip(Hc, hdeg)]
    weak = [dict(s) for s in prev]

    for k in range(1, dj + 1):
        for l in range(1, r):
            pre[l][k] = _prefix_coeff(ctx, pre, Hc, l, k)
        c = _sp_sub(At[k], pre[r - 1][k], P)
        if not c:
            continue
        forms = [prev] if prev == weak else [prev, weak]
        sigma = solver.solve(c, forms)
        changed = False
        for i, s in enumerate(sigma):
            if s:
                Hc[i][k] = _sp_add(Hc[i][k], s, P)
                prev[i] = s
                changed = True
        if changed:
            for l in range(1, r):
                pre[l][k] = _prefix_coeff(ctx, pre, Hc, l, k)

    # Fail early if the factors do not lift; after the last step the exact
    # check in the caller is enough.
    if j < ctx.nvars - 1 and not _random_check(ctx, At, Hc, j):
        raise _NoSolution

    return [_sp_untaylor(ctx, T, j, alpha) for T in Hc]
