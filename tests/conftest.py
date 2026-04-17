from typing import Any, Literal, NamedTuple

import diffrax
import jax
import jax.numpy as jnp
import lineax
from diffrax import Bosh3, Heun, MultiTerm, ODETerm, ReversibleHeun
from jaxtyping import PyTree

from diffrax_lowstorage import BWRRK33, BWRRK53, CKRK54, EES25, EES27, YRK135

NO_ERROR_SOLVERS = {
    ("bwrrk33", BWRRK33),
    ("yrk135", YRK135),
    ("ckrk53", CKRK54),
    ("ees25", EES25),
    ("ees27", EES27),
    ("reversible_heun", ReversibleHeun),
}

PENULTIMATE_ERROR_SOLVERS = {
    ("bwrrk53", BWRRK53),
}

SOLVERS = NO_ERROR_SOLVERS | PENULTIMATE_ERROR_SOLVERS


# A small MLP as the ODE drift. Realistic per-stage activation retention cost
# (many intermediate tensors per vf_prod eval) — matters for reverse-mode
# memory comparisons, where a cheap `y**3` VF would be dominated by solver-
# independent scratch and wouldn't reveal stage-count scaling.
_BENCH_DIM = 1028
_BENCH_HIDDEN = 768
_BENCH_NUM_LAYERS = 3
_bench_keys = jax.random.split(jax.random.key(42), _BENCH_NUM_LAYERS)
_BENCH_W_IN = jax.random.normal(_bench_keys[0], (_BENCH_HIDDEN, _BENCH_DIM)) / jnp.sqrt(
    _BENCH_DIM
)
_BENCH_W_MID = tuple(
    jax.random.normal(key, (_BENCH_HIDDEN, _BENCH_HIDDEN)) / jnp.sqrt(_BENCH_HIDDEN)
    for key in _bench_keys[1:-1]
)
_BENCH_W_OUT = jax.random.normal(
    _bench_keys[-1], (_BENCH_DIM, _BENCH_HIDDEN)
) / jnp.sqrt(_BENCH_HIDDEN)
_BENCH_PARAMS = (_BENCH_W_IN, _BENCH_W_MID, _BENCH_W_OUT)


def _bench_apply(weights, y):
    w_in, w_mid_layers, w_out = weights
    h = jnp.tanh(w_in @ y)
    for w_mid in w_mid_layers:
        h = jnp.tanh(w_mid @ h)
    return -0.1 * (w_out @ h)


def _bench_vf_global(t, y, args):
    del t, args
    return _bench_apply(_BENCH_PARAMS, y)


def _bench_vf_args(t, y, args):
    del t
    return _bench_apply(args, y)


BENCH_ODE_TERM = diffrax.ODETerm(_bench_vf_global)
BENCH_ODE_TERM_ARGS = diffrax.ODETerm(_bench_vf_args)

_bench_bm = diffrax.VirtualBrownianTree(
    t0=0.0, t1=1.0, tol=1e-3, shape=(_BENCH_DIM,), key=jax.random.key(0)
)
BENCH_SDE_TERM = diffrax.MultiTerm(
    BENCH_ODE_TERM,
    diffrax.ControlTerm(
        lambda t, y, args: lineax.DiagonalLinearOperator(0.1 * y), _bench_bm
    ),
)
BENCH_SDE_TERM_ARGS = diffrax.MultiTerm(
    BENCH_ODE_TERM_ARGS,
    diffrax.ControlTerm(
        lambda t, y, args: lineax.DiagonalLinearOperator(0.1 * y), _bench_bm
    ),
)


class BenchCase(NamedTuple):
    name: str
    term: ODETerm[Any] | MultiTerm[Any]
    y0_shape: tuple[int, ...]
    args: PyTree | None
    grad_target: Literal["y0", "args"]


# `global` keeps the MLP weights baked into the HLO and differentiates wrt `y0`.
# `args` threads the MLP weights through `args` so reverse-mode must retain the
# full activation path needed for parameter gradients.
BENCH_CASES = [
    BenchCase("ode/global", BENCH_ODE_TERM, (_BENCH_DIM,), None, "y0"),
    BenchCase("ode/args", BENCH_ODE_TERM_ARGS, (_BENCH_DIM,), _BENCH_PARAMS, "args"),
    BenchCase("sde/global", BENCH_SDE_TERM, (_BENCH_DIM,), None, "y0"),
    BenchCase("sde/args", BENCH_SDE_TERM_ARGS, (_BENCH_DIM,), _BENCH_PARAMS, "args"),
]

BENCH_SOLVERS = [
    ("bosh3", Bosh3),
    ("heun", Heun),
    *SOLVERS,
]
