import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import pytest

from diffrax_lowstorage import BWRRK53, EES25


class AffineVF(eqx.Module):
    matrix: jax.Array
    bias: jax.Array
    time_weight: jax.Array

    def __call__(self, t, y, args):
        scale, shift = args
        return self.matrix @ y + self.bias * shift + self.time_weight * t + scale * y


def _make_term(kind):
    vf1 = AffineVF(
        matrix=jnp.array([[0.3, -0.2], [0.1, -0.4]]),
        bias=jnp.array([0.5, -0.7]),
        time_weight=jnp.array([0.2, -0.1]),
    )
    if kind == "single":
        return diffrax.ODETerm(vf1)
    if kind == "multi":
        vf2 = AffineVF(
            matrix=jnp.array([[-0.1, 0.4], [0.2, 0.3]]),
            bias=jnp.array([0.1, 0.2]),
            time_weight=jnp.array([-0.05, 0.15]),
        )
        return diffrax.MultiTerm(diffrax.ODETerm(vf1), diffrax.ODETerm(vf2))
    raise ValueError(f"Unknown term kind: {kind}")


def _reference_step(solver, terms, t0, t1, y0, args):
    recurrence = solver.recurrence
    a = jnp.asarray(recurrence.A)
    b = jnp.asarray(recurrence.B)
    c = jnp.asarray(recurrence.C)

    dt = t1 - t0
    control = terms.contr(t0, t1)
    ts = jnp.where(c[1:] == 1.0, t1, t0 + c[1:] * dt)

    tmp = terms.vf_prod(t0, y0, args, control)
    y = jtu.tree_map(lambda yi, tmpi: yi + b[0] * tmpi, y0, tmp)

    y_pen = None
    for i, (a_i, b_i, t_stage) in enumerate(zip(a, b[1:], ts), start=1):
        if recurrence.penultimate_stage_error and i == len(b) - 1:
            y_pen = y
        k = terms.vf_prod(t_stage, y, args, control)
        tmp = jtu.tree_map(lambda tmpi, ki: a_i * tmpi + ki, tmp, k)
        y = jtu.tree_map(lambda yi, tmpi: yi + b_i * tmpi, y, tmp)

    y_error = (
        jtu.tree_map(lambda y1i, ypeni: y1i - ypeni, y, y_pen)
        if recurrence.penultimate_stage_error
        else None
    )
    dense_info = dict(y0=y0, y1=y)
    return y, y_error, dense_info, None, diffrax.RESULTS.successful


def _loss_from_step(step_fn, packed):
    terms, t0, t1, y0, args = packed
    y1, y_error, dense_info, _, _ = step_fn(terms, t0, t1, y0, args)
    loss = (
        jnp.sum(y1) + 0.5 * jnp.sum(dense_info["y0"]) - 0.25 * jnp.sum(dense_info["y1"])
    )
    if y_error is not None:
        loss = loss + 0.3 * jnp.sum(y_error)
    return loss


def _assert_tree_allclose(got, expected, *, atol=1e-7, rtol=1e-6):
    is_none = lambda x: x is None
    got_leaves = jtu.tree_leaves(got, is_leaf=is_none)
    exp_leaves = jtu.tree_leaves(expected, is_leaf=is_none)
    assert jtu.tree_structure(got, is_leaf=is_none) == jtu.tree_structure(
        expected, is_leaf=is_none
    )
    for got_leaf, exp_leaf in zip(got_leaves, exp_leaves, strict=True):
        if got_leaf is None or exp_leaf is None:
            assert got_leaf is None and exp_leaf is None
        else:
            assert jnp.allclose(got_leaf, exp_leaf, atol=atol, rtol=rtol)


@pytest.mark.parametrize("solver_cls", [EES25, BWRRK53])
@pytest.mark.parametrize("term_kind", ["single", "multi"])
def test_lowstorage_step_custom_vjp_matches_reference_gradients(solver_cls, term_kind):
    solver = solver_cls()
    packed = (
        _make_term(term_kind),
        jnp.array(0.1),
        jnp.array(0.25),
        jnp.array([0.2, -0.3]),
        (jnp.array(0.7), jnp.array(-0.4)),
    )

    custom_grad = eqx.filter_grad(
        lambda packed_: _loss_from_step(
            lambda terms, t0, t1, y0, args: solver.step(
                terms, t0, t1, y0, args, None, False
            ),
            packed_,
        )
    )(packed)
    reference_grad = eqx.filter_grad(
        lambda packed_: _loss_from_step(
            lambda terms, t0, t1, y0, args: _reference_step(
                solver, terms, t0, t1, y0, args
            ),
            packed_,
        )
    )(packed)

    _assert_tree_allclose(custom_grad, reference_grad)


def test_reversible_adjoint_multiterm_args_grad_matches_checkpointed():
    solver = EES25()
    term = _make_term("multi")
    y0 = jnp.array([0.1, -0.2])
    args = (jnp.array(0.3), jnp.array(-0.1))
    saveat = diffrax.SaveAt(t1=True)

    def solve_loss(adjoint, solve_args):
        out = diffrax.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=0.3,
            dt0=0.05,
            y0=y0,
            args=solve_args,
            saveat=saveat,
            adjoint=adjoint,
            max_steps=16,
            throw=True,
        )
        assert out.ys is not None
        return jnp.sum(out.ys)

    reversible_grad = eqx.filter_grad(
        lambda solve_args: solve_loss(diffrax.ReversibleAdjoint(), solve_args)
    )(args)
    checkpointed_grad = eqx.filter_grad(
        lambda solve_args: solve_loss(
            diffrax.RecursiveCheckpointAdjoint(checkpoints=4), solve_args
        )
    )(args)

    _assert_tree_allclose(reversible_grad, checkpointed_grad, atol=1e-6, rtol=1e-5)
