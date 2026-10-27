import diffrax
import numpy as np
import pytest
from conftest import SOLVERS

from diffrax_lowstorage import CosineEES, LowStorageRecurrence, LowStorageSolver


@pytest.mark.parametrize(("solver_name", "solver_cls"), SOLVERS)
def test_builtin_tableau_round_trip(solver_name, solver_cls):
    del solver_name
    solver = solver_cls()
    if not isinstance(solver, LowStorageSolver):
        pytest.skip("Alternating solver has no single tableau.")
    recurrence = solver.recurrence
    reconstructed = LowStorageRecurrence.from_butcher(recurrence.to_butcher())
    np.testing.assert_allclose(reconstructed.A, recurrence.A, atol=1e-12)
    np.testing.assert_allclose(reconstructed.B, recurrence.B, atol=1e-12)
    np.testing.assert_allclose(reconstructed.C, recurrence.C, atol=1e-12)
    assert reconstructed.penultimate_stage_error == recurrence.penultimate_stage_error


def test_cosine_ees9_matches_supplied_tableau():
    s5 = np.sqrt(5.0)
    tableau = CosineEES(9).recurrence.to_butcher()
    expected_rows = (
        np.array([(3 - s5) / 4]),
        np.array([0.0, 1 / 2]),
        np.array([(3 - s5) / 4, 0.0, (s5 - 1) / 2]),
        np.array([0.0, 1 / 2, 0.0, 1 / 2]),
    )
    assert tableau.num_stages == 5
    for got, expected in zip(tableau.a_lower, expected_rows, strict=True):
        np.testing.assert_allclose(got, expected, atol=1e-14)
    np.testing.assert_allclose(
        tableau.b_sol,
        [(3 - s5) / 8, 1 / 4, (s5 - 1) / 4, 1 / 4, (3 - s5) / 8],
        atol=1e-14,
    )
    assert tableau.c1 == 0.0
    np.testing.assert_allclose(
        tableau.c, [(3 - s5) / 4, 1 / 2, (1 + s5) / 4, 1.0], atol=1e-14
    )
    np.testing.assert_array_equal(tableau.b_error, np.zeros(5))


@pytest.mark.parametrize("order", [3, 5, 7, 9, 11, 31])
def test_cosine_ees_matches_cosine_tableau(order):
    solver = CosineEES(reversible_order=order)
    recurrence = solver.recurrence
    tableau = recurrence.to_butcher()
    s = (order + 1) // 2
    d = -np.diff(np.cos(np.arange(s + 1) * np.pi / s))
    assert tableau.num_stages == s
    for i, row in enumerate(tableau.a_lower, start=1):
        expected = np.where((i - np.arange(i)) % 2, d[:i], 0.0)
        np.testing.assert_allclose(row, expected, atol=1e-14)
        np.testing.assert_allclose(recurrence.C[i], expected.sum(), atol=1e-14)
    np.testing.assert_allclose(tableau.b_sol, d / 2, atol=1e-14)
    assert recurrence.C[0] == 0.0
    assert recurrence.C[-1] == 1.0
    assert solver.order(None) == 2
    assert solver.strong_order(None) == 0.5
    assert solver.antisymmetric_order(None) == order
    assert solver.error_order(None) is None


@pytest.mark.parametrize("order", [-3, 0, 1, 2, 4, 10, 3.0, 3.5, True, "9", None])
def test_cosine_ees_rejects_invalid_order(order):
    with pytest.raises(ValueError, match="odd integer >= 3"):
        CosineEES(order)


@pytest.mark.parametrize(
    "recurrence",
    [
        LowStorageRecurrence(
            A=np.array([-1.0, 0.0]),
            B=np.array([1.0, 0.5, 0.0]),
            C=np.array([0.0, 1.0, 1.0]),
        ),
        LowStorageRecurrence(
            A=np.array([-0.5, -0.5]),
            B=np.array([0.25, 0.0, 1.0]),
            C=np.array([0.0, 0.25, 0.25]),
        ),
    ],
)
def test_zero_weight_tableau_round_trip(recurrence):
    tableau = recurrence.to_butcher()
    with np.errstate(divide="raise", invalid="raise"):
        converted = LowStorageRecurrence.from_butcher(tableau)
    assert not converted.penultimate_stage_error
    reconstructed = converted.to_butcher()
    for got, expected in zip(reconstructed.a_lower, tableau.a_lower):
        np.testing.assert_allclose(got, expected)
    np.testing.assert_allclose(reconstructed.b_sol, tableau.b_sol)
    np.testing.assert_allclose(reconstructed.b_error, tableau.b_error)


def test_nonrepresentable_tableau_raises_value_error():
    with pytest.raises(ValueError, match="not representable"):
        LowStorageRecurrence.from_butcher(diffrax.Bosh3.tableau)


def test_implicit_tableau_is_rejected():
    with pytest.raises(ValueError, match="must be explicit"):
        LowStorageRecurrence.from_butcher(diffrax.Kvaerno3.tableau)
