import diffrax
import numpy as np
import pytest
from conftest import SOLVERS

from diffrax_lowstorage import EES29, LowStorageRecurrence, LowStorageSolver


@pytest.mark.parametrize(("solver_name", "solver_cls"), SOLVERS)
def test_builtin_tableau_round_trip(solver_name, solver_cls):
    del solver_name
    if not issubclass(solver_cls, LowStorageSolver):
        pytest.skip("Alternating solver has no single tableau.")
    recurrence = solver_cls.recurrence
    reconstructed = LowStorageRecurrence.from_butcher(recurrence.to_butcher())
    np.testing.assert_allclose(reconstructed.A, recurrence.A, atol=1e-12)
    np.testing.assert_allclose(reconstructed.B, recurrence.B, atol=1e-12)
    np.testing.assert_allclose(reconstructed.C, recurrence.C, atol=1e-12)
    assert reconstructed.penultimate_stage_error == recurrence.penultimate_stage_error


def test_ees29_matches_supplied_tableau():
    s5 = np.sqrt(5.0)
    tableau = EES29.recurrence.to_butcher()
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
