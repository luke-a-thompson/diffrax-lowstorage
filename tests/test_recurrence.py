import diffrax
import numpy as np
import pytest
from conftest import SOLVERS

from diffrax_lowstorage import LowStorageRecurrence, LowStorageSolver


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
