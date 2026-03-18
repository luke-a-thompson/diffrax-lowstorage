from typing import ClassVar, override

import numpy as np

from .base import LowStorageRecurrence, LowStorageSolver

_tableau = LowStorageRecurrence(
    # A2..A13 (A1 is implicitly 0)
    A=np.array(
        [
            -0.33672143119427413,
            -1.2018205782908164,
            -2.6261919625495068,
            -1.5418507843260567,
            -0.2845614242371758,
            -0.1700096844304301,
            -1.0839412680446804,
            -11.61787957751822,
            -4.5205208057464192,
            -35.86177355832474,
            -0.000021340899996007288,
            -0.066311516687861348,
        ]
    ),
    # B1..B13
    B=np.array(
        [
            0.069632640247059393,
            0.088918462778092020,
            1.0461490123426779,
            0.42761794305080487,
            0.20975844551667144,
            -0.11457151862012136,
            -0.01392019988507068,
            4.0330655626956709,
            0.35106846752457162,
            -0.16066651367556576,
            -0.0058633163225038929,
            0.077296133865151863,
            0.054301254676908338,
        ]
    ),
    # Stage times c1..c13
    C=np.array(
        [
            0.0,
            0.069632640247059393,
            0.12861035097891748,
            0.34083022189561149,
            0.54063706308495402,
            0.59927749518613931,
            0.49382042519248519,
            0.48207852767699775,
            0.82762865209834452,
            0.82923953914857933,
            0.67190565554748019,
            0.87194975193167848,
            0.94930216564503562,
        ]
    ),
)


class YRK135(LowStorageSolver):
    """13-stage 2N low-storage Runge-Kutta method by Yan (2017).

    Achieves 8th-order accuracy for autonomous linear differential equations,
    and 5th-order accuracy for time-variant or nonlinear differential equations.

    Reference:
        Yan, Yun-an. "Low-Storage Runge-Kutta Method for Simulating Time-Dependent
        Quantum Dynamics." Chinese Journal of Chemical Physics 30, no. 3 (2017):
        277–86. https://doi.org/10.1063/1674-0068/30/cjcp1703025. Table II.
    """

    tableau: ClassVar[LowStorageRecurrence] = _tableau

    @override
    def order(self, terms):
        del terms
        return 5
