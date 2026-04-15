from diffrax import Bosh3, Heun

from diffrax_lowstorage import BWRRK33, BWRRK53, CKRK54, EES25, EES27, YRK135

NO_ERROR_SOLVERS = {
    ("bwrrk33", BWRRK33),
    ("yrk135", YRK135),
    ("ckrk53", CKRK54),
    ("ees25", EES25),
    ("ees27", EES27),
}

PENULTIMATE_ERROR_SOLVERS = {
    ("bwrrk53", BWRRK53),
}

SOLVERS = NO_ERROR_SOLVERS | PENULTIMATE_ERROR_SOLVERS

BENCH_SOLVERS = [
    ("bosh3", Bosh3),
    ("heun", Heun),
    *SOLVERS,
]
