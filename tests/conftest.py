from diffrax import Bosh3, Heun

from diffrax_lowstorage import BWRRK33, BWRRK53, CKRK54, YRK135

SOLVERS = [
    ("bwrrk33", BWRRK33),
    ("bwrrk53", BWRRK53),
    ("yrk135", YRK135),
    ("ckrk53", CKRK54),
]

NO_ERROR_SOLVERS = {("bwrrk33", BWRRK33), ("yrk135", YRK135), ("ckrk53", CKRK54)}
PENULTIMATE_ERROR_SOLVERS = {("bwrrk53", BWRRK53)}

BENCH_SOLVERS = [
    ("bosh3", Bosh3),
    ("heun", Heun),
] + SOLVERS
