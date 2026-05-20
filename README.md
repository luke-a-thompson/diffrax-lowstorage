# diffrax-lowstorage

2N low-storage Runge-Kutta solvers for [Diffrax](https://github.com/patrick-kidger/diffrax), implemented in the Williamson form. Only two state vectors are needed regardless of stage count.

## Solvers

| Class | Stages | Order | Automatic stepsizing |
|-------|--------|-------|----------------------|
| `BWRRK33` | 3 | 3 | No |
| `BWRRK53` | 5 | 3 | Yes |
| `BPRKO52` | 5 | 2 | No |
| `CKRK54` | 5 | 4 | No |
| `EES25` | 3 | 2 | No |
| `EES27` | 4 | 2 | No |
| `SHRK52` | 5 | 2 | No |
| `SHRK64` | 6 | 4 | No |
| `SHRK2N` | 5/6 alternating | 4 | No |
| `BBBRKNL64` | 6 | 4 | No |
| `TSRKC73` | 7 | 3 | No |
| `TSRKC84` | 8 | 4 | No |
| `TSRKF84` | 8 | 4 | No |
| `NDBRK124` | 12 | 4 | No |
| `NDBRK134` | 13 | 4 | No |
| `NDBRK144` | 14 | 4 | No |
| `YRK135` | 13 | 5 (8 for autonomous linear) | No |

## Usage

```python
import diffrax
from diffrax_lowstorage import YRK135

sol = diffrax.diffeqsolve(
    diffrax.ODETerm(lambda t, y, args: -y),
    BWRRK53(),
    t0=0.0, t1=1.0, dt0=0.01, y0=1.0,
)
```

## Commutator-Free Conversion

If you want a commutator-free equivalent, call `to_commutator_free()` on one of the
low-storage solvers. This uses [georax](https://github.com/luke-a-thompson/georax)
to build the matching commutator-free solver.

## Install

```
uv sync
```
