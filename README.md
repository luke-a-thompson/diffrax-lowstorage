# diffrax-lowstorage

2N low-storage Runge-Kutta solvers for [Diffrax](https://github.com/patrick-kidger/diffrax), implemented in the Williamson form. Only two state vectors are needed regardless of stage count.

## Solvers

| Class | Stages | Order | Notes |
|-------|--------|-------|-------|
| `BWRRK33` | 3 | 3 | |
| `BWRRK53` | 5 | 3 | Embedded (3,2) error via penultimate stage |
| `YRK135` | 13 | 5 (8 for autonomous linear) | Yan 2017 |

## Usage

```python
import diffrax
from diffrax_lowstorage import YRK135

sol = diffrax.diffeqsolve(
    diffrax.ODETerm(lambda t, y, args: -y),
    YRK135(),
    t0=0.0, t1=1.0, dt0=0.01, y0=1.0,
)
```

## Install

```
pip install diffrax-lowstorage
```
