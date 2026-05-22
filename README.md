<p align="center">
  <picture>
    <source srcset="https://raw.githubusercontent.com/luke-a-thompson/diffrax-lowstorage/main/docs/_static/diffrax-lowstorage.dark.svg" media="(prefers-color-scheme: dark)">
    <source srcset="https://raw.githubusercontent.com/luke-a-thompson/diffrax-lowstorage/main/docs/_static/diffrax-lowstorage.light.svg" media="(prefers-color-scheme: light)">
    <img src="https://raw.githubusercontent.com/luke-a-thompson/diffrax-lowstorage/main/docs/_static/diffrax-lowstorage.light.svg" width="350" alt="Logo">
  </picture>
</p>

<h2 align='center'>2N ODE integrators for Diffrax.</h2>

Diffrax-lowstorage provides memory-efficient ODE integrators for [Diffrax](https://github.com/patrick-kidger/diffrax). Its Williamson-form methods use custom forward and backward passes, so only two state vectors are needed regardless of stage count -- ideal for large PDE discretisations, huge GPU batches of ODEs, and neural ODE/SDE workloads with EES methods for $\mathcal{O}(1)$ backpropagation.

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
from diffrax_lowstorage import BWRRK53

sol = diffrax.diffeqsolve(
    diffrax.ODETerm(lambda t, y, args: -y),
    BWRRK53(),
    t0=0.0, t1=1.0, dt0=0.01, y0=1.0,
)
```

## Commutator-Free Conversion

If you want a commutator-free equivalent, call `to_commutator_free()` on one of the
low-storage solvers. This uses [diffrax-lowstorage](https://github.com/luke-a-thompson/diffrax-lowstorage)
to build the matching commutator-free solver.

## Install

```
uv sync
```
