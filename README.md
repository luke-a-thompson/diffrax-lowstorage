<p align="center">
  <picture>
    <source srcset="docs/_static/diffrax-lowstorage.dark.svg" media="(prefers-color-scheme: dark)">
    <source srcset="docs/_static/diffrax-lowstorage.light.svg" media="(prefers-color-scheme: light)">
    <img src="docs/_static/diffrax-lowstorage.light.svg" width="520" alt="Logo">
  </picture>
</p>

<h2 align='center'>Memory Efficient 2N Integrators for Diffrax.</h2>

Diffrax-lowstorage provides memory-efficient ODE integrators for [Diffrax](https://github.com/patrick-kidger/diffrax). Its Williamson-form methods use two evolving state vectors per forward step, making them useful for large PDE discretisations and batches of ODEs.

Differentiation uses standard JAX autodiff with checkpoints around each step and stage. Checkpoints recompute intermediates during the backward pass, and both forward-mode differentiation and gradients through vector-field closures are supported. Reversing a step still requires storage proportional to its stage count, plus vector-field workspace and parameter gradients. Storage across time steps depends on the Diffrax adjoint. The EES solvers support the fork's `ReversibleAdjoint`, which avoids storing the solution trajectory, but reconstructs it approximately rather than exactly.

## Solvers

| Class | Stages | Order | Automatic stepsizing |
|-------|--------|-------|----------------------|
| `BWRRK33` | 3 | 3 | No |
| `BWRRK53` | 5 | 3 | Yes |
| `BPRKO52` | 5 | 2 | No |
| `CKRK54` | 5 | 4 | No |
| `EES25` | 3 | 2 | No |
| `EES27` | 4 | 2 | No |
| `EES29` | 5 | 2 | No |
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

## Convergence experiments

The deterministic examples compare `EES25`, `EES27`, and `EES29` using Georax's convergence chart style: crimson crosses for measured errors and blue lines for the expected rates. Grey crosses mark errors at floating-point roundoff and are excluded from slope fits. Each run saves PNG and PDF charts plus JSON results in `docs/examples/outputs`. Install the development dependencies to run these examples.

```bash
python docs/examples/solver_convergence.py
python docs/examples/solver_convergence_decimal.py
```

The ODE experiment uses the exact solution of `y' = -y**3` and measures reconstruction through `backward_step` in float64. The separate Decimal experiment evaluates the same low-storage recurrences with 80-digit coefficients, solution values, and errors, allowing the ninth-order round-trip convergence of `EES29` to remain visible below float64 roundoff. Set `--precision` to choose the number of decimal digits.

## Commutator-Free Conversion

If you want a commutator-free equivalent, call `to_commutator_free()` on one of the low-storage solvers. This requires `georax` to build the matching commutator-free solver.

## Install

From a checkout, run `pip install .` or `uv sync --extra dev` for development. Both install the pinned [Diffrax fork](https://github.com/sammccallum/diffrax/tree/aeb1335b5a6278e85a270231e0f97b8db4453ae6) needed for `ReversibleAdjoint`. Python 3.11 and later are supported.
