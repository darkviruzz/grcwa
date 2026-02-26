Optimization task definitions (1D mode + S-matrix caching)
===========================================================

This note turns the recent performance discussion into implementation
tasks with explicit scope, physics constraints, and compatibility rules.


Task A: Native 1D mode inferred from ``Nx``/``Ny``
--------------------------------------------------

Goal
~~~~

Accelerate simulations of structures that are periodic in only one lateral
direction (e.g., binary gratings), while keeping the user-facing API
compatible with existing code.

Current issue
~~~~~~~~~~~~~

The current solver stack is built around 2D unit-cell discretization and a 2D
harmonic index set ``G[:, 0], G[:, 1]``. This causes avoidable overhead when a
layer is effectively 1D (for example ``Ny == 1``), because the code still
executes a full 2D path for Fourier transforms, convolution indexing, and
matrix assembly.

Why this is necessary for performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In 1D gratings, the physically relevant Fourier basis is one index (``m``)
instead of two indices (``m, n``). Keeping a 2D representation for a 1D
problem inflates matrix dimensions and therefore CPU time and memory for
eigensolves and S-matrix operations.

Proposed optimization
~~~~~~~~~~~~~~~~~~~~~

Infer 1D mode directly from grid shape, without introducing a new required
flag:

* If ``Nx > 1`` and ``Ny == 1``: use x-periodic 1D path.
* If ``Ny > 1`` and ``Nx == 1``: use y-periodic 1D path.
* If ``Nx > 1`` and ``Ny > 1``: keep existing 2D path.
* If ``Nx == 1`` and ``Ny == 1``: treat as degenerate/uniform and use current
  behavior or raise a clear error.

Important architectural rule:

* Do **not** overload ``Gmethod`` to represent dimensionality. ``Gmethod``
  should keep its current meaning (2D truncation strategy). 1D dimensional
  inference should be orthogonal and handled before harmonic selection.

Implementation outline
~~~~~~~~~~~~~~~~~~~~~~

1. Add an internal per-layer dimensional descriptor (e.g., ``periodic_dim``)
   derived from ``Nx, Ny`` in ``Add_LayerGrid``.
2. Introduce 1D FFT/convolution helpers for patterned layers with one singleton
   axis.
3. Add a 1D harmonic generator path that produces a single integer harmonic
   list and map it to the existing matrix assembly interface with minimal
   branching.
4. Keep existing 2D APIs and default behavior unchanged for ``Nx > 1`` and
   ``Ny > 1``.

Backward compatibility and risks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Existing 2D scripts remain unchanged.
* Existing 1D-as-2D scripts (``Ny == 1`` or ``Nx == 1``) should produce
  numerically consistent results, with lower runtime and memory use.
* Risk: hidden assumptions that both harmonic axes exist in downstream code.
  Mitigation: isolate 1D branching at Fourier/convolution and harmonic-index
  construction boundaries; keep the rest of the solver working on consistent
  shaped arrays.

Physical validity constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Valid only when one in-plane direction is truly invariant.
* Must preserve Maxwell boundary conditions and the same sign/branch
  conventions used in the current eigensystem solver.
* Polarization coupling behavior in the 1D path must remain consistent with
  the existing field conventions.

Acceptance criteria
~~~~~~~~~~~~~~~~~~~

* Unit tests showing that 1D inferred path (``Ny == 1`` or ``Nx == 1``)
  matches the legacy 2D emulation within tolerance.
* Benchmarks demonstrating lower runtime/memory for representative 1D grating
  problems.


Task B: Cache matrix inverses/factorizations in repeated S-matrix use
----------------------------------------------------------------------

Goal
~~~~

Reduce repeated linear-algebra cost in S-matrix assembly during consecutive
solves where layer eigensystems are unchanged.

Current issue
~~~~~~~~~~~~~

``GetSMatrix`` repeatedly computes expensive inverses inside layer loops,
including ``inv(phi_l)`` and ``inv(kp_l @ phi_l)``. For repeated solves on the
same solved state, these costs are paid again and again.

Why this is necessary for performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Matrix inversion/factorization dominates runtime for moderate/large harmonic
bases. Reusing already computed inverse/factorization objects can substantially
reduce per-solve overhead in repeated post-processing workflows.

Proposed optimization
~~~~~~~~~~~~~~~~~~~~~

Add cacheable per-layer linear algebra artifacts:

* ``phi_inv`` (or factorization equivalent) for each solved layer.
* factorization/inverse of ``kp @ phi`` used in ``GetSMatrix`` transitions.

Use cached artifacts when valid; recompute on invalidation.

Strict invalidation rules (must-have)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cache is valid only if all dependencies are unchanged:

* frequency ``omega``
* incident-angle-dependent wavevectors ``kx, ky``
* reciprocal lattice / harmonic set ``G``
* patterned-layer Fourier material matrices (from epsilon grids)
* layer thickness-dependent phase factors where applicable

Any change to these must invalidate affected layer caches before S-matrix
assembly.

Backward compatibility and risks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Public API should remain unchanged; caching is internal.
* Results must be numerically identical to non-cached computation (within
  floating-point tolerance).
* Main risk is stale-cache reuse. Mitigation: conservative invalidation keyed
  to the dependency set above.

Physical validity constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The eigenvectors ``phi`` are **not** universally invariant; they vary when the
operator changes (e.g., through ``omega``, ``kx``, ``ky``, or epsilon Fourier
matrices). Therefore caching is physically valid only with strict dependency
tracking and invalidation.

Acceptance criteria
~~~~~~~~~~~~~~~~~~~

* Regression tests proving cached and non-cached S-matrix paths agree.
* Tests that modify one dependency at a time and verify cache invalidation.
* Performance checks showing reduced repeated-solve wall time for unchanged
  layer states.


Task C: Incremental implementation plan for 1D support
------------------------------------------------------

Goal
~~~~

Define a practical, low-risk implementation sequence to deliver inferred 1D
support quickly while preserving numerical correctness and backward
compatibility.

Why this task is separate from Task A
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Task A defines the target architecture. This task defines *how* to implement it
in shippable stages, so performance gains are achieved early and regressions are
contained.

Proposed phased plan
~~~~~~~~~~~~~~~~~~~~

Phase 1 (detection + plumbing)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Add internal detection logic from ``Nx, Ny``:

  - ``Nx > 1, Ny == 1`` -> ``periodic_dim='x'``
  - ``Ny > 1, Nx == 1`` -> ``periodic_dim='y'``
  - otherwise -> existing 2D path

* Store per-layer dimension metadata and surface it only internally (no public
  API change).
* Add explicit validation/error messages for ambiguous/degenerate grids.

Phase 2 (1D Fourier path)
^^^^^^^^^^^^^^^^^^^^^^^^^

* Implement 1D FFT/convolution helpers and dispatch to them for inferred 1D
  layers.
* Keep existing 2D helpers untouched for non-1D layers.
* Ensure returned tensor/matrix shapes remain compatible with downstream
  eigensystem code to minimize refactoring risk.

Phase 3 (1D harmonic basis path)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Add a dedicated 1D harmonic generator for a single index basis.
* Keep ``Gmethod`` semantics unchanged for 2D truncation only.
* Provide internal adapters so field reconstruction and post-processing APIs
  keep working with both 1D and 2D solved states.

Phase 4 (verification + docs + benchmark)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Add regression tests comparing inferred 1D path against current 1D-as-2D
  emulation on representative gratings.
* Add benchmark cases demonstrating runtime and memory improvements.
* Document the inference rules and edge-case behavior in user docs.

Performance expectations
~~~~~~~~~~~~~~~~~~~~~~~~

* Early wins in Phase 2 from avoiding unnecessary 2D FFT/convolution work for
  singleton-axis grids.
* Largest gains in Phase 3 once harmonic basis dimensionality is reduced from
  2D to 1D for truly 1D periodic structures.

Backward compatibility
~~~~~~~~~~~~~~~~~~~~~~

* No required new flags; existing scripts should continue to run unchanged.
* Users currently setting ``Ny == 1`` or ``Nx == 1`` receive the optimized path
  automatically.
* 2D behavior remains the default whenever both dimensions exceed 1.

Physical validity constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Use 1D path only when one in-plane direction is invariant.
* Ensure matching boundary-condition treatment and branch/sign conventions
  relative to existing 2D solver.
* Include tests for both normal and oblique incidence to verify consistency of
  field/polarization behavior.

Acceptance criteria
~~~~~~~~~~~~~~~~~~~

* Stage-gated PRs where each phase is independently testable and reversible.
* Numerical agreement with baseline 1D-as-2D results within tolerance.
* Demonstrated performance gains on at least one binary-grating benchmark.
