=======
History
=======

Unreleased
----------

* Define ``pol_sigma`` as a fraction of the sampled unit-cell period (default
  ``1/12``), so Pol tangent-field smoothing does not shrink when a grid is
  refined.
* Diffuse doubled interface orientations in the Pol tangent field so opposite
  boundary normals reinforce instead of cancelling on refinement.
* Use the reference central-gradient, single-diffusion path by default so
  curved boundaries are not re-pinned to raster staircase normals.

0.1.2 (2020-11-01)
------------------

* Add example for hexagonal lattice

0.1.1 (2020-05-18)
------------------

* Fix license
  
0.1 (2020-05-12)
------------------

* First release on PyPI.
