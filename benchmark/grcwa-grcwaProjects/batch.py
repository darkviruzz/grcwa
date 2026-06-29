"""Batch RCWA solving utilities."""
import numpy as np
from multiprocessing import Pool
from .rcwa import obj


def _solve_one(params, obj_args, obj_kwargs, structure_fn):
    """Worker helper to solve a single (freq, theta, phi) triple."""
    freq, theta, phi = params
    sim = obj(*obj_args, freq, theta, phi, **(obj_kwargs or {}))
    structure_fn(sim)
    return sim.RT_Solve()


def batch_solve(freq, theta, phi, *, nproc=1, obj_args=(), obj_kwargs=None,
                structure_fn=lambda sim: None):
    """Run RCWA solves for arrays of ``freq``, ``theta`` and ``phi`` in parallel.

    Parameters
    ----------
    freq, theta, phi : array_like or float
        Frequencies and angles. Scalars are broadcast to match the longest
        input.
    nproc : int, optional
        Number of worker processes to launch. Defaults to ``1``.
    obj_args : tuple, optional
        Positional arguments passed to :class:`grcwa.obj` *before* ``freq``,
        ``theta`` and ``phi``.
    obj_kwargs : dict, optional
        Keyword arguments forwarded to :class:`grcwa.obj`.
    structure_fn : callable, optional
        Callback invoked with each freshly created :class:`grcwa.obj` to set up
        layers, excitation, etc.

    Returns
    -------
    numpy.ndarray
        Array with shape ``(N, ...)`` where ``N`` is the number of parameter
        triples.
    """
    # --- symmetric broadcasting for freq, theta, phi ---
    f = np.atleast_1d(freq)
    t = np.atleast_1d(theta)
    p = np.atleast_1d(phi)
    try:
        f, t, p = np.broadcast_arrays(f, t, p)  # all three now share the same shape
    except ValueError as e:
        raise ValueError(
            f"freq, theta, and phi must be broadcastable to a common shape; "
            f"got shapes {np.shape(f)}, {np.shape(t)}, {np.shape(p)}"
        ) from e

    params = list(zip(f, t, p))
    tasks = [(param, obj_args, obj_kwargs, structure_fn) for param in params]

    with Pool(processes=nproc) as pool:
        results = pool.starmap(_solve_one, tasks)

    return np.array(results)
