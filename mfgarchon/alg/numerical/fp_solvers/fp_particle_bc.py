"""
Dimension-agnostic boundary condition handling for particle-based FP solvers (Issue #635).

This module provides unified functions that work for any dimension:
- Topology detection from boundary conditions
- Boundary enforcement (periodic wrap, reflecting bounce)
- Obstacle boundary handling (implicit domains)
- Segment-aware BC checking

All functions accept both 1D and nD inputs with consistent APIs.

Usage:
    from mfgarchon.alg.numerical.fp_solvers.fp_particle_bc import (
        get_topology_per_dimension,
        apply_boundary_conditions,
        enforce_obstacle_boundary,
        needs_segment_aware_bc,
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mfgarchon.geometry import BoundaryConditions
    from mfgarchon.geometry.implicit import ImplicitDomain

from mfgarchon.geometry.boundary.types import BCType

# Mapping from dimension index to boundary name prefix
_DIM_TO_AXIS_PREFIX = {0: "x", 1: "y", 2: "z"}


def _get_axis_prefix(dim_idx: int) -> str:
    """Get axis prefix for dimension index (x, y, z, dim3, dim4, ...)."""
    if dim_idx < 3:
        return _DIM_TO_AXIS_PREFIX[dim_idx]
    return f"dim{dim_idx}"


# =============================================================================
# Topology Detection (unified)
# =============================================================================


def get_topology_per_dimension(
    boundary_conditions: BoundaryConditions | str | None,
    dimension: int,
) -> list[str]:
    """
    Get grid topology for each dimension from boundary conditions (dimension-agnostic).

    This determines the INDEXING STRATEGY for particles, not the physical BC:
    - "periodic": Space wraps around (particles use modular arithmetic)
    - "bounded": Space has walls (particles reflect at boundaries)

    Note: This is about topology (how space connects), not physics (what values
    are prescribed). For particles, all non-periodic boundaries are treated as
    reflecting walls, regardless of whether the underlying BC is Dirichlet,
    Neumann, Robin, or no-flux.

    Parameters
    ----------
    boundary_conditions : BoundaryConditions or str or None
        Boundary condition specification. Can also be "periodic" string sentinel
        for fully periodic implicit geometry (e.g., Hyperrectangle torus).
    dimension : int
        Number of spatial dimensions

    Returns
    -------
    topologies : list[str]
        Topology per dimension: ["periodic", "bounded", ...]

    Examples
    --------
    >>> from mfgarchon.geometry.boundary import periodic_bc, neumann_bc
    >>> get_topology_per_dimension(periodic_bc(1), 1)  # ["periodic"]
    >>> get_topology_per_dimension(neumann_bc(2), 2)   # ["bounded", "bounded"]
    >>> get_topology_per_dimension("periodic", 2)       # ["periodic", "periodic"]
    """
    # Default to bounded (reflecting walls) for all dimensions
    topologies = ["bounded"] * dimension

    if boundary_conditions is None:
        return topologies

    # Handle "periodic" string sentinel (for implicit geometry with periodic_dims)
    if boundary_conditions == "periodic":
        return ["periodic"] * dimension

    bc = boundary_conditions

    # For uniform BCs, check if periodic - Issue #543: use getattr instead of hasattr
    is_uniform = getattr(bc, "is_uniform", False)
    if is_uniform:
        bc_type = getattr(bc, "type", None)
        if bc_type == "periodic":
            return ["periodic"] * dimension
        return topologies

    # For mixed BCs, check per dimension
    # Periodic requires BOTH min and max to be periodic (topological constraint)
    # Issue #543: use callable() for method existence check
    if not callable(getattr(bc, "get_bc_type_at_boundary", None)):
        return topologies  # Method doesn't exist, use default bounded

    for d in range(dimension):
        axis_prefix = _get_axis_prefix(d)
        min_boundary = f"{axis_prefix}_min"
        max_boundary = f"{axis_prefix}_max"

        try:
            bc_min = bc.get_bc_type_at_boundary(min_boundary)
            bc_max = bc.get_bc_type_at_boundary(max_boundary)

            # Periodic topology requires both boundaries to be periodic
            if bc_min == BCType.PERIODIC and bc_max == BCType.PERIODIC:
                topologies[d] = "periodic"
            # All other cases: bounded topology (reflecting for particles)
        except (KeyError, AttributeError):
            pass  # Keep default "bounded"

    return topologies


def needs_segment_aware_bc(boundary_conditions: BoundaryConditions | str | None) -> bool:
    """
    Check if boundary conditions require segment-aware handling.

    Returns True if:
    - BC has multiple segments with different types
    - BC has absorbing (DIRICHLET) segments that should remove particles

    Returns False for uniform periodic/reflecting BC where fast path can be used.

    Parameters
    ----------
    boundary_conditions : BoundaryConditions or str or None
        Boundary condition specification. Can also be "periodic" string sentinel
        for fully periodic implicit geometry.

    Returns
    -------
    bool
        True if segment-aware handling is needed

    Examples
    --------
    >>> needs_segment_aware_bc(periodic_bc(1))  # False (uniform)
    >>> needs_segment_aware_bc(mixed_bc_with_exit)  # True (has DIRICHLET)
    >>> needs_segment_aware_bc("periodic")  # False (uniform periodic)
    """
    if boundary_conditions is None:
        return False

    # "periodic" string sentinel = uniform periodic, no segment-aware needed
    if boundary_conditions == "periodic":
        return False

    bc = boundary_conditions

    # Check if uniform BC (same type everywhere) - Issue #543: use getattr instead of hasattr
    is_uniform = getattr(bc, "is_uniform", False)
    if is_uniform:
        # Uniform BC - use fast path unless it's DIRICHLET (absorbing)
        segments = getattr(bc, "segments", [])
        return len(segments) > 0 and segments[0].bc_type == BCType.DIRICHLET

    # Mixed BC with segments - need segment-aware handling
    segments = getattr(bc, "segments", [])
    if len(segments) > 1:
        return True

    # Check for DIRICHLET segments (absorbing BC)
    return any(segment.bc_type == BCType.DIRICHLET for segment in segments)


# =============================================================================
# Boundary Enforcement (unified)
# =============================================================================


def apply_boundary_conditions(
    particles: np.ndarray,
    bounds: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    topology: str | list[str] = "bounded",
) -> np.ndarray:
    """
    Apply boundary handling per dimension based on topology (dimension-agnostic).

    This function delegates to the canonical implementations in
    mfgarchon.utils.numerical.particle.boundary (Issue #521).

    At corners, all dimensions are processed simultaneously (not sequentially),
    producing diagonal reflection. This is equivalent to 'average' corner
    strategy for position-based reflection.

    Parameters
    ----------
    particles : np.ndarray
        Particle positions.
        - 1D: shape (num_particles,) or (num_particles, 1)
        - nD: shape (num_particles, dimension)
    bounds : list of tuple or tuple of tuple
        Bounds per dimension [(xmin, xmax), (ymin, ymax), ...]
    topology : str or list[str]
        Grid topology: "periodic" (wrap) or "bounded" (reflect).
        Can be a single string (same for all dims) or per-dimension list.

    Returns
    -------
    particles : np.ndarray
        Updated particle positions (may be modified in place for efficiency)

    Examples
    --------
    >>> particles_1d = np.array([1.5, -0.2, 0.5])  # Some outside [0, 1]
    >>> apply_boundary_conditions(particles_1d, [(0, 1)], "bounded")
    >>> # particles_1d now reflected into [0, 1]

    Note
    ----
    See Issue #521 for corner handling architecture. Position-based reflection
    uses fold reflection which implicitly produces diagonal corner reflection.
    """
    from mfgarchon.geometry.boundary.corner import reflect_positions, wrap_positions

    particles = np.asarray(particles)
    bounds_list = list(bounds)

    # Handle 1D case: convert to 2D for uniform processing
    is_1d_flat = particles.ndim == 1
    if is_1d_flat:
        particles = particles[:, np.newaxis]

    dimension = particles.shape[1] if particles.ndim > 1 else 1

    # Handle per-dimension topologies
    if isinstance(topology, str):
        topologies = [topology] * dimension
    else:
        topologies = list(topology)

    # Check if all dimensions have same topology (common case)
    all_periodic = all(t == "periodic" for t in topologies)
    all_bounded = all(t != "periodic" for t in topologies)

    if all_periodic:
        # Fast path: all periodic
        result = wrap_positions(particles, bounds_list)
    elif all_bounded:
        # Fast path: all bounded (reflecting)
        result = reflect_positions(particles, bounds_list)
    else:
        # Mixed topology: process dimension by dimension
        result = particles.copy()
        for d in range(dimension):
            xmin, xmax = bounds_list[d]
            Lx = xmax - xmin

            if Lx < 1e-14:
                continue  # Skip degenerate dimension

            dim_topology = topologies[d] if d < len(topologies) else "bounded"

            if dim_topology == "periodic":
                # Periodic: wrap around
                result[:, d] = xmin + (result[:, d] - xmin) % Lx
            else:
                # Bounded: fold reflection
                shifted = result[:, d] - xmin
                period = 2 * Lx
                pos_in_period = shifted % period
                in_second_half = pos_in_period > Lx
                pos_in_period[in_second_half] = period - pos_in_period[in_second_half]
                result[:, d] = xmin + pos_in_period

    # Convert back to 1D if input was 1D
    if is_1d_flat:
        return result.ravel()

    return result


# =============================================================================
# Obstacle Boundary Handling (unified)
# =============================================================================


def enforce_obstacle_boundary(
    particles: np.ndarray,
    implicit_domain: ImplicitDomain | None,
) -> np.ndarray:
    """
    Enforce obstacle boundaries via implicit domain geometry (dimension-agnostic).

    If an implicit domain with obstacles is defined, particles that have
    entered obstacle regions (domain.contains() returns False AND inside outer
    bbox) are projected back to the valid domain using domain.project_to_domain().

    Issue #1064: particles past the outer bounding box are NOT projected here —
    they are an outer-boundary concern handled by the caller's
    BoundaryConditions (segment-aware reflect/absorb/wrap). Re-projecting
    outer-boundary violations preempts segment-aware Dirichlet absorption.

    The discriminator:
    - inside bbox + ~contains  →  obstacle interior  →  project here
    - outside bbox + ~contains →  outer-boundary    →  leave for outer BC
    - outside bbox + contains  →  impossible (contains implies inside bbox)
    - inside bbox + contains   →  navigable region  →  no-op

    Parameters
    ----------
    particles : np.ndarray
        Particle positions.
        - 1D: shape (num_particles,) or (num_particles, 1)
        - nD: shape (num_particles, dimension)
    implicit_domain : ImplicitDomain or None
        Domain definition with contains() and project_to_domain() methods

    Returns
    -------
    particles : np.ndarray
        Updated particle positions with obstacle violations corrected

    Examples
    --------
    >>> from mfgarchon.geometry.implicit import DifferenceDomain, Hyperrectangle, Hypersphere
    >>> domain = DifferenceDomain(Hyperrectangle(...), Hypersphere(center=[0.5, 0.5], radius=0.1))
    >>> particles = enforce_obstacle_boundary(particles, domain)
    """
    if implicit_domain is None:
        return particles

    particles = np.asarray(particles)

    # Handle 1D case
    is_1d_flat = particles.ndim == 1
    if is_1d_flat:
        particles = particles[:, np.newaxis]

    # Check which particles are outside the valid domain (inside obstacles
    # OR past the outer boundary).
    inside_valid = implicit_domain.contains(particles)

    # Handle scalar return (single particle case)
    if np.isscalar(inside_valid):
        inside_valid = np.array([inside_valid])

    # Issue #1064: discriminate obstacle-interior violations (we project) from
    # outer-boundary violations (caller's segment-aware BC handles these).
    if not np.all(inside_valid):
        bounds = implicit_domain.get_bounding_box()  # shape (d, 2)
        in_bbox = np.all(
            (particles >= bounds[:, 0]) & (particles <= bounds[:, 1]),
            axis=1,
        )
        # Project only particles inside the outer bbox but in an obstacle.
        in_obstacle = in_bbox & (~inside_valid)
        if np.any(in_obstacle):
            obs_idx = np.where(in_obstacle)[0]
            particles[obs_idx] = implicit_domain.project_to_domain(particles[obs_idx])

    # Convert back to 1D if input was 1D
    if is_1d_flat:
        return particles.ravel()

    return particles


# =============================================================================
# Coordinate/Grid Utilities (unified)
# =============================================================================


def create_coordinate_arrays(
    bounds: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    grid_shape: tuple[int, ...],
) -> list[np.ndarray]:
    """
    Create coordinate arrays from bounds and grid shape (dimension-agnostic).

    Parameters
    ----------
    bounds : list or tuple of (min, max) tuples
        Domain bounds per dimension
    grid_shape : tuple[int, ...]
        Number of grid points per dimension

    Returns
    -------
    coordinates : list[np.ndarray]
        1D coordinate arrays for each dimension

    Examples
    --------
    >>> coords = create_coordinate_arrays([(0, 1), (0, 2)], (11, 21))
    >>> coords[0]  # array([0.0, 0.1, ..., 1.0])
    >>> coords[1]  # array([0.0, 0.1, ..., 2.0])
    """
    coordinates = []
    for d, (bmin, bmax) in enumerate(bounds):
        n_points = grid_shape[d]
        coords = np.linspace(bmin, bmax, n_points)
        coordinates.append(coords)
    return coordinates


def compute_spacings(
    bounds: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    grid_shape: tuple[int, ...],
) -> list[float]:
    """
    Compute grid spacings from bounds and shape (dimension-agnostic).

    Parameters
    ----------
    bounds : list or tuple of (min, max) tuples
        Domain bounds per dimension
    grid_shape : tuple[int, ...]
        Number of grid points per dimension

    Returns
    -------
    spacings : list[float]
        Grid spacing per dimension [dx, dy, ...]

    Examples
    --------
    >>> compute_spacings([(0, 1), (0, 2)], (11, 21))
    [0.1, 0.1]
    """
    spacings = []
    for d, (bmin, bmax) in enumerate(bounds):
        n_points = grid_shape[d]
        if n_points > 1:
            dx = (bmax - bmin) / (n_points - 1)
        else:
            dx = bmax - bmin  # Single point
        spacings.append(dx)
    return spacings


# =============================================================================
# Smoke Tests
# =============================================================================

if __name__ == "__main__":
    """Quick smoke test for development."""
    print("Testing unified fp_particle_bc functions...")

    # Test 1: Topology detection
    # We can't easily test with real BoundaryConditions without imports
    # So test the None case and basic logic
    topo_none = get_topology_per_dimension(None, 2)
    assert topo_none == ["bounded", "bounded"], f"Expected bounded, got {topo_none}"
    print("  get_topology_per_dimension (None BC): OK")

    # Test 2: needs_segment_aware_bc
    result = needs_segment_aware_bc(None)
    assert result is False, "None BC should not need segment-aware"
    print("  needs_segment_aware_bc (None BC): OK")

    # Test 3: Boundary conditions - bounded (reflecting)
    # Test 1D
    particles_1d = np.array([-0.1, 0.5, 1.1, 1.5, 2.5])
    result_1d = apply_boundary_conditions(particles_1d, [(0.0, 1.0)], "bounded")
    assert result_1d.shape == (5,), f"Expected (5,), got {result_1d.shape}"
    assert np.all(result_1d >= 0), f"Negative values: {result_1d}"
    assert np.all(result_1d <= 1), f"Values > 1: {result_1d}"
    print("  apply_boundary_conditions (1D bounded): OK")

    # Test 2D
    particles_2d = np.array(
        [
            [-0.1, 0.5],  # x out of bounds (left)
            [0.5, -0.2],  # y out of bounds (bottom)
            [1.1, 0.5],  # x out of bounds (right)
            [0.5, 1.3],  # y out of bounds (top)
            [0.5, 0.5],  # inside
        ]
    )
    result_2d = apply_boundary_conditions(particles_2d, [(0.0, 1.0), (0.0, 1.0)], "bounded")
    assert result_2d.shape == (5, 2), f"Expected (5, 2), got {result_2d.shape}"
    assert np.all(result_2d >= 0), f"Negative values: {result_2d}"
    assert np.all(result_2d <= 1), f"Values > 1: {result_2d}"
    print("  apply_boundary_conditions (2D bounded): OK")

    # Test 4: Boundary conditions - periodic
    particles_periodic = np.array([-0.1, 0.5, 1.1, 1.5])
    result_periodic = apply_boundary_conditions(particles_periodic, [(0.0, 1.0)], "periodic")
    assert np.all(result_periodic >= 0), f"Negative values: {result_periodic}"
    assert np.all(result_periodic <= 1), f"Values > 1: {result_periodic}"
    # -0.1 should wrap to 0.9, 1.1 should wrap to 0.1
    assert np.abs(result_periodic[0] - 0.9) < 1e-10, f"Expected 0.9, got {result_periodic[0]}"
    assert np.abs(result_periodic[2] - 0.1) < 1e-10, f"Expected 0.1, got {result_periodic[2]}"
    print("  apply_boundary_conditions (1D periodic): OK")

    # Test 5: Mixed topology (x periodic, y bounded)
    particles_mixed = np.array(
        [
            [-0.1, -0.1],  # Both out
            [1.1, 1.1],  # Both out
        ]
    )
    result_mixed = apply_boundary_conditions(particles_mixed, [(0.0, 1.0), (0.0, 1.0)], ["periodic", "bounded"])
    # x should wrap, y should reflect
    assert np.abs(result_mixed[0, 0] - 0.9) < 1e-10, f"x periodic: expected 0.9, got {result_mixed[0, 0]}"
    assert np.abs(result_mixed[0, 1] - 0.1) < 1e-10, f"y bounded: expected 0.1, got {result_mixed[0, 1]}"
    print("  apply_boundary_conditions (mixed periodic/bounded): OK")

    # Test 6: Coordinate array creation
    coords = create_coordinate_arrays([(0.0, 1.0), (0.0, 2.0)], (11, 21))
    assert len(coords) == 2
    assert len(coords[0]) == 11
    assert len(coords[1]) == 21
    assert np.abs(coords[0][-1] - 1.0) < 1e-10
    assert np.abs(coords[1][-1] - 2.0) < 1e-10
    print("  create_coordinate_arrays: OK")

    # Test 7: Spacing computation
    spacings = compute_spacings([(0.0, 1.0), (0.0, 2.0)], (11, 21))
    assert len(spacings) == 2
    assert np.abs(spacings[0] - 0.1) < 1e-10
    assert np.abs(spacings[1] - 0.1) < 1e-10
    print("  compute_spacings: OK")

    # Test 8: Obstacle boundary (None domain - should pass through)
    particles_obs = np.array([[0.5, 0.5], [0.3, 0.3]])
    result_obs = enforce_obstacle_boundary(particles_obs, None)
    assert np.allclose(result_obs, particles_obs)
    print("  enforce_obstacle_boundary (None domain): OK")

    print("\nAll smoke tests passed!")
