"""
Constructive Solid Geometry (CSG) Operations

CSG operations allow building complex domains from simple primitives:
- Union: D₁ ∪ D₂ ∪ ... ∪ Dₙ
- Intersection: D₁ ∩ D₂ ∩ ... ∩ Dₙ
- Complement: ℝ^d \\ D
- Difference: D₁ \\ D₂ = D₁ ∩ (ℝ^d \\ D₂)

Key applications:
- Obstacles: domain \\\\ obstacle
- Mazes: base ∩ (complement of walls)
- Complex geometry: unions/intersections of primitives

For signed distance functions:
- Union: φ_{D₁∪D₂}(x) = min(φ_{D₁}(x), φ_{D₂}(x))
- Intersection: φ_{D₁∩D₂}(x) = max(φ_{D₁}(x), φ_{D₂}(x))
- Complement: φ_{ℝ^d\\D}(x) = -φ_D(x)

References:
- Ricci (1973): An Constructive Geometry for Computer Graphics
- TECHNICAL_REFERENCE_HIGH_DIMENSIONAL_MFG.md Section 4.3
"""

import numpy as np
from numpy.typing import NDArray

from .implicit_domain import ImplicitDomain


class UnionDomain(ImplicitDomain):
    """
    Union of multiple domains: D = D₁ ∪ D₂ ∪ ... ∪ Dₙ

    A point is inside if it's inside ANY of the constituent domains.

    Signed distance function:
        φ_union(x) = min(φ₁(x), φ₂(x), ..., φₙ(x))

    Example:
        >>> # Two overlapping circles
        >>> circle1 = Hypersphere(center=[0, 0], radius=1.0)
        >>> circle2 = Hypersphere(center=[1, 0], radius=1.0)
        >>> union = UnionDomain([circle1, circle2])  # Peanut shape
        >>> union.contains([0.5, 0])  # True (in overlap)
    """

    def __init__(self, domains: list[ImplicitDomain]) -> None:
        """
        Initialize union of domains.

        Args:
            domains: List of ImplicitDomain objects

        Raises:
            ValueError: If domains list is empty or dimensions don't match
        """
        if not domains:
            raise ValueError("domains list cannot be empty")

        self.domains = domains
        self._dimension = domains[0].dimension

        # Verify all domains have same dimension
        if not all(d.dimension == self._dimension for d in domains):
            raise ValueError(f"All domains must have same dimension, got dimensions: {[d.dimension for d in domains]}")

        # Issue #1041: expose `.bounds` so callers like
        # FPParticleSolver._get_grid_params don't silently fall back to the unit
        # hypercube. Mirrors the Hyperrectangle.bounds attribute (instance attr
        # form, not @property — Hyperrectangle subclasses store as instance attr,
        # which would conflict with a base-class @property without a setter).
        self.bounds: NDArray[np.float64] = self.get_bounding_box()

    @property
    def dimension(self) -> int:
        """Spatial dimension."""
        return self._dimension

    def signed_distance(self, x: NDArray[np.float64]) -> float | NDArray[np.float64]:
        """
        Compute signed distance to union.

        φ_union(x) = min_i φ_i(x)

        A point is inside union if it's inside any constituent domain.

        Args:
            x: Point(s) - shape (d,) or (N, d)

        Returns:
            Signed distance(s) - scalar float or array of shape (N,)
        """
        x = np.asarray(x, dtype=float)
        is_single = x.ndim == 1

        # Compute SDF for each domain
        distances = np.array([domain.signed_distance(x) for domain in self.domains])

        # Union: minimum distance
        sd = np.min(distances, axis=0)

        return float(sd) if is_single and np.isscalar(sd) else sd

    def get_bounding_box(self) -> NDArray[np.float64]:
        """
        Get bounding box containing all domains.

        Returns:
            bounds: Array of shape (d, 2) with min/max over all domain boxes
        """
        boxes = np.array([domain.get_bounding_box() for domain in self.domains])

        # Union bounding box: min of lower bounds, max of upper bounds
        bounds = np.zeros((self.dimension, 2))
        bounds[:, 0] = np.min(boxes[:, :, 0], axis=0)
        bounds[:, 1] = np.max(boxes[:, :, 1], axis=0)

        return bounds

    def __repr__(self) -> str:
        """String representation."""
        return f"UnionDomain({len(self.domains)} domains)"


class IntersectionDomain(ImplicitDomain):
    """
    Intersection of multiple domains: D = D₁ ∩ D₂ ∩ ... ∩ Dₙ

    A point is inside if it's inside ALL of the constituent domains.

    Signed distance function:
        φ_intersection(x) = max(φ₁(x), φ₂(x), ..., φₙ(x))

    Example:
        >>> # Rectangle with circular hole removed
        >>> rect = Hyperrectangle(np.array([[0, 1], [0, 1]]))
        >>> hole = Hypersphere(center=[0.5, 0.5], radius=0.2)
        >>> domain = IntersectionDomain([rect, ComplementDomain(hole)])
        >>> # domain = rect \\ hole
    """

    def __init__(self, domains: list[ImplicitDomain]) -> None:
        """
        Initialize intersection of domains.

        Args:
            domains: List of ImplicitDomain objects

        Raises:
            ValueError: If domains list is empty or dimensions don't match
        """
        if not domains:
            raise ValueError("domains list cannot be empty")

        self.domains = domains
        self._dimension = domains[0].dimension

        # Verify all domains have same dimension
        if not all(d.dimension == self._dimension for d in domains):
            raise ValueError(f"All domains must have same dimension, got dimensions: {[d.dimension for d in domains]}")

        # Issue #1041: expose `.bounds` (see UnionDomain for rationale).
        self.bounds: NDArray[np.float64] = self.get_bounding_box()

    @property
    def dimension(self) -> int:
        """Spatial dimension."""
        return self._dimension

    def signed_distance(self, x: NDArray[np.float64]) -> float | NDArray[np.float64]:
        """
        Compute signed distance to intersection.

        φ_intersection(x) = max_i φ_i(x)

        A point is inside intersection only if inside all constituent domains.

        Args:
            x: Point(s) - shape (d,) or (N, d)

        Returns:
            Signed distance(s) - scalar float or array of shape (N,)
        """
        x = np.asarray(x, dtype=float)
        is_single = x.ndim == 1

        # Compute SDF for each domain
        distances = np.array([domain.signed_distance(x) for domain in self.domains])

        # Intersection: maximum distance
        sd = np.max(distances, axis=0)

        return float(sd) if is_single and np.isscalar(sd) else sd

    def get_bounding_box(self) -> NDArray[np.float64]:
        """
        Get bounding box for intersection (conservative estimate).

        Uses the smallest bounding box among constituent domains.

        Returns:
            bounds: Array of shape (d, 2)
        """
        boxes = np.array([domain.get_bounding_box() for domain in self.domains])

        # Intersection bounding box: max of lower bounds, min of upper bounds
        bounds = np.zeros((self.dimension, 2))
        bounds[:, 0] = np.max(boxes[:, :, 0], axis=0)
        bounds[:, 1] = np.min(boxes[:, :, 1], axis=0)

        # Check if intersection is empty
        if np.any(bounds[:, 0] >= bounds[:, 1]):
            # Empty intersection - return first domain's box as fallback
            return self.domains[0].get_bounding_box()

        return bounds

    def __repr__(self) -> str:
        """String representation."""
        return f"IntersectionDomain({len(self.domains)} domains)"


class ComplementDomain(ImplicitDomain):
    """
    Complement of a domain: D^c = ℝ^d \\ D

    A point is inside complement if it's OUTSIDE the original domain.

    Signed distance function:
        φ_complement(x) = -φ(x)

    Example:
        >>> # Everything outside a circle (infinite domain!)
        >>> circle = Hypersphere(center=[0, 0], radius=1.0)
        >>> exterior = ComplementDomain(circle)
        >>> exterior.contains([2, 0])  # True (outside circle)
        >>> exterior.contains([0, 0])  # False (inside circle)
    """

    def __init__(self, domain: ImplicitDomain) -> None:
        """
        Initialize complement of a domain.

        Args:
            domain: ImplicitDomain to complement

        Note:
            Complement domain is typically unbounded! Use with caution for
            sampling - you'll need to provide an explicit bounding box.
        """
        self.domain = domain
        self._dimension = domain.dimension
        self._bounding_box: NDArray[np.float64] | None = None  # Must be set manually for unbounded complements

    @property
    def dimension(self) -> int:
        """Spatial dimension."""
        return self._dimension

    def signed_distance(self, x: NDArray[np.float64]) -> float | NDArray[np.float64]:
        """
        Compute signed distance to complement.

        φ_complement(x) = -φ(x)

        Args:
            x: Point(s) - shape (d,) or (N, d)

        Returns:
            Signed distance(s) - scalar float or array of shape (N,)
        """
        return -self.domain.signed_distance(x)

    def get_bounding_box(self) -> NDArray[np.float64]:
        """
        Get bounding box for complement domain.

        Warning: Complement is typically unbounded! This returns a user-specified
        box if set, otherwise raises an error.

        Returns:
            bounds: Array of shape (d, 2) if manually set

        Raises:
            ValueError: If bounding box not manually specified
        """
        if self._bounding_box is None:
            raise ValueError(
                "ComplementDomain is unbounded. Set bounding box manually via:\ncomplement.set_bounding_box(bounds)"
            )
        return self._bounding_box.copy()

    @property
    def bounds(self) -> NDArray[np.float64]:
        """Issue #1041: expose `.bounds` for FPParticleSolver compatibility.

        Same constraint as ``get_bounding_box``: requires ``set_bounding_box`` to
        have been called (ComplementDomain is unbounded by default).
        """
        return self.get_bounding_box()

    def set_bounding_box(self, bounds: NDArray[np.float64]) -> None:
        """
        Manually set bounding box for sampling from complement.

        Args:
            bounds: Array of shape (d, 2) where bounds[i] = [min_i, max_i]

        Example:
            >>> circle = Hypersphere(center=[0, 0], radius=0.5)
            >>> exterior = ComplementDomain(circle)
            >>> # Sample from unit square minus circle
            >>> exterior.set_bounding_box(np.array([[0, 1], [0, 1]]))
            >>> particles = exterior.sample_uniform(1000)
        """
        self._bounding_box = np.asarray(bounds, dtype=float)

    def __repr__(self) -> str:
        """String representation."""
        return f"ComplementDomain({self.domain})"


class DifferenceDomain(IntersectionDomain):
    """
    Set difference: D₁ \\ D₂ = D₁ ∩ (ℝ^d \\ D₂)

    Points inside D₁ but outside D₂.

    This is a convenience class that's equivalent to:
        IntersectionDomain([D₁, ComplementDomain(D₂)])

    Example:
        >>> # Rectangle with circular hole
        >>> rect = Hyperrectangle(np.array([[0, 1], [0, 1]]))
        >>> hole = Hypersphere(center=[0.5, 0.5], radius=0.2)
        >>> domain_with_hole = DifferenceDomain(rect, hole)
        >>> # Equivalent to: IntersectionDomain([rect, ComplementDomain(hole)])
    """

    def __init__(self, domain1: ImplicitDomain, domain2: ImplicitDomain) -> None:
        """
        Initialize set difference D₁ \\ D₂.

        Args:
            domain1: Base domain (what we start with)
            domain2: Domain to subtract (the "hole")

        Example:
            >>> # Square with circular obstacle removed
            >>> square = Hyperrectangle(np.array([[0, 1], [0, 1]]))
            >>> obstacle = Hypersphere(center=[0.5, 0.5], radius=0.3)
            >>> navigable = DifferenceDomain(square, obstacle)
        """
        complement = ComplementDomain(domain2)

        # Set bounding box for complement (use domain1's box)
        complement.set_bounding_box(domain1.get_bounding_box())

        # Initialize as intersection of domain1 and complement of domain2
        super().__init__([domain1, complement])

        # Store originals for repr
        self.domain1 = domain1
        self.domain2 = domain2

    def __repr__(self) -> str:
        """String representation."""
        return f"DifferenceDomain({self.domain1} \\ {self.domain2})"
