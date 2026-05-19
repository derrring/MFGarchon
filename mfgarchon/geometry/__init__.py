"""
Geometry package for MFGarchon: Professional mesh generation and complex domain support.

This package implements comprehensive geometry management for MFG problems:
- TensorProductGrid: Unified Cartesian grids for 1D/2D/3D/nD (recommended)
- Unstructured meshes (Mesh2D/3D): FEM/FVM triangular/tetrahedral meshes via Gmsh
- Implicit domains: High-dimensional meshfree domains with signed distance functions
- Gmsh → Meshio → PyVista pipeline for professional mesh generation
- Advanced boundary condition management for complex domains

Key Components:
- TensorProductGrid: Unified nD Cartesian grids for finite difference methods
- Mesh2D/3D: Unstructured meshes for FEM/FVM (d≤3)
- implicit: Meshfree geometry infrastructure for any dimension
  - Hyperrectangle: Axis-aligned boxes (O(d) sampling, no rejection!)
  - Hypersphere: Balls/circles for obstacles
  - CSG operations: Union, Intersection, Difference for complex domains
- Geometry: Unified ABC for all geometry types (in base.py)
- MeshData: Universal mesh data container
- MeshPipeline: Complete Gmsh → Meshio → PyVista workflow orchestration
- MeshManager: High-level mesh management for multiple geometries
- BoundaryConditions: Unified boundary condition specification

Discretization Methods:
- Use TensorProductGrid for finite difference solvers (all dimensions)
- Use Mesh* (Gmsh) for FEM/FVM problems (d≤3)
- Use implicit.* (SDF) for high-dimensional particle-collocation (d≥4)

Note: SimpleGrid1D/2D/3D have been removed. Use TensorProductGrid instead.
"""

from __future__ import annotations

# Geometric operators (now in mfgarchon.operators, re-exported here for convenience)
from mfgarchon.operators import GeometryProjector, ProjectionRegistry

# AMR stub (for future library integration)
from .amr import AdaptiveGeometry, AMRNotImplementedError, create_amr_grid, is_adaptive

# Boundary conditions (unified BC framework)
from .boundary import (
    BCSegment,
    BCType,
    BoundaryConditions,
    MixedBoundaryConditions,
    create_standard_boundary_names,
    dirichlet_bc,
    neumann_bc,
    no_flux_bc,
    periodic_bc,
    robin_bc,
)
from .cloud_geodesic import build_geodesic_field, compute_geodesic_distance

# Collocation point generation
from .collocation import (
    CollocationPointSet,
    CollocationSampler,
    generate_collocation,
)

# Graph-based geometry (networks + mazes)
from .graph import (
    BaseNetworkGeometry,
    GridNetwork,
    HybridMazeGenerator,
    MazeAlgorithm,
    MazeConfig,
    MazeGeometry,
    NetworkData,
    NetworkType,
    RandomNetwork,
    ScaleFreeNetwork,
    VoronoiMazeGenerator,
    maze_Algorithm,
    maze_CellularAutomataConfig,
    maze_CellularAutomataGenerator,
    maze_Config,
    maze_Geometry,
    maze_HybridGenerator,
    maze_RecursiveDivisionConfig,
    maze_RecursiveDivisionGenerator,
    maze_VoronoiGenerator,
)

# Network backend (from graph subdirectory - canonical location)
from .graph.network_backend import NetworkBackendType, OperationType, get_backend_manager, set_preferred_backend

# Legacy network imports (from old file names - now in graph subdirectory)
from .graph.network_geometry import compute_network_statistics, create_network

# Grid geometry - Import from subdirectories (canonical locations)
from .grids.tensor_grid import TensorProductGrid

# Implicit geometry
from .implicit import (
    ComplementDomain,
    DifferenceDomain,
    Hyperrectangle,
    Hypersphere,
    ImplicitDomain,
    IntersectionDomain,
    PointCloudGeometry,
    UnionDomain,
)

# Mask generation utilities
from .masks import (
    boundary_segment_mask,
    circle_mask,
    combine_masks,
    create_mask,
    get_boundary_mask,
    indices_to_mask,
    invert_mask,
    load_mask,
    mask_to_indices,
    polygon_mask,
    rectangle_mask,
    save_mask,
)

# Mesh data structures (from meshes subdirectory)
# Mesh geometry
from .meshes import Mesh1D, Mesh2D, Mesh3D, MeshData, MeshManager, MeshPipeline, MeshVisualizationMode

# Region predicate factories (BC Roadmap Phase 1.3)
from .predicates import box_region, halfspace_region, sdf_region, sphere_region

# Unified geometry protocol
from .protocol import (
    # Boundary-aware protocol (for unified BC handling)
    BoundaryAwareProtocol,
    BoundaryType,
    # Core geometry protocol
    GeometryProtocol,
    GeometryType,
    detect_geometry_type,
    is_boundary_aware,
    is_geometry_compatible,
    validate_boundary_aware,
    validate_geometry,
)

# Geometry traits (Issue #732 Tier 1b)
from .traits import (
    BoundaryAware,
    BoundaryDef,
    ConnectivityAware,
    ConnectivityType,
    StructureAware,
    StructureType,
    TraitAwareGeometry,
)

__all__ = [
    # Collocation point generation
    "CollocationPointSet",
    "CollocationSampler",
    "generate_collocation",
    # Multi-dimensional geometry components
    "BaseNetworkGeometry",
    # Boundary condition components
    "BCSegment",
    "BCType",
    "BoundaryConditions",
    "MixedBoundaryConditions",
    "create_standard_boundary_names",
    "dirichlet_bc",
    "neumann_bc",
    "no_flux_bc",
    "periodic_bc",
    "robin_bc",
    # Geometry components
    "Mesh1D",
    "Mesh2D",
    "Mesh3D",
    # Unified geometry protocol
    "GeometryProtocol",
    "GeometryType",
    # Geometry traits (Issue #732 Tier 1b)
    "ConnectivityType",
    "StructureType",
    "BoundaryDef",
    "ConnectivityAware",
    "StructureAware",
    "BoundaryAware",
    "TraitAwareGeometry",
    # AMR stub (for future library integration)
    "AdaptiveGeometry",
    "AMRNotImplementedError",
    "create_amr_grid",
    "is_adaptive",
    # Cloud geodesic distance (Issue #1093)
    "compute_geodesic_distance",
    "build_geodesic_field",
    # Boundary-aware protocol
    "BoundaryAwareProtocol",
    "BoundaryType",
    "is_boundary_aware",
    "validate_boundary_aware",
    # Geometry projection (Issue #257)
    "GeometryProjector",
    "ProjectionRegistry",
    "GridNetwork",
    "MeshData",
    "MeshManager",
    "MeshPipeline",
    "MeshVisualizationMode",
    # Network backend components
    "NetworkBackendType",
    # Network geometry components
    "NetworkData",
    "NetworkType",
    "OperationType",
    "PointCloudGeometry",
    "RandomNetwork",
    "ScaleFreeNetwork",
    "TensorProductGrid",
    # Implicit geometry (CSG operations)
    "ComplementDomain",
    "DifferenceDomain",
    "Hyperrectangle",
    "Hypersphere",
    "ImplicitDomain",
    "IntersectionDomain",
    "UnionDomain",
    # Region predicate factories (BC Roadmap Phase 1.3)
    "box_region",
    "sphere_region",
    "sdf_region",
    "halfspace_region",
    # Factory and utility functions
    "compute_network_statistics",
    "create_network",
    "detect_geometry_type",
    "get_backend_manager",
    "is_geometry_compatible",
    # Maze geometry (primary names)
    "MazeGeometry",
    "MazeAlgorithm",
    "MazeConfig",
    "HybridMazeGenerator",
    "VoronoiMazeGenerator",
    # Maze with maze_ prefix
    "maze_Geometry",
    "maze_Algorithm",
    "maze_CellularAutomataConfig",
    "maze_CellularAutomataGenerator",
    "maze_Config",
    "maze_HybridGenerator",
    "maze_RecursiveDivisionConfig",
    "maze_RecursiveDivisionGenerator",
    "maze_VoronoiGenerator",
    "set_preferred_backend",
    "validate_geometry",
    # Mask generation utilities
    "boundary_segment_mask",
    "circle_mask",
    "combine_masks",
    "create_mask",
    "get_boundary_mask",
    "indices_to_mask",
    "invert_mask",
    "load_mask",
    "mask_to_indices",
    "polygon_mask",
    "rectangle_mask",
    "save_mask",
]

# Version information
__version__ = "1.0.0"


# Optional dependency checks with helpful error messages
def _check_optional_dependencies():
    """Check for optional dependencies and provide helpful messages."""
    dependencies = {
        "gmsh": "pip install gmsh",
        "meshio": "pip install meshio",
        "pyvista": "pip install pyvista",
    }

    missing = []
    for dep, install_cmd in dependencies.items():
        try:
            __import__(dep)
        except ImportError:
            missing.append(f"{dep} ({install_cmd})")

    if missing:
        import warnings

        warnings.warn(
            f"Optional dependencies missing for full geometry functionality: {', '.join(missing)}",
            ImportWarning,
        )


# Check dependencies on import
_check_optional_dependencies()
