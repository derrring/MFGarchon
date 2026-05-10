"""
JAX Backend for MFGarchon

High-performance backend using JAX for GPU acceleration, automatic differentiation,
and JIT compilation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # For type checking, always assume JAX is available
    import jax
    import jax.numpy as jnp
    from jax import device_get, device_put, grad, jit, vmap
    from jax.scipy.integrate import trapezoid as jax_trapezoid

    JAX_AVAILABLE = True
else:
    # At runtime, handle optional JAX import
    try:
        import jax
        import jax.numpy as jnp
        from jax import device_get, device_put, grad, jit, vmap
        from jax.scipy.integrate import trapezoid as jax_trapezoid

        JAX_AVAILABLE = True
    except ImportError:
        JAX_AVAILABLE = False
        jax = None  # type: ignore
        jnp = None  # type: ignore
        device_get = None  # type: ignore
        device_put = None  # type: ignore
        grad = None  # type: ignore
        jit = None  # type: ignore
        vmap = None  # type: ignore
        jax_trapezoid = None  # type: ignore


import numpy as np

from mfgarchon.utils.mfg_logging import get_logger

from .base_backend import BaseBackend

# Module logger
logger = get_logger(__name__)


class JAXBackend(BaseBackend):
    """JAX-based computational backend with GPU support."""

    def __init__(
        self,
        device: str = "auto",
        precision: str = "float64",
        jit_compile: bool = True,
        **kwargs,
    ):
        if not JAX_AVAILABLE:
            raise ImportError(
                "JAX backend requested but JAX is not installed. "
                "Install with: pip install 'mfgarchon[jax]' or 'pip install jax jaxlib'"
            )

        self.jit_compile = jit_compile
        # Issue #1068: explicit None-init for JIT cache slots — replaces hasattr
        # duck-typing per CLAUDE.md "Object Shape Stability". Also helps Numba/JAX
        # static analyzers track the Optional[Callable] type.
        self._jit_hjb_step: Callable | None = None
        self._jit_fpk_step: Callable | None = None
        self._jit_hamiltonian: Callable | None = None
        self._jit_optimal_control: Callable | None = None
        super().__init__(device=device, precision=precision, **kwargs)

    def _setup_backend(self):
        """Initialize JAX backend."""
        import jax

        # Set precision
        if self.precision == "float32":
            self.dtype = jnp.float32
            # Enable float32 mode in JAX
            jax.config.update("jax_enable_x64", False)
        else:
            self.dtype = jnp.float64
            jax.config.update("jax_enable_x64", True)

        # Device selection
        if self.device == "auto":
            # Auto-select best available device
            devices = jax.devices()
            gpu_devices = [d for d in devices if "gpu" in str(d).lower()]
            if gpu_devices:
                self.target_device = gpu_devices[0]
                self.device = "gpu"
            else:
                self.target_device = devices[0]  # CPU
                self.device = "cpu"
        elif self.device == "gpu":
            gpu_devices = [d for d in jax.devices() if "gpu" in str(d).lower()]
            if not gpu_devices:
                raise RuntimeError("GPU requested but no GPU devices available")
            self.target_device = gpu_devices[0]
        else:  # cpu
            cpu_devices = [d for d in jax.devices() if "cpu" in str(d).lower()]
            self.target_device = cpu_devices[0] if cpu_devices else jax.devices()[0]

        # Compile commonly used functions if JIT is enabled
        if self.jit_compile:
            self._compile_core_functions()

    def _compile_core_functions(self):
        """Pre-compile frequently used functions."""
        # Create JIT-compiled versions of core operations
        self._jit_hjb_step = jit(self._hjb_step_impl)
        self._jit_fpk_step = jit(self._fpk_step_impl)
        self._jit_hamiltonian = jit(self._hamiltonian_impl)
        self._jit_optimal_control = jit(self._optimal_control_impl)

    @property
    def name(self) -> str:
        return "jax"

    @property
    def array_module(self):
        return jnp

    # Array Operations
    def array(self, data, dtype=None):
        if dtype is None:
            dtype = self.dtype
        arr = jnp.array(data, dtype=dtype)
        return device_put(arr, self.target_device)

    def zeros(self, shape, dtype=None):
        if dtype is None:
            dtype = self.dtype
        arr = jnp.zeros(shape, dtype=dtype)
        return device_put(arr, self.target_device)

    def ones(self, shape, dtype=None):
        if dtype is None:
            dtype = self.dtype
        arr = jnp.ones(shape, dtype=dtype)
        return device_put(arr, self.target_device)

    def linspace(self, start, stop, num):
        arr = jnp.linspace(start, stop, num, dtype=self.dtype)
        return device_put(arr, self.target_device)

    def meshgrid(self, *arrays, indexing="xy"):
        grids = jnp.meshgrid(*arrays, indexing=indexing)
        return [device_put(grid, self.target_device) for grid in grids]

    # Mathematical Operations
    def grad(self, func, argnum=0):
        """Automatic differentiation using JAX."""
        return grad(func, argnums=argnum)

    def trapezoid(self, y, x=None, dx=1.0, axis=-1):
        return jax_trapezoid(y, x=x, dx=dx, axis=axis)

    def diff(self, a, n=1, axis=-1):
        return jnp.diff(a, n=n, axis=axis)

    def interp(self, x, xp, fp):
        return jnp.interp(x, xp, fp)

    # Linear Algebra
    def solve(self, A, b):
        return jnp.linalg.solve(A, b)

    def eig(self, a):
        return jnp.linalg.eig(a)

    # Statistics
    def mean(self, a, axis=None):
        return jnp.mean(a, axis=axis)

    def std(self, a, axis=None):
        return jnp.std(a, axis=axis)

    def max(self, a, axis=None):
        return jnp.max(a, axis=axis)

    def min(self, a, axis=None):
        return jnp.min(a, axis=axis)

    # MFG-Specific Operations (JIT-compiled implementations)
    def _hamiltonian_impl(self, x, p, m, problem_params):
        """Default Hamiltonian implementation."""
        return 0.5 * p**2

    def _optimal_control_impl(self, x, p, m, problem_params):
        """Default optimal control implementation."""
        return -p

    def compute_hamiltonian(self, x, p, m, problem_params):
        if self.jit_compile and self._jit_hamiltonian is not None:
            return self._jit_hamiltonian(x, p, m, problem_params)
        else:
            return self._hamiltonian_impl(x, p, m, problem_params)

    def compute_optimal_control(self, x, p, m, problem_params):
        if self.jit_compile and self._jit_optimal_control is not None:
            return self._jit_optimal_control(x, p, m, problem_params)
        else:
            return self._optimal_control_impl(x, p, m, problem_params)

    def _hjb_step_impl(self, U, M, dt, dx, x_grid, problem_params):
        """JIT-compiled HJB step implementation."""

        # Compute spatial gradient using automatic differentiation
        def U_interp(x_val):
            return jnp.interp(x_val, x_grid, U)

        # Vectorized gradient computation
        dU_dx = vmap(grad(U_interp))(x_grid)

        # Compute Hamiltonian
        H = self._hamiltonian_impl(x_grid, dU_dx, M, problem_params)

        # Time step
        U_new = U - dt * H
        return U_new

    def _fpk_step_impl(self, M, U, dt, dx, x_grid, problem_params):
        """JIT-compiled FPK step implementation."""

        # Compute spatial gradient of U
        def U_interp(x_val):
            return jnp.interp(x_val, x_grid, U)

        dU_dx = vmap(grad(U_interp))(x_grid)

        # Compute optimal control
        a_opt = self._optimal_control_impl(x_grid, dU_dx, M, problem_params)

        # Compute flux and its divergence
        flux = M * a_opt

        # Finite difference for divergence (vectorized)
        div_flux = jnp.zeros_like(M)
        div_flux = div_flux.at[1:-1].set((flux[2:] - flux[:-2]) / (2 * dx))
        div_flux = div_flux.at[0].set((flux[1] - flux[0]) / dx)
        div_flux = div_flux.at[-1].set((flux[-1] - flux[-2]) / dx)

        # Diffusion term
        sigma_sq = problem_params.get("sigma_sq", 0.01)
        d2M_dx2 = jnp.zeros_like(M)
        d2M_dx2 = d2M_dx2.at[1:-1].set((M[2:] - 2 * M[1:-1] + M[:-2]) / (dx**2))
        d2M_dx2 = d2M_dx2.at[0].set(d2M_dx2[1])
        d2M_dx2 = d2M_dx2.at[-1].set(d2M_dx2[-2])

        diffusion = 0.5 * sigma_sq * d2M_dx2

        # Time step
        M_new = M + dt * (-div_flux + diffusion)

        # Ensure non-negativity and conservation
        M_new = jnp.maximum(M_new, 0)
        total_mass = self.trapezoid(M_new, dx=dx)
        M_new = jnp.where(total_mass > 1e-12, M_new / total_mass, M_new)

        return M_new

    def hjb_step(self, U, M, dt, dx, problem_params):
        x_grid = problem_params.get("x_grid", jnp.linspace(0, 1, len(U)))
        if self.jit_compile and self._jit_hjb_step is not None:
            return self._jit_hjb_step(U, M, dt, dx, x_grid, problem_params)
        else:
            return self._hjb_step_impl(U, M, dt, dx, x_grid, problem_params)

    def fpk_step(self, M, U, dt, dx, problem_params):
        x_grid = problem_params.get("x_grid", jnp.linspace(0, 1, len(M)))
        if self.jit_compile and self._jit_fpk_step is not None:
            return self._jit_fpk_step(M, U, dt, dx, x_grid, problem_params)
        else:
            return self._fpk_step_impl(M, U, dt, dx, x_grid, problem_params)

    # Performance and Compilation
    def compile_function(self, func, *args, **kwargs):
        """JIT compile function for performance."""
        if self.jit_compile:
            return jit(func, *args, **kwargs)
        else:
            return func

    def vectorize(self, func, signature=None):
        """Vectorize function using JAX vmap."""
        return vmap(func)

    # Device Management
    def to_device(self, array):
        """Move array to JAX device."""
        return device_put(array, self.target_device)

    def from_device(self, array):
        """Move array from JAX device to CPU."""
        return device_get(array)

    def to_numpy(self, array) -> np.ndarray:
        """Convert JAX array to numpy array."""
        return np.asarray(device_get(array))

    def from_numpy(self, array: np.ndarray):
        """Convert numpy array to JAX array on device."""
        jax_array = jnp.array(array, dtype=self.dtype)
        return device_put(jax_array, self.target_device)

    def get_device_info(self) -> dict:
        """Get JAX device information."""
        return {
            "backend": self.name,
            "device": str(self.target_device),
            "device_type": self.device,
            "precision": self.precision,
            "jit_enabled": self.jit_compile,
            "jax_version": jax.__version__,
            "available_devices": [str(d) for d in jax.devices()],
        }

    def memory_usage(self) -> dict | None:
        """Get GPU memory usage if available."""
        try:
            if "gpu" in str(self.target_device).lower():
                # Try to get GPU memory info
                import subprocess

                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used,memory.total",
                        "--format=csv,nounits,noheader",
                    ],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split("\n")
                    used, total = map(int, lines[0].split(","))
                    return {
                        "gpu_memory_used_mb": used,
                        "gpu_memory_total_mb": total,
                        "gpu_memory_percent": (used / total) * 100,
                    }
            return None
        except (RuntimeError, AttributeError, OSError) as e:
            # Issue #547: GPU memory query can fail for various reasons
            logger.debug("Failed to get JAX GPU memory stats: %s", e)
            return None

    # Backend Capabilities (for auto-switching)
    def has_capability(self, capability: str) -> bool:
        """Check if JAX backend supports a specific capability."""
        capabilities = {
            "parallel_kde": self.device == "gpu",  # GPU enables parallel KDE
            "parallel_interpolation": self.device == "gpu",
            "low_latency": self.device == "gpu",  # JAX GPU has very low latency
            "high_bandwidth": self.device == "gpu",  # CUDA has high bandwidth
            "unified_memory": False,  # JAX uses separate GPU memory
            "jit_compilation": self.jit_compile,  # XLA JIT compilation
            "kernel_fusion": self.jit_compile,  # XLA auto-fuses kernels
            "auto_vectorization": True,  # vmap auto-vectorizes
        }
        return capabilities.get(capability, False)

    def get_performance_hints(self) -> dict:
        """Return performance characteristics for strategy selection."""
        if self.device == "gpu":
            # Assume CUDA GPU (JAX primary GPU target)
            return {
                "kernel_overhead_us": 5,  # Very low latency with XLA
                "memory_bandwidth_gb": 900,  # High for modern CUDA GPUs
                "device_type": "cuda",
                "optimal_problem_size": (10000, 100, 50),
                "kernel_fusion": self.jit_compile,  # XLA fuses operations
                "expected_speedup": 4.0 if self.jit_compile else 2.0,
            }
        else:  # CPU
            return {
                "kernel_overhead_us": 0,  # No kernel overhead on CPU
                "memory_bandwidth_gb": 50,
                "device_type": "cpu",
                "optimal_problem_size": (5000, 50, 20),
                "kernel_fusion": self.jit_compile,  # XLA can fuse on CPU too
                "expected_speedup": 1.5 if self.jit_compile else 1.0,
            }
