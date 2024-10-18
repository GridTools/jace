# Main Differences Between DaCe and JaCe and JAX and JaCe

Essentially JaCe is a frontend that allows DaCe to process JAX code, thus it has to be compatible with both, at least in some sense.
We will now list the main differences between them, furthermore, you should also consult the ROADMAP.

### JAX vs. JaCe:

- JaCe always traces with enabled `x64` mode.
  This is a restriction that might be lifted in the future.
- JAX returns scalars as zero-dimensional arrays, JaCe returns them as array with shape `(1, )`.
- In JAX parts of the computation runs on CPU parts on GPU, in JaCe everything runs (currently) either on CPU or GPU.
- Currently JaCe is only able to run on CPU (will be lifted soon).
- Currently JaCe is not able to run distributed (will be lifted later).
- Currently not all primitives are supported.
- JaCe does not return `jax.Array` instances, but NumPy/CuPy arrays.
- The execution is not asynchronous.

### DaCe vs. JaCe:

- JaCe accepts complex objects using JAX' pytrees.
- JaCe will support scalar inputs on GPU.
