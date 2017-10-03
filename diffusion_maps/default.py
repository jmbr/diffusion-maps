"""Default values.

"""

renormalization = 1             # Renormalization paramter (0 <= alpha <= 1).

num_eigenpairs = 11             # Number of eigenvalue/eigenvectors to obtain.

try:
    import pycuda.autoinit      # noqa
except ImportError:
    use_cuda = False            # Use CPU code.
else:
    use_cuda = True             # Use GPU-accelerated code.
