"""ML-KEM-specific Triton kernels.

The scheme-agnostic kernels live in `triton_crypto.primitives`; these implement the
two GEMM passes, the base-case multiply, ByteEncode/Compress, CBD and the padded
concatenation helpers used by the ML-KEM pipeline.
"""
