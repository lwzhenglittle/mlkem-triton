"""Reference and baseline implementations for ML-KEM.

Not on the hot path: `refimpl` wraps kyber-py, `sampling_cpu` is a hashlib reference
for the sampling and hashing steps, and `pipeline_host_hash` is the deliberately
host-hashing decapsulator used for comparisons. Production code lives one level up.
"""
