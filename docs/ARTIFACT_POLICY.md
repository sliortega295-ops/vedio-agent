# Artifact publication policy

Ordinary Git contains source, scripts, preregistration, structured result data,
telemetry, failure records, hashes, and parsed profiler output. Large binary or
high-churn artifacts are kept in the tagged GitHub Release:

- Nsight Systems `.nsys-rep` files;
- exported `.sqlite` databases;
- saved `.pt` output tensors;
- complete raw logs;
- the byte-for-byte frozen 147-file experiment snapshot.

This avoids turning normal Git history into a large-binary store while still
publishing the complete test. The archive SHA256 and internal file manifest are
release assets and are copied into `manifests/` where practical.
