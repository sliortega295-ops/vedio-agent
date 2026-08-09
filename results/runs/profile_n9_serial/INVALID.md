# INVALID N9 serial Nsight attempt

The N9 serial reference trace was aborted before the capture range because an external
/data/anqian/stable-diffusion.cpp-cuda-sparse/bin/sd-cli process appeared on physical
GPU0. No report was generated and no datum from this attempt is used. The process was
not killed. Retry v2 waits for a sustained exclusive-idle interval and retains the
runtime interference guard.
