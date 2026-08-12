# Terminal UCX qualification checkpoint

This checkpoint runs a bounded, real-UCX DRAM loopback matrix before any
higher terminal-progress layer is allowed to build on the native completion
contract:

| `UCX_TLS` | `UCX_NET_DEVICES` | Engine paths |
|---|---|---|
| `self` | unset | shared progress thread, thread pool/composite |
| `tcp` | `lo` | shared progress thread, thread pool/composite |

Each transport arm runs in a fresh process because UCX configuration is
process-scoped. The runner sets `CUDA_VISIBLE_DEVICES=` and
`NVIDIA_VISIBLE_DEVICES=void`, passes only `DRAM` descriptors, captures the
NIXL/UCX revisions and executable digest, and traces device opens without
calling CUDA, NVML, or `nvidia-smi`. The receipt records the observed transport
from completion attestation rather than trusting the requested environment.

The native executable must cover subscription-before-post, immediate and
asynchronous completion populations, notification terminality, capability
snapshot-after-ready and retirement, destination byte/SHA-256 verification,
completion-attestation digest and take-once authority, explicit
callback-before-return observations, loaded runtime paths/build IDs, exact
argv/environment, and exact zero
subscription/producer inventory at shutdown. `terminal_ucx_receipt.py`
validates and seals the combined matrix.

`terminal_ucx_api_adapter.*` is the only API seam. It exists so this checkpoint
can compile while the terminal-agent and UCX callback branches converge; the
transfer and evidence path itself remains public-API authoritative.
