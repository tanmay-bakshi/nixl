# Terminal UCX qualification checkpoint

This checkpoint runs a bounded, real-UCX DRAM loopback matrix before any
higher terminal-progress layer is allowed to build on the native completion
contract:

| `UCX_TLS` | `UCX_NET_DEVICES` | Engine paths |
|---|---|---|
| `self` | unset | shared progress thread, thread pool/composite |
| `tcp` | `lo` | shared progress thread, thread pool/composite |

Each coordinate runs in a fresh process because UCX configuration is
process-scoped. The runner sets `CUDA_VISIBLE_DEVICES=` and
`NVIDIA_VISIBLE_DEVICES=void`, passes only `DRAM` descriptors, captures the
NIXL/UCX revisions and executable digest, and traces device opens without
calling CUDA, NVML, or `nvidia-smi`. The receipt records exact argv and relevant
environment beside the transport observed from completion attestation, rather
than trusting launch intent.

Every coordinate must cover separately registered source and destination
memory over two exact endpoints, both immediate and asynchronous callback
populations, subscription-before-post, notification-before-terminal ordering,
destination byte/SHA-256 verification, completion-attestation digest and
take-once authority, and callback/poster ordering counts. It must also bind
capability READY, FAILED, RETIRED, and epoch transitions plus isolated transfer
cancellation, remote failure, notification failure, queue overflow, and
shutdown-cancellation cases. Coordinate and aggregate shutdown inventories must
contain zero public subscriptions, backend producers, callback slots,
continuations, and queued events. `terminal_ucx_receipt.py` rejects the
predecessor schema and seals only this complete matrix.

`terminal_ucx_api_adapter.*` is the only API seam. It exists so this checkpoint
can compile while the terminal-agent and UCX callback branches converge; the
transfer and evidence path itself remains public-API authoritative.
