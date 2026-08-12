# Terminal UCX qualification checkpoint

This checkpoint runs a bounded, real-UCX DRAM loopback matrix before any
higher terminal-progress layer is allowed to build on the native completion
contract:

| `UCX_TLS` | `UCX_NET_DEVICES` | Engine paths |
|---|---|---|
| `self` | unset | shared progress thread, thread pool/composite |
| `tcp` | `lo` | shared progress thread, thread pool/composite |

Each coordinate runs in a fresh process because UCX configuration is
process-scoped. The runner binds `NIXL_PLUGIN_DIR`, sets
`CUDA_VISIBLE_DEVICES=` and `NVIDIA_VISIBLE_DEVICES=void`, passes only `DRAM`
descriptors, captures NIXL/UCX revisions and executable identity, and traces
device opens without calling CUDA, NVML, or `nvidia-smi`. The receipt records
exact argv and environment beside the transport selected in native attestation.

Every coordinate uses separate source and destination registrations, runs small
and large writes, verifies destination bytes and SHA-256, consumes take-once
completion authority, and proves transfer cancellation, bounded queue overflow,
shutdown cancellation, and zero retained lifecycle inventory. Thread-pool
coordinates also repost one composite request across two ordered transfer
generations and verify that its stable request identity acquires fresh terminal
authority each time.

The transport split is structural. `self` uses one NIXL agent's preinstalled
local route and the string-target transfer API. Its data callbacks arrive before
poster return, while real UCX self endpoint flushes complete immediately without
invoking a callback. Schema v4 reconciles the callback-free immediate post count
against the exact set of attested, remote-flushed endpoints. A local route has no
remote handle, remote capability, authenticated attached notification, or peer
failure; those semantics are typed non-applicability with their UCX/NIXL source
anchors. It may not load its own metadata or manufacture a remote handle.

TCP over `lo` uses two distinct agents and endpoints. Every attested endpoint
flush must complete through a callback, and the coordinate must prove callback
delivery after poster return. TCP additionally proves attached
authenticated-notification success and failure. Its notification-failure fixture
stops the independent peer only after the transfer attestation records
all-endpoint remote-flush authority, verifies that notification terminality is
still pending at that boundary, and then kills the peer. TCP also proves remote failure and capability
epoch advancement, endpoint failure, and retirement on three distinct route
generations. FAILED and RETIRED are exercised by separate terminal
subscriptions. `terminal_ucx_receipt.py` rejects hard-coded pass fields and the
physically over-constrained predecessor schema.

`terminal_ucx_api_adapter.*` is the only API seam. It exists so this checkpoint
can compile while the terminal-agent and UCX callback branches converge; the
transfer and evidence path itself remains public-API authoritative.
