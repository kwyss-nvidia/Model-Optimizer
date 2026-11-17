# Activation row validity

`TensorQuantizer(inputs, validity=TensorValidity(...))` accepts optional execution metadata.
Existing calls and registered two-argument backends are unchanged. The metadata describes
flattened leading dimensions; the final dimension contains features. It is never checkpointed.

Backends opt in with `register_quant_backend(name, entrypoint, supports_validity=True)`.
Their entrypoint accepts `(inputs, quantizer, *, validity=None)`. A backend must exclude
invalid rows from statistics, preserve the input shape, and read device count/mask contents
at execution time. It may use `validity.num_valid_rows` as a prefix only when
`validity.valid_rows_are_prefix` is true; otherwise it needs `validity.is_padding`.
A prefix-only backend must reject masks it cannot support. For psx-luts, a guaranteed prefix
maps to `valid_rows`; this adapter belongs in the separately registered backend, not vLLM.

The vLLM plugin borrows ForwardContext tensors without creating a runtime or importing
ModelOpt into vLLM. Ordinary linear calls preserve the token row axis. Known unsorted,
local MoE kernel layouts map FC2 through token-major top-k repetition. Context lookup,
translation and row expansion live in `plugins/vllm_validity.py`; `TensorValidity` itself
contains only metadata and validation. Expansion is performed only for a supporting FC2
input or MoE output quantizer; FC1 input-only quantization reuses the original metadata. Sorted buffers,
expert-parallel dispatch, and mismatched capacities fail rather than guessing a mapping.
Other transformed layouts need explicit adapters before they can use this contract.

The generic path never masks, crops, or zeros activation values. Native quantizers currently
use their existing behavior without validity. Only explicitly opted-in backends receive the
metadata and decide how their kernels consume it. Calibration, static block reshaping, bias
statistics, real quantization and DTensor inputs also use the existing path without validity.
A TODO in `TensorQuantizer.forward` tracks mapping validity through DTensor placements to
local shards before enabling support.
Unsupported backends, subclasses and modes fall back to their existing call path without
validity. Supported quantizers still validate supplied layout metadata; an invalid layout
is not silently accepted. Sequential quantizers forward validity independently to each
supporting stage.

Validity does not disable weight caching. An enrolled quantizer forwards validity to the
backend on a cache miss and reuses its result on hits. The caller promises that both weights
and validity remain constant within a cache scope; close or invalidate that scope before
changing either. Unenrolled activation quantizers continue computing on every call.

The producer must provide a count in [0, row_capacity], update buffers before replay on the
appropriate stream, and keep addresses stable. Validation checks metadata without device
synchronization. Prefix guarantees are static, not inferred from dummy capture values.
The count is used during capture as supplied; the consumer does not replace it with zero.

GPU graph replay and torch.compile validation require a compatible CUDA environment;
CPU tests cover padding invariance, mutable counts, backend compatibility, and layout rejection.
