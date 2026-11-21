# Activation row validity

`TensorQuantizer(inputs, validity=TensorValidity(...))` accepts optional execution metadata.
Existing calls and registered two-argument backends are unchanged. The metadata describes
flattened leading dimensions; the final dimension contains features. It is never checkpointed.

Backends opt in with `register_quant_backend(name, entrypoint, supports_validity=True)`.
Their entrypoint accepts `(inputs, quantizer, *, validity=None)`. A backend must exclude
invalid rows from statistics, preserve the input shape, and read device count/mask contents
at execution time. It may use `validity.num_valid_rows` as a prefix only when
`validity.valid_rows_are_prefix` is true; otherwise it needs `validity.is_padding`.
A prefix-only provider must not interpret a non-prefix count as a prefix. Adapters may
fall back to its legacy call when unsupported. For psx-luts, a guaranteed prefix maps to
`valid_rows`; this adapter belongs in the separately registered backend, not vLLM.

The vLLM plugin borrows ForwardContext tensors without creating a runtime or importing
ModelOpt into vLLM. Ordinary linear calls preserve the token row axis. Known unsorted,
local MoE kernel layouts map FC2 through token-major top-k repetition. Context lookup,
translation and row expansion live in `plugins/vllm_validity.py`; `TensorValidity` itself
contains metadata, validation, and a helper for padding-free amax statistics. Expansion is performed only for a supporting FC2
input or MoE output quantizer; FC1 input-only quantization reuses the original metadata. Sorted buffers,
expert-parallel dispatch, and mismatched capacities fail rather than guessing a mapping.
Other transformed layouts need explicit adapters before they can use this contract.

The generic path does not mask or crop the values sent to quantization. Native quantizers currently
use their existing behavior without validity. Only explicitly opted-in backends receive the
metadata and decide how their kernels consume it. For these backends, max calibration
excludes padding, and `_get_amax(inputs, validity=validity)` computes a padding-free dynamic
statistic when no stored amax exists. Stored and constant amax remain unchanged. The statistic
helper uses a fixed-shape masked temporary; it does not modify quantizer inputs or read a
count on the host. Zero is neutral for absolute maximum, including all-padding inputs.
Other calibration algorithms, static block reshaping, bias statistics, real quantization
and DTensor inputs use the existing path without validity.
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
