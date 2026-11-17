# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Adapt vLLM token metadata without depending on a quantization backend."""

import inspect
from functools import cache

from ..nn import TensorValidity

__all__ = [
    "TOKEN_MAJOR_MOE_KERNELS",
    "activation_validity",
    "moe_validity",
    "validity_from_forward_context",
]

TOKEN_MAJOR_MOE_KERNELS = (
    "invoke_fused_moe_kernel",
    "invoke_fused_moe_triton_kernel",
    "dispatch_fused_moe_kernel",
)


def validity_from_forward_context(context) -> TensorValidity | None:
    """Borrow runner-owned tensors; callers must establish the activation's row mapping."""
    count = getattr(context, "num_valid_tokens", None)
    mask = getattr(context, "is_padding", None)
    capacity = getattr(context, "token_capacity", None)
    if count is None and mask is None and capacity is None:
        return None
    if capacity is None and mask is not None:
        capacity = mask.numel()
    return TensorValidity(
        num_valid_rows=count,
        is_padding=mask,
        valid_rows_are_prefix=getattr(context, "valid_tokens_are_prefix", False),
        row_capacity=capacity,
    )


def _repeat_rows(validity: TensorValidity, repeats: int) -> "TensorValidity":
    """Map token-major repetition (e.g. top-k expansion) to activation rows."""
    if repeats < 1 or validity.row_capacity is None:
        raise ValueError("Row repetition requires positive repeats and a known capacity.")
    if repeats == 1:
        return validity
    return TensorValidity(
        None if validity.num_valid_rows is None else validity.num_valid_rows * repeats,
        None
        if validity.valid_rows_are_prefix or validity.is_padding is None
        else validity.is_padding.repeat_interleave(repeats),
        validity.valid_rows_are_prefix,
        validity.row_capacity * repeats,
    )


def activation_validity():
    """Borrow validity from the active vLLM forward, if one exists."""
    # Keep vLLM optional for users of the framework-independent metadata adapter.
    from vllm.forward_context import get_forward_context, is_forward_context_available

    if not is_forward_context_available():
        return None
    return validity_from_forward_context(get_forward_context())


@cache
def _kernel_signature(kernel):
    return inspect.signature(kernel)


def moe_validity(a, b, c, args, kwargs, kernel, *, second_layer, need_input, need_output):
    """Map runner validity only for the MoE operands that will consume it."""
    # vLLM is optional until serving invokes this adapter.
    from vllm.distributed.parallel_state import get_ep_group

    validity = activation_validity()
    if validity is None:
        return None, None
    # Only these launchers establish token-major inputs; packed expert layouts
    # need their dispatch permutation applied to the validity mask first.
    if kernel.__name__ not in TOKEN_MAJOR_MOE_KERNELS:
        raise NotImplementedError("Unknown MoE activation layout for row validity.")
    bound = _kernel_signature(kernel).bind(a, b, c, *args, **kwargs)
    bound.apply_defaults()
    if bound.arguments.get("A_is_sorted", False) or bound.arguments.get("C_is_sorted", False):
        raise NotImplementedError("Sorted MoE buffers require a transformed validity mask.")
    if get_ep_group().world_size > 1:
        raise NotImplementedError("Expert dispatch requires locally mapped row validity.")
    topk_weights = bound.arguments.get("topk_weights")
    if topk_weights is None or topk_weights.ndim != 2:
        raise NotImplementedError("MoE validity requires token-major top-k assignments.")
    if topk_weights.shape[0] != validity.row_capacity:
        raise ValueError("MoE assignments do not match the runner token layout.")
    return _moe_row_validities(
        validity,
        a,
        c,
        topk_weights.shape[1],
        second_layer=second_layer,
        need_input=need_input,
        need_output=need_output,
    )


def _moe_row_validities(validity, inputs, outputs, top_k, *, second_layer, need_input, need_output):
    expanded = (
        _repeat_rows(validity, top_k) if need_output or (need_input and second_layer) else None
    )
    input_validity = (expanded if second_layer else validity) if need_input else None
    output_validity = expanded if need_output else None
    if input_validity is not None:
        input_validity.validate(inputs)
    if output_validity is not None:
        output_validity.validate(outputs)
    return input_validity, output_validity
