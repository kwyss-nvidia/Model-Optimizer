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

from ..nn import TensorValidity

__all__ = ["validity_from_forward_context"]


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
