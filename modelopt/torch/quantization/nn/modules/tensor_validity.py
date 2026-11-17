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


"""Framework-independent row validity for activation quantization."""

from dataclasses import dataclass
from math import prod

import torch

__all__ = ["TensorValidity", "quantize_with_validity"]


@dataclass(frozen=True)
class TensorValidity:
    """Validity over flattened leading dimensions, with features on the last axis.

    Producers own and update device tensors before graph replay. Prefix validity is
    a static layout guarantee, never inferred from a device count. Invalid output
    rows are unspecified; consumers must preserve the input shape.
    """

    num_valid_rows: torch.Tensor | None = None
    is_padding: torch.Tensor | None = None
    valid_rows_are_prefix: bool = False
    row_capacity: int | None = None

    def validate(self, inputs: torch.Tensor) -> None:
        """Check layout metadata without reading device values or synchronizing."""
        if inputs.ndim < 2:
            raise ValueError("Row validity requires at least a row and feature axis.")
        rows = prod(inputs.shape[:-1])
        if self.row_capacity != rows:
            raise ValueError("Validity capacity does not match the activation row layout.")
        count = self.num_valid_rows
        if count is not None and (
            count.ndim != 0 or count.dtype != torch.int64 or count.device != inputs.device
        ):
            raise ValueError("Valid row count must be an int64 scalar on the input device.")
        if self.valid_rows_are_prefix:
            if count is None:
                raise ValueError("Prefix validity requires a valid row count.")
        elif self.is_padding is None:
            raise ValueError("Non-prefix validity requires a padding mask.")
        mask = self.is_padding
        if mask is not None and (
            mask.ndim != 1
            or mask.numel() != rows
            or mask.dtype != torch.bool
            or mask.device != inputs.device
        ):
            raise ValueError("Padding mask must be a boolean vector over input rows on its device.")


def quantize_with_validity(quantizer, inputs, validity: TensorValidity | None = None):
    """Pass validity to supporting quantizers, otherwise preserve the legacy call."""
    if (
        validity is None
        or not quantizer.is_enabled
        or not getattr(quantizer, "supports_validity", False)
    ):
        return quantizer(inputs)
    return quantizer(inputs, validity=validity)
