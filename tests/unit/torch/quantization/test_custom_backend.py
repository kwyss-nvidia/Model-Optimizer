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

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for registering and using a custom quantization backend."""

from types import SimpleNamespace

import pytest
import torch

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.nn import (
    SequentialQuantizer,
    TensorQuantizer,
    TensorQuantizerCache,
    TensorValidity,
    quantize_with_validity,
    register_quant_backend,
    unregister_quant_backend,
)
from modelopt.torch.quantization.plugins.vllm_validity import (
    _moe_row_validities,
    validity_from_forward_context,
)


def test_custom_backend_via_quantize():
    # Define and register a simple dummy backend that adds a constant to inputs
    def dummy_backend(inputs: torch.Tensor, tq) -> torch.Tensor:
        extra = getattr(tq, "backend_extra_args", None) or {}
        offset = extra.get("offset", 1.0)
        return inputs + offset

    register_quant_backend("dummy_backend", dummy_backend)

    model = torch.nn.Linear(16, 16, bias=False)

    cfg = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "*weight_quantizer",
                "cfg": {
                    "num_bits": 8,
                    "axis": None,
                    "backend": "dummy_backend",
                    "backend_extra_args": {"offset": 2.5},
                },
                "enable": True,
            },
        ],
        "algorithm": "max",
    }

    inputs = torch.randn(1, 16)

    def forward_loop(m):
        m(inputs)

    mtq.quantize(model, cfg, forward_loop=forward_loop)
    output_test = model(inputs)

    assert torch.allclose(output_test, inputs @ (model.weight.T + 2.5))

    # Unregister the backend to avoid impacting other tests
    unregister_quant_backend("dummy_backend")


def test_custom_backend_with_quantizer_cache():
    """Test that TensorQuantizerCache is initialized on first forward and not saved/restored."""

    # Simple cache class
    class DummyCache(TensorQuantizerCache):
        def __init__(self):
            super().__init__()
            self.value = 3.0

    # Backend that creates cache on first call
    def cached_backend(inputs: torch.Tensor, tq: TensorQuantizer) -> torch.Tensor:
        if tq._quantizer_cache is None:
            tq._quantizer_cache = DummyCache()
        return inputs + tq._quantizer_cache.value

    register_quant_backend("cached_backend", cached_backend)

    model = torch.nn.Linear(16, 16, bias=False)
    cfg = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "*weight_quantizer",
                "cfg": {"backend": "cached_backend"},
                "enable": True,
            },
        ],
        "algorithm": "max",
    }
    inputs = torch.randn(1, 16)

    mtq.quantize(model, cfg, forward_loop=lambda m: m(inputs))

    # Cache is None before first forward
    assert model.weight_quantizer._quantizer_cache is None

    # Cache gets initialized on forward
    model(inputs)
    assert isinstance(model.weight_quantizer._quantizer_cache, DummyCache)

    # Save modelopt state
    modelopt_state = mto.modelopt_state(model)

    # Cache is not saved in modelopt_state
    assert not any("_quantizer_cache" in key for key in modelopt_state)

    # Restore to a new model
    model_restored = torch.nn.Linear(16, 16, bias=False)
    mto.restore_from_modelopt_state(model_restored, modelopt_state)
    model_restored.load_state_dict(model.state_dict())

    # Cache is None after restore
    assert model_restored.weight_quantizer._quantizer_cache is None

    # Cache gets re-initialized on forward
    model_restored(inputs)
    assert isinstance(model_restored.weight_quantizer._quantizer_cache, DummyCache)

    unregister_quant_backend("cached_backend")


@pytest.mark.parametrize("prefix", [True, False])
def test_validity_backend_padding(prefix):
    """Padding outliers cannot affect valid rows; a changing count is read on every call."""
    x = torch.tensor([[0.3, -1.0], [1.0e4, -1.0e4], [0.7, -2.0]])
    count = torch.tensor(1, dtype=torch.int64)
    mask = torch.tensor([False, True, False])
    context = SimpleNamespace(
        num_valid_tokens=count, is_padding=mask, valid_tokens_are_prefix=prefix, token_capacity=3
    )
    validity = validity_from_forward_context(context)
    assert validity.num_valid_rows is count
    assert validity.is_padding is mask

    def backend(inputs, quantizer, *, validity=None):
        if validity is not None:
            padding = (
                torch.arange(validity.row_capacity, device=inputs.device) >= validity.num_valid_rows
                if validity.valid_rows_are_prefix
                else validity.is_padding
            )
            inputs = inputs.masked_fill(padding[:, None], 0)
        return inputs / inputs.abs().amax().clamp_min(1e-12)

    register_quant_backend("validity_test", backend, supports_validity=True)
    try:
        for quantizer in (
            TensorQuantizer(QuantizerAttributeConfig(axis=None, backend="validity_test")),
        ):
            expected_rows = torch.tensor([0]) if prefix else torch.tensor([0, 2])
            result = quantize_with_validity(quantizer, x, validity)
            torch.testing.assert_close(result[expected_rows], quantizer(x[expected_rows]))
            if prefix:
                count.fill_(3)
                torch.testing.assert_close(quantizer(x, validity=validity), quantizer(x))
                count.fill_(1)
            assert all("validity" not in k for k in quantizer.state_dict())
        native = TensorQuantizer(QuantizerAttributeConfig(axis=None))
        torch.testing.assert_close(native(x, validity=validity), native(x))
        seq = SequentialQuantizer(
            TensorQuantizer(QuantizerAttributeConfig(backend="validity_test"))
        )
        torch.testing.assert_close(seq(x, validity=validity), backend(x, None, validity=validity))
    finally:
        unregister_quant_backend("validity_test")


def test_legacy_backend_validity_opt_in():
    """Unsupported backends and subclasses retain their legacy behavior with validity."""

    def legacy(inputs, quantizer):
        return inputs + 1

    register_quant_backend("legacy_validity_test", legacy)
    try:
        q = TensorQuantizer(QuantizerAttributeConfig(backend="legacy_validity_test"))
        x = torch.ones(2, 4)
        validity = TensorValidity(torch.tensor(1), None, True, 2)
        torch.testing.assert_close(q(x), x + 1)
        torch.testing.assert_close(quantize_with_validity(q, x, validity), x + 1)
        torch.testing.assert_close(q(x, validity=validity), x + 1)

        class LegacyQuantizer(TensorQuantizer):
            def forward(self, inputs):
                return inputs + 2

        torch.testing.assert_close(quantize_with_validity(LegacyQuantizer(), x, validity), x + 2)
        seq = SequentialQuantizer(q, LegacyQuantizer())
        torch.testing.assert_close(seq(x, validity=validity), seq(x))
        q.disable()
        torch.testing.assert_close(quantize_with_validity(q, x, validity), x)
    finally:
        unregister_quant_backend("legacy_validity_test")


def test_validity_expansion_and_invalid_layout():
    """Token-major top-k expansion preserves holes; capacity and prefix alone are insufficient."""
    v = TensorValidity(torch.tensor(2), torch.tensor([False, True, False]), False, 3)
    x = torch.ones(3, 2, 4)
    expanded, _ = _moe_row_validities(
        v, x, None, 2, second_layer=True, need_input=True, need_output=False
    )
    expanded.validate(x)
    original, unused = _moe_row_validities(
        v, x[:, 0], None, 2, second_layer=False, need_input=True, need_output=False
    )
    assert original is v and unused is None
    _, output_only = _moe_row_validities(
        v, None, x, 2, second_layer=False, need_input=False, need_output=True
    )
    torch.testing.assert_close(output_only.is_padding, expanded.is_padding)
    expected = x.clone()
    expected[1] = 0
    torch.testing.assert_close(x.masked_fill(expanded.is_padding.reshape(3, 2, 1), 0), expected)
    q = TensorQuantizer(QuantizerAttributeConfig(axis=None))
    with pytest.raises(ValueError):
        v.validate(x)
    with pytest.raises(ValueError):
        TensorValidity(torch.tensor(4), None, False, 6).validate(x)
    q.enable_calib()
    torch.testing.assert_close(q(x, validity=expanded), q(x))
