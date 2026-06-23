# Copyright 2025 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from diffusers import TeaCacheConfig, apply_teacache
from diffusers.hooks._helpers import TransformerBlockMetadata, TransformerBlockRegistry
from diffusers.hooks.teacache import (
    _MODEL_CONFIG,
    _TEACACHE_BLOCK_HOOK,
    _TEACACHE_LEADER_BLOCK_HOOK,
    FLUX_TEACACHE_COEFFICIENTS,
    TeaCacheState,
    _compute_relative_l1_distance,
    _evaluate_polynomial,
)
from diffusers.models import ModelMixin
from diffusers.utils import logging


logger = logging.get_logger(__name__)


class DummyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, encoder_hidden_states=None, **kwargs):
        return hidden_states * 2.0


class DummyTransformer(ModelMixin):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([DummyBlock(), DummyBlock()])

    def forward(self, hidden_states, encoder_hidden_states=None):
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, encoder_hidden_states=encoder_hidden_states)
        return hidden_states


class TupleOutputBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, encoder_hidden_states=None, **kwargs):
        return hidden_states * 2.0, encoder_hidden_states


class TupleTransformer(ModelMixin):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([TupleOutputBlock()])

    def forward(self, hidden_states, encoder_hidden_states=None):
        for block in self.transformer_blocks:
            output = block(hidden_states, encoder_hidden_states=encoder_hidden_states)
            hidden_states = output[0]
            encoder_hidden_states = output[1]
        return hidden_states, encoder_hidden_states


class ThreeBlockTransformer(ModelMixin):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([DummyBlock(), DummyBlock(), DummyBlock()])

    def forward(self, hidden_states, encoder_hidden_states=None):
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, encoder_hidden_states=encoder_hidden_states)
        return hidden_states


class UnsupportedTransformer(ModelMixin):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([DummyBlock()])


def _identity_modulated_input_extractor(block, hidden_states, args, kwargs):
    return hidden_states


def _register_test_block(block_class, metadata):
    try:
        TransformerBlockRegistry.register(block_class, metadata)
    except ValueError:
        TransformerBlockRegistry._is_registered = True
        metadata._cls = block_class
        TransformerBlockRegistry._registry[block_class] = metadata


@pytest.fixture
def teacache_test_models():
    _register_test_block(
        DummyBlock,
        TransformerBlockMetadata(return_hidden_states_index=None, return_encoder_hidden_states_index=None),
    )
    _register_test_block(
        TupleOutputBlock,
        TransformerBlockMetadata(return_hidden_states_index=0, return_encoder_hidden_states_index=1),
    )

    _MODEL_CONFIG[DummyTransformer] = {
        "coefficients": FLUX_TEACACHE_COEFFICIENTS,
        "extract_modulated_input": _identity_modulated_input_extractor,
    }
    _MODEL_CONFIG[TupleTransformer] = {
        "coefficients": FLUX_TEACACHE_COEFFICIENTS,
        "extract_modulated_input": _identity_modulated_input_extractor,
    }
    _MODEL_CONFIG[ThreeBlockTransformer] = {
        "coefficients": FLUX_TEACACHE_COEFFICIENTS,
        "extract_modulated_input": _identity_modulated_input_extractor,
    }

    yield

    _MODEL_CONFIG.pop(DummyTransformer, None)
    _MODEL_CONFIG.pop(TupleTransformer, None)
    _MODEL_CONFIG.pop(ThreeBlockTransformer, None)


def test_apply_teacache_unsupported_model():
    model = UnsupportedTransformer()
    config = TeaCacheConfig(num_inference_steps=4)

    with pytest.raises(ValueError, match="TeaCache is not supported for UnsupportedTransformer"):
        apply_teacache(model, config)


def test_teacache_skipping_logic(teacache_test_models):
    model = DummyTransformer()
    config = TeaCacheConfig(rel_l1_thresh=100.0, num_inference_steps=3)

    apply_teacache(model, config)
    _set_context(model, "test_context")

    input_t0 = torch.tensor([[[10.0]]])
    output_t0 = model(input_t0)
    assert torch.allclose(output_t0, torch.tensor([[[40.0]]])), "Step 0 failed"

    input_t1 = torch.tensor([[[11.0]]])
    output_t1 = model(input_t1)
    assert torch.allclose(output_t1, torch.tensor([[[41.0]]])), f"Expected Skip (41.0), got {output_t1.item()}"


def test_teacache_first_last_always_compute(teacache_test_models):
    model = DummyTransformer()
    config = TeaCacheConfig(rel_l1_thresh=100.0, num_inference_steps=4)

    apply_teacache(model, config)
    _set_context(model, "test_context")

    input_t = torch.tensor([[[10.0]]])

    model(input_t)
    model(input_t)
    model(input_t)

    input_t3 = torch.tensor([[[13.0]]])
    output_t3 = model(input_t3)
    assert torch.allclose(output_t3, torch.tensor([[[52.0]]])), (
        f"Expected boundary compute (52.0), got {output_t3.item()}"
    )


def test_teacache_tuple_outputs(teacache_test_models):
    model = TupleTransformer()
    config = TeaCacheConfig(rel_l1_thresh=100.0, num_inference_steps=3)

    apply_teacache(model, config)
    _set_context(model, "test_context")

    input_t0 = torch.tensor([[[10.0]]])
    enc_t0 = torch.tensor([[[1.0]]])
    out_0, _ = model(input_t0, encoder_hidden_states=enc_t0)
    assert torch.allclose(out_0, torch.tensor([[[20.0]]]))

    input_t1 = torch.tensor([[[11.0]]])
    out_1, enc_1 = model(input_t1, encoder_hidden_states=enc_t0)

    assert torch.allclose(out_1, torch.tensor([[[21.0]]])), f"Tuple skip failed. Expected 21.0, got {out_1.item()}"
    assert torch.allclose(enc_1, enc_t0), "Encoder hidden states should be unchanged on skip"


def test_teacache_reset(teacache_test_models):
    model = DummyTransformer()
    config = TeaCacheConfig(rel_l1_thresh=100.0, num_inference_steps=3)
    apply_teacache(model, config)
    _set_context(model, "test_context")

    input_t = torch.ones(1, 1, 1)

    model(input_t)
    model(input_t)
    model(input_t)

    input_t2 = torch.tensor([[[2.0]]])
    output_t2 = model(input_t2)

    assert torch.allclose(output_t2, torch.tensor([[[8.0]]])), "State did not reset correctly"


def test_teacache_hook_reapplication(teacache_test_models):
    model = ThreeBlockTransformer()
    config = TeaCacheConfig(rel_l1_thresh=100.0, num_inference_steps=3)

    apply_teacache(model, config)
    apply_teacache(model, config)

    leader_hooks = 0
    block_hooks = 0
    for module in model.modules():
        if hasattr(module, "_diffusers_hook"):
            if module._diffusers_hook.get_hook(_TEACACHE_LEADER_BLOCK_HOOK) is not None:
                leader_hooks += 1
            if module._diffusers_hook.get_hook(_TEACACHE_BLOCK_HOOK) is not None:
                block_hooks += 1

    assert leader_hooks == 1, "Head hook should be registered exactly once"
    assert block_hooks == 2, "Middle and tail block hooks should be registered without duplication"


def test_teacache_accumulator_reset_on_compute(teacache_test_models):
    model = DummyTransformer()
    config = TeaCacheConfig(rel_l1_thresh=0.0, num_inference_steps=4)

    apply_teacache(model, config)
    _set_context(model, "test_context")

    state = _get_teacache_state(model)

    model(torch.tensor([[[10.0]]]))
    assert state.accumulated_distance == 0.0

    model(torch.tensor([[[100.0]]]))
    assert state.accumulated_distance == 0.0
    assert state.should_compute is True


def _set_context(model, context_name):
    for module in model.modules():
        if hasattr(module, "_diffusers_hook"):
            module._diffusers_hook._set_context(context_name)


def _get_teacache_state(model):
    for module in model.modules():
        if hasattr(module, "_diffusers_hook"):
            hook = module._diffusers_hook.get_hook(_TEACACHE_BLOCK_HOOK)
            if hook is not None:
                return hook.state_manager.get_state()
            hook = module._diffusers_hook.get_hook(_TEACACHE_LEADER_BLOCK_HOOK)
            if hook is not None:
                return hook.state_manager.get_state()
    return None


def test_teacache_polynomial_rescaling():
    current = torch.tensor([[[1.1]]])
    previous = torch.tensor([[[1.0]]])
    rel_l1 = _compute_relative_l1_distance(current, previous)
    expected = _evaluate_polynomial(FLUX_TEACACHE_COEFFICIENTS, rel_l1)

    assert rel_l1 == pytest.approx(0.1)
    assert expected == pytest.approx(_evaluate_polynomial(FLUX_TEACACHE_COEFFICIENTS, 0.1))

    state = TeaCacheState()
    state.accumulated_distance += expected
    assert state.accumulated_distance == pytest.approx(expected)
