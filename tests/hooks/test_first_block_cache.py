# Copyright 2026 The HuggingFace Team. All rights reserved.
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

from diffusers.hooks._helpers import TransformerBlockMetadata, TransformerBlockRegistry
from diffusers.hooks.first_block_cache import FirstBlockCacheConfig, apply_first_block_cache
from diffusers.models import ModelMixin


class DummyBlock(torch.nn.Module):
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
    def forward(self, hidden_states, encoder_hidden_states=None, **kwargs):
        return hidden_states * 2.0, encoder_hidden_states


class TupleTransformer(ModelMixin):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([TupleOutputBlock(), TupleOutputBlock()])

    def forward(self, hidden_states, encoder_hidden_states=None):
        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states = block(hidden_states, encoder_hidden_states=encoder_hidden_states)
        return hidden_states, encoder_hidden_states


def _set_context(model, context_name):
    for module in model.modules():
        if hasattr(module, "_diffusers_hook"):
            module._diffusers_hook._set_context(context_name)


@pytest.fixture(autouse=True)
def register_dummy_blocks():
    TransformerBlockRegistry.register(
        DummyBlock,
        TransformerBlockMetadata(return_hidden_states_index=None, return_encoder_hidden_states_index=None),
    )
    TransformerBlockRegistry.register(
        TupleOutputBlock,
        TransformerBlockMetadata(return_hidden_states_index=0, return_encoder_hidden_states_index=1),
    )


def test_first_block_cache_skips_when_residual_is_stable():
    """When head-block residuals are similar, tail blocks should be skipped."""
    model = DummyTransformer()
    apply_first_block_cache(model, FirstBlockCacheConfig(threshold=0.05))
    _set_context(model, "test_context")

    input_t0 = torch.tensor([[[10.0]]])
    output_t0 = model(input_t0)
    assert torch.allclose(output_t0, torch.tensor([[[40.0]]]))

    # Identical input -> residual diff is 0 -> skip tail block (40.0, not 44.0).
    output_t1 = model(input_t0)
    assert torch.allclose(output_t1, torch.tensor([[[40.0]]]))


def test_first_block_cache_recomputes_when_residual_changes():
    """When residuals exceed the threshold, the full block stack must run."""
    model = DummyTransformer()
    apply_first_block_cache(model, FirstBlockCacheConfig(threshold=0.05))
    _set_context(model, "test_context")

    model(torch.tensor([[[10.0]]]))

    output_t1 = model(torch.tensor([[[11.0]]]))
    assert torch.allclose(output_t1, torch.tensor([[[44.0]]]))


def test_first_block_cache_tuple_outputs():
    """First Block Cache must support tuple block outputs (Flux-style)."""
    model = TupleTransformer()
    apply_first_block_cache(model, FirstBlockCacheConfig(threshold=0.05))
    _set_context(model, "test_context")

    input_t0 = torch.tensor([[[10.0]]])
    enc_t0 = torch.tensor([[[1.0]]])
    out_0, _ = model(input_t0, encoder_hidden_states=enc_t0)
    assert torch.allclose(out_0, torch.tensor([[[40.0]]]))

    out_1, _ = model(input_t0, encoder_hidden_states=enc_t0)
    assert torch.allclose(out_1, torch.tensor([[[40.0]]]))


def test_first_block_cache_recomputes_after_skip_when_input_changes():
    """A large input change after a cached step must trigger full recomputation."""
    model = DummyTransformer()
    apply_first_block_cache(model, FirstBlockCacheConfig(threshold=0.05))
    _set_context(model, "test_context")

    model(torch.tensor([[[10.0]]]))
    model(torch.tensor([[[10.0]]]))

    output = model(torch.tensor([[[12.0]]]))
    assert torch.allclose(output, torch.tensor([[[48.0]]]))
