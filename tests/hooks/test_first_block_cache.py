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


def test_first_block_cache_skipping_logic():
    """
    FirstBlockCache skips tail blocks when the first-block residual change is below threshold.
    """
    model = DummyTransformer()
    apply_first_block_cache(model, FirstBlockCacheConfig(threshold=1.0))

    # Step 0: Input 10.0 -> Output 40.0 (2 blocks * 2x each). Head residual = 10, tail residual = 20.
    input_t0 = torch.tensor([[[10.0]]])
    output_t0 = model(input_t0)
    assert torch.allclose(output_t0, torch.tensor([[[40.0]]])), "Step 0 failed"

    # Step 1: Input 11.0.
    # If skipped: head output 22 + tail residual 20 = 42.0
    # If computed: 11 * 4 = 44.0
    input_t1 = torch.tensor([[[11.0]]])
    output_t1 = model(input_t1)
    assert torch.allclose(output_t1, torch.tensor([[[42.0]]])), f"Expected skip (42.0), got {output_t1.item()}"


def test_first_block_cache_compute_when_residual_changes():
    """A low threshold forces full recomputation when the first-block residual shifts."""
    model = DummyTransformer()
    apply_first_block_cache(model, FirstBlockCacheConfig(threshold=0.01))

    model(torch.tensor([[[10.0]]]))

    input_t1 = torch.tensor([[[11.0]]])
    output_t1 = model(input_t1)
    assert torch.allclose(output_t1, torch.tensor([[[44.0]]])), (
        f"Expected compute (44.0) due to low threshold, got {output_t1.item()}"
    )


def test_first_block_cache_tuple_outputs():
    """Test compatibility with models returning (hidden, encoder_hidden) like Flux."""
    model = TupleTransformer()
    apply_first_block_cache(model, FirstBlockCacheConfig(threshold=1.0))

    input_t0 = torch.tensor([[[10.0]]])
    enc_t0 = torch.tensor([[[1.0]]])
    out_0, _ = model(input_t0, encoder_hidden_states=enc_t0)
    assert torch.allclose(out_0, torch.tensor([[[40.0]]]))

    # Step 1: Skip. Input 11.0 -> head output 22 + tail residual 20 = 42.0
    input_t1 = torch.tensor([[[11.0]]])
    out_1, _ = model(input_t1, encoder_hidden_states=enc_t0)
    assert torch.allclose(out_1, torch.tensor([[[42.0]]])), f"Tuple skip failed. Expected 42.0, got {out_1.item()}"


def test_first_block_cache_first_pass_always_computes():
    """The first forward pass must always run all blocks to populate tail residuals."""
    model = DummyTransformer()
    apply_first_block_cache(model, FirstBlockCacheConfig(threshold=1.0))

    input_t0 = torch.tensor([[[5.0]]])
    output_t0 = model(input_t0)
    assert torch.allclose(output_t0, torch.tensor([[[20.0]]]))


def test_first_block_cache_large_input_change_recomputes():
    """Large input changes exceed the threshold and trigger full recomputation."""
    model = DummyTransformer()
    apply_first_block_cache(model, FirstBlockCacheConfig(threshold=0.05))

    model(torch.tensor([[[10.0]]]))

    input_t1 = torch.tensor([[[20.0]]])
    output_t1 = model(input_t1)
    assert torch.allclose(output_t1, torch.tensor([[[80.0]]])), (
        f"Expected full compute (80.0) for large input change, got {output_t1.item()}"
    )
