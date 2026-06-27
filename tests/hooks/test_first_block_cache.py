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

from diffusers.hooks import FirstBlockCacheConfig, apply_first_block_cache
from diffusers.hooks._helpers import TransformerBlockMetadata, TransformerBlockRegistry
from diffusers.models import ModelMixin
from diffusers.models.cache_utils import CacheMixin


class DummyBlock(torch.nn.Module):
    def forward(self, hidden_states, encoder_hidden_states=None, **kwargs):
        return hidden_states * 2.0


class DummyTransformer(ModelMixin, CacheMixin):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([DummyBlock(), DummyBlock(), DummyBlock()])

    def forward(self, hidden_states, encoder_hidden_states=None):
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, encoder_hidden_states=encoder_hidden_states)
        return hidden_states


@pytest.fixture(autouse=True)
def register_dummy_blocks():
    TransformerBlockRegistry.register(
        DummyBlock,
        TransformerBlockMetadata(return_hidden_states_index=None, return_encoder_hidden_states_index=None),
    )


def test_first_block_cache_requires_cache_context():
    model = DummyTransformer()
    apply_first_block_cache(model, FirstBlockCacheConfig(threshold=0.2))

    with pytest.raises(ValueError, match="No context is set"):
        model(torch.tensor([[[1.0]]]))


def test_first_block_cache_forward_with_cache_context():
    model = DummyTransformer()
    apply_first_block_cache(model, FirstBlockCacheConfig(threshold=0.2))

    with model.cache_context("inference"):
        output = model(torch.tensor([[[1.0]]]))

    assert output.shape == torch.Size([1, 1, 1])
