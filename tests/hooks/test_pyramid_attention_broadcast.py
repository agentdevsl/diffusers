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

import torch

from diffusers.hooks.pyramid_attention_broadcast import _apply_pyramid_attention_broadcast_hook
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.modeling_utils import ModelMixin

from ..testing_utils import torch_device


class CountingBlock(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(dim, dim)
        self.call_count = 0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.call_count += 1
        return self.proj(hidden_states)


class DummyTransformer(ModelMixin, CacheMixin, torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([CountingBlock(dim)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states)
        return hidden_states


def test_pab_isolates_state_between_cond_and_uncond_contexts():
    """PAB must not reuse cond attention cache during the uncond forward pass."""
    dim = 4
    model = DummyTransformer(dim).to(torch_device)
    block = model.transformer_blocks[0]

    _apply_pyramid_attention_broadcast_hook(
        block,
        timestep_skip_range=(100, 800),
        block_skip_range=2,
        current_timestep_callback=lambda: 500,
    )

    x = torch.randn(1, dim, device=torch_device)

    with model.cache_context("cond"):
        model(x)

    assert block.call_count == 1

    with model.cache_context("uncond"):
        model(x)

    assert block.call_count == 2, (
        "Uncond pass must compute attention instead of reusing cond cache from a shared hook state."
    )
