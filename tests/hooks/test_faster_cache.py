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

from diffusers.hooks import HookRegistry
from diffusers.hooks.faster_cache import FasterCacheBlockHook


class DummyAttention(torch.nn.Module):
    def forward(self, hidden_states):
        return hidden_states * 2.0


def test_guidance_distilled_first_forward_in_skip_range_does_not_crash():
    module = DummyAttention()
    hook = FasterCacheBlockHook(
        block_skip_range=2,
        timestep_skip_range=(0, 1000),
        is_guidance_distilled=True,
        weight_callback=lambda _: 0.5,
        current_timestep_callback=lambda: 500,
    )
    registry = HookRegistry.check_if_exists_or_initialize(module)
    registry.register_hook(hook, "faster_cache_block")
    hook.initialize_hook(module)

    hidden_states = torch.randn(2, 4)
    output = module(hidden_states)
    assert output.shape == hidden_states.shape

    output = module(hidden_states)
    assert output.shape == hidden_states.shape
