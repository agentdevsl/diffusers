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

from unittest.mock import MagicMock

import torch

from diffusers.hooks.hooks import HookRegistry
from diffusers.hooks.pyramid_attention_broadcast import (
    _PYRAMID_ATTENTION_BROADCAST_HOOK,
    PyramidAttentionBroadcastConfig,
    PyramidAttentionBroadcastHook,
    PyramidAttentionBroadcastState,
)
from diffusers.modular_pipelines.modular_pipeline import ModularPipeline


def test_modular_pipeline_resets_stateful_caches_on_components():
    transformer = MagicMock()
    transformer._reset_stateful_cache = MagicMock()

    pipeline = MagicMock()
    pipeline.components = {"transformer": transformer}

    ModularPipeline._reset_stateful_caches(pipeline)

    transformer._reset_stateful_cache.assert_called_once()


def test_modular_pipeline_call_resets_pab_hook_state():
    class AttentionModule(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states + 1.0

    class Transformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer_blocks = torch.nn.ModuleList([AttentionModule()])

        def _reset_stateful_cache(self, recurse=True):
            HookRegistry.check_if_exists_or_initialize(self).reset_stateful_hooks(recurse=recurse)

    transformer = Transformer()
    config = PyramidAttentionBroadcastConfig(
        spatial_attention_block_skip_range=2,
        spatial_attention_timestep_skip_range=(0, 1000),
        current_timestep_callback=lambda: 500,
    )
    hook = PyramidAttentionBroadcastHook(
        timestep_skip_range=config.spatial_attention_timestep_skip_range,
        block_skip_range=config.spatial_attention_block_skip_range,
        current_timestep_callback=config.current_timestep_callback,
    )
    hook.state = PyramidAttentionBroadcastState()
    HookRegistry.check_if_exists_or_initialize(transformer.transformer_blocks[0]).register_hook(
        hook, _PYRAMID_ATTENTION_BROADCAST_HOOK
    )

    hidden_states = torch.tensor([1.0])
    transformer.transformer_blocks[0](hidden_states)
    assert hook.state.iteration == 1

    pipeline = MagicMock()
    pipeline.components = {"transformer": transformer}
    ModularPipeline._reset_stateful_caches(pipeline)

    assert hook.state.iteration == 0
    assert hook.state.cache is None
