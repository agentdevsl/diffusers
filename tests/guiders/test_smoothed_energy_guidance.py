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

from unittest.mock import patch

import torch

from diffusers.guiders.smoothed_energy_guidance import SmoothedEnergyGuidance
from diffusers.models.attention import Attention
from diffusers.models.attention_processor import AttnProcessor


class DummyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn1 = Attention(
            query_dim=4,
            cross_attention_dim=None,
            heads=1,
            dim_head=4,
            processor=AttnProcessor(),
        )

    def forward(self, hidden_states):
        return self.attn1(hidden_states)


class DummyTransformer(torch.nn.Module):
    def __init__(self, num_blocks=10):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([DummyBlock() for _ in range(num_blocks)])


def test_seg_shorthand_layers_builds_list_indices():
    guider = SmoothedEnergyGuidance(
        guidance_scale=7.5,
        seg_guidance_scale=2.8,
        seg_guidance_layers=[7, 8, 9],
    )
    assert guider.seg_guidance_config[0].indices == [7]
    assert guider.seg_guidance_config[1].indices == [8]
    assert guider.seg_guidance_config[2].indices == [9]


def test_seg_prepare_models_increments_count_and_installs_hooks_on_third_pass():
    guider = SmoothedEnergyGuidance(
        guidance_scale=7.5,
        seg_guidance_scale=2.8,
        seg_guidance_layers=[7],
    )
    guider.set_state(step=10, num_inference_steps=50, timestep=torch.tensor([500]))

    denoiser = DummyTransformer(num_blocks=10)

    guider.prepare_models(denoiser)
    assert guider._count_prepared == 1

    guider.prepare_models(denoiser)
    assert guider._count_prepared == 2

    with patch("diffusers.guiders.smoothed_energy_guidance._apply_smoothed_energy_guidance_hook") as mock_apply:
        guider.prepare_models(denoiser)
        assert guider._count_prepared == 3
        mock_apply.assert_called_once()
