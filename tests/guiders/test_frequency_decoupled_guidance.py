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

from diffusers.guiders.frequency_decoupled_guidance import FrequencyDecoupledGuidance


pytest.importorskip("kornia")


def test_frequency_decoupled_guidance_disabled_level_uses_cond_pyramid():
    guider = FrequencyDecoupledGuidance(
        guidance_scales=[0.0, 7.5],
        use_original_formulation=True,
    )
    guider.set_state(step=0, num_inference_steps=10, timestep=torch.tensor([500]))
    pred_cond = torch.randn(1, 3, 32, 32)
    pred_uncond = torch.randn(1, 3, 32, 32)

    output = guider.forward(pred_cond, pred_uncond)

    assert output.pred is not None
    assert output.pred.shape == pred_cond.shape
