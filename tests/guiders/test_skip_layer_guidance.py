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

from diffusers.guiders.skip_layer_guidance import SkipLayerGuidance
from diffusers.hooks import LayerSkipConfig


def _slg_guider(guidance_scale: float = 1.0) -> SkipLayerGuidance:
    guider = SkipLayerGuidance(
        guidance_scale=guidance_scale,
        skip_layer_guidance_scale=2.8,
        skip_layer_config=LayerSkipConfig(indices=[7], fqn="auto"),
    )
    guider.set_state(step=2, num_inference_steps=20, timestep=torch.tensor([500]))
    return guider


def test_skip_layer_guidance_applies_hooks_on_second_batch_without_cfg():
    guider = _slg_guider(guidance_scale=1.0)

    guider._count_prepared = 1
    assert not guider._should_apply_skip_hooks()

    guider._count_prepared = 2
    assert guider._should_apply_skip_hooks()


def test_skip_layer_guidance_applies_hooks_on_third_batch_with_cfg():
    guider = _slg_guider(guidance_scale=7.5)

    guider._count_prepared = 1
    assert not guider._should_apply_skip_hooks()

    guider._count_prepared = 2
    assert not guider._should_apply_skip_hooks()

    guider._count_prepared = 3
    assert guider._should_apply_skip_hooks()
