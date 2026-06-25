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

from diffusers.guiders.perturbed_attention_guidance import PerturbedAttentionGuidance


class TestPerturbedAttentionGuidanceConfig:
    def test_perturbed_guidance_layers_int_shorthand_builds_list_indices(self):
        guider = PerturbedAttentionGuidance(
            guidance_scale=7.5,
            perturbed_guidance_layers=7,
            perturbed_guidance_scale=2.8,
        )

        assert guider.skip_layer_config[0].indices == [7]

    def test_perturbed_guidance_layers_list_shorthand_builds_list_indices(self):
        guider = PerturbedAttentionGuidance(
            guidance_scale=7.5,
            perturbed_guidance_layers=[7, 8, 9],
            perturbed_guidance_scale=2.8,
        )

        assert guider.skip_layer_config[0].indices == [7, 8, 9]
