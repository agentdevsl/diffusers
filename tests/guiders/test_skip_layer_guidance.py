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

import unittest

from diffusers.guiders.auto_guidance import AutoGuidance
from diffusers.guiders.skip_layer_guidance import SkipLayerGuidance


class SkipLayerGuidanceConfigTest(unittest.TestCase):
    def test_shorthand_layers_wrap_each_index_in_a_list(self):
        guider = SkipLayerGuidance(skip_layer_guidance_layers=[7, 8, 9])
        self.assertEqual([config.indices for config in guider.skip_layer_config], [[7], [8], [9]])

    def test_single_int_layer_shorthand(self):
        guider = SkipLayerGuidance(skip_layer_guidance_layers=7)
        self.assertEqual([config.indices for config in guider.skip_layer_config], [[7]])


class AutoGuidanceConfigTest(unittest.TestCase):
    def test_shorthand_layers_wrap_each_index_in_a_list(self):
        guider = AutoGuidance(auto_guidance_layers=[3, 4])
        self.assertEqual([config.indices for config in guider.auto_guidance_config], [[3], [4]])
