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

from diffusers.hooks.faster_cache import FasterCacheConfig, apply_faster_cache
from diffusers.models import ModelMixin


class DummyTransformer(ModelMixin):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, hidden_states):
        return self.linear(hidden_states)


def test_apply_faster_cache_requires_timestep_callback():
    model = DummyTransformer()
    config = FasterCacheConfig(spatial_attention_block_skip_range=2)

    with pytest.raises(ValueError, match="current_timestep_callback"):
        apply_faster_cache(model, config)
