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

from diffusers.hooks.taylorseer_cache import TaylorSeerCacheConfig, apply_taylorseer_cache


def test_taylorseer_forward_without_cache_context():
    module = torch.nn.Linear(4, 4, bias=False)
    config = TaylorSeerCacheConfig(
        cache_interval=2,
        disable_cache_before_step=1,
        max_order=1,
        cache_identifiers=["^$"],
    )
    apply_taylorseer_cache(module, config)

    x = torch.randn(1, 4)
    output = module(x)

    assert output.shape == x.shape
