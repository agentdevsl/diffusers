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

from diffusers import UNet2DModel


def test_offload_state_dict_loads_cpu_staged_weights(tmp_path):
    torch.manual_seed(0)
    model = UNet2DModel(
        block_out_channels=(4, 8),
        layers_per_block=1,
        norm_num_groups=4,
        sample_size=8,
        in_channels=3,
        out_channels=3,
        down_block_types=("DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D"),
    )
    sample = torch.randn(1, 3, 8, 8)
    reference_output = model(sample).sample
    reference_weight = model.conv_in.weight.detach().clone()

    model.save_pretrained(tmp_path)

    loaded = UNet2DModel.from_pretrained(
        tmp_path,
        device_map={"": "cpu"},
        offload_state_dict=True,
        low_cpu_mem_usage=True,
    )

    assert torch.allclose(loaded.conv_in.weight, reference_weight)
    loaded_output = loaded(sample).sample
    assert torch.allclose(loaded_output, reference_output)
