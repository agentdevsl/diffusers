# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import torch

from diffusers.utils.state_dict_utils import (
    StateDictType,
    convert_state_dict_to_diffusers,
    convert_state_dict_to_peft,
    convert_unet_state_dict_to_peft,
)


class StateDictUtilsTest(unittest.TestCase):
    """
    Unit tests for LoRA state-dict conversion utilities. Every LoRA load path goes through these
    functions; incorrect key remapping silently produces wrong weights.
    """

    def test_convert_diffusers_old_to_peft(self):
        weight = torch.randn(4, 4)
        state_dict = {
            "attn1.to_out_lora.down.weight": weight,
            "attn1.to_out_lora.up.weight": weight,
        }
        converted = convert_state_dict_to_peft(state_dict)
        self.assertIn("attn1.out_proj.lora_A.weight", converted)
        self.assertIn("attn1.out_proj.lora_B.weight", converted)
        torch.testing.assert_close(converted["attn1.out_proj.lora_A.weight"], weight)
        torch.testing.assert_close(converted["attn1.out_proj.lora_B.weight"], weight)

    def test_convert_diffusers_new_is_noop(self):
        weight = torch.randn(4, 4)
        state_dict = {
            "attn1.q_proj.lora_linear_layer.down.weight": weight,
            "attn1.q_proj.lora_linear_layer.up.weight": weight,
        }
        converted = convert_state_dict_to_diffusers(state_dict)
        self.assertEqual(set(converted.keys()), set(state_dict.keys()))

    def test_convert_peft_to_diffusers(self):
        weight = torch.randn(4, 4)
        state_dict = {
            "attn1.q_proj.lora_A.weight": weight,
            "attn1.q_proj.lora_B.weight": weight,
        }
        converted = convert_state_dict_to_diffusers(state_dict, original_type=StateDictType.PEFT)
        self.assertIn("attn1.q_proj.lora_linear_layer.down.weight", converted)
        self.assertIn("attn1.q_proj.lora_linear_layer.up.weight", converted)

    def test_convert_unet_state_dict_to_peft(self):
        weight = torch.randn(4, 4)
        state_dict = {
            "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q_lora.down.weight": weight,
            "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q_lora.up.weight": weight,
        }
        converted = convert_unet_state_dict_to_peft(state_dict)
        self.assertIn("down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.weight", converted)
        self.assertIn("down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_B.weight", converted)

    def test_convert_state_dict_to_peft_unrecognized_raises(self):
        state_dict = {"some.unrelated.weight": torch.randn(2, 2)}
        with self.assertRaises(ValueError):
            convert_state_dict_to_peft(state_dict)

    def test_convert_state_dict_to_diffusers_unrecognized_raises(self):
        state_dict = {"some.unrelated.weight": torch.randn(2, 2)}
        with self.assertRaises(ValueError):
            convert_state_dict_to_diffusers(state_dict)
