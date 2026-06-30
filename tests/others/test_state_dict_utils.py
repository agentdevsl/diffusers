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

import torch

from diffusers.utils.state_dict_utils import (
    StateDictType,
    convert_all_state_dict_to_peft,
    convert_state_dict,
    convert_state_dict_to_diffusers,
    convert_state_dict_to_kohya,
    convert_state_dict_to_peft,
    convert_unet_state_dict_to_peft,
    state_dict_all_zero,
)


class StateDictUtilsTest(unittest.TestCase):
    def test_convert_state_dict_applies_first_matching_pattern(self):
        state_dict = {"layer.processor.weight": torch.ones(1)}
        converted = convert_state_dict(state_dict, {".processor.": "."})
        self.assertIn("layer.weight", converted)
        self.assertNotIn("layer.processor.weight", converted)

    def test_convert_state_dict_to_peft_auto_infers_diffusers_old(self):
        state_dict = {
            "unet.down_blocks.0.attentions.0.to_out_lora.down.weight": torch.ones(2, 2),
            "unet.down_blocks.0.attentions.0.to_out_lora.up.weight": torch.ones(2, 2),
        }
        converted = convert_state_dict_to_peft(state_dict)
        self.assertIn("unet.down_blocks.0.attentions.0.out_proj.lora_A.weight", converted)

        state_dict = {
            "unet.down_blocks.0.attentions.0.to_q_lora.down.weight": torch.ones(2, 2),
            "unet.down_blocks.0.attentions.0.to_q_lora.up.weight": torch.ones(2, 2),
        }
        converted = convert_state_dict_to_peft(state_dict, original_type=StateDictType.DIFFUSERS_OLD)
        self.assertIn("unet.down_blocks.0.attentions.0.q_proj.lora_A.weight", converted)
        self.assertIn("unet.down_blocks.0.attentions.0.q_proj.lora_B.weight", converted)

    def test_convert_state_dict_to_peft_diffusers(self):
        state_dict = {
            "text_encoder.encoder.layers.0.self_attn.q_proj.lora_linear_layer.down.weight": torch.ones(2, 2),
            "text_encoder.encoder.layers.0.self_attn.q_proj.lora_linear_layer.up.weight": torch.ones(2, 2),
        }
        converted = convert_state_dict_to_peft(state_dict, original_type=StateDictType.DIFFUSERS)
        self.assertIn("text_encoder.encoder.layers.0.self_attn.q_proj.lora_A.weight", converted)
        self.assertIn("text_encoder.encoder.layers.0.self_attn.q_proj.lora_B.weight", converted)

    def test_convert_state_dict_to_diffusers_from_peft(self):
        state_dict = {
            "unet.down_blocks.0.attentions.0.to_q.lora_A.weight": torch.ones(2, 2),
            "unet.down_blocks.0.attentions.0.to_q.lora_B.weight": torch.ones(2, 2),
        }
        converted = convert_state_dict_to_diffusers(state_dict, original_type=StateDictType.PEFT)
        self.assertIn("unet.down_blocks.0.attentions.0.to_q.lora.down.weight", converted)
        self.assertIn("unet.down_blocks.0.attentions.0.to_q.lora.up.weight", converted)

    def test_convert_state_dict_to_diffusers_already_diffusers(self):
        state_dict = {
            "layer.lora_linear_layer.down.weight": torch.ones(2, 2),
            "layer.lora_linear_layer.up.weight": torch.ones(2, 2),
        }
        converted = convert_state_dict_to_diffusers(state_dict)
        self.assertIs(converted, state_dict)

    def test_convert_unet_state_dict_to_peft(self):
        state_dict = {
            "down_blocks.0.attentions.0.to_q_lora.down.weight": torch.ones(2, 2),
            "down_blocks.0.attentions.0.to_q_lora.up.weight": torch.ones(2, 2),
        }
        converted = convert_unet_state_dict_to_peft(state_dict)
        self.assertIn("down_blocks.0.attentions.0.to_q.lora_A.weight", converted)
        self.assertIn("down_blocks.0.attentions.0.to_q.lora_B.weight", converted)

    def test_convert_all_state_dict_to_peft_falls_back_to_unet(self):
        state_dict = {
            "down_blocks.0.attentions.0.to_q_lora.down.weight": torch.ones(2, 2),
            "down_blocks.0.attentions.0.to_q_lora.up.weight": torch.ones(2, 2),
        }
        converted = convert_all_state_dict_to_peft(state_dict)
        self.assertTrue(any("lora_A" in key or "lora_B" in key for key in converted))

    def test_state_dict_all_zero(self):
        state_dict = {
            "a": torch.zeros(2, 2),
            "b": torch.zeros(3),
        }
        self.assertTrue(state_dict_all_zero(state_dict))
        state_dict["b"] = torch.ones(3)
        self.assertFalse(state_dict_all_zero(state_dict))

    def test_state_dict_all_zero_with_filter(self):
        state_dict = {
            "lora.down": torch.zeros(2, 2),
            "bias": torch.ones(3),
        }
        self.assertTrue(state_dict_all_zero(state_dict, filter_str="lora"))

    def test_convert_state_dict_to_kohya_remaps_component_prefixes(self):
        state_dict = {
            "text_encoder.encoder.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(2, 2),
            "text_encoder.encoder.layers.0.self_attn.q_proj.lora_B.weight": torch.ones(2, 2),
            "text_encoder_2.encoder.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(2, 2),
            "text_encoder_2.encoder.layers.0.self_attn.q_proj.lora_B.weight": torch.ones(2, 2),
            "unet.down_blocks.0.attentions.0.to_q.lora_A.weight": torch.ones(2, 2),
            "unet.down_blocks.0.attentions.0.to_q.lora_B.weight": torch.ones(2, 2),
        }
        kohya = convert_state_dict_to_kohya(state_dict)
        self.assertTrue(any(key.startswith("lora_te1_") for key in kohya))
        self.assertTrue(any(key.startswith("lora_te2_") for key in kohya))
        self.assertTrue(any("lora_unet" in key for key in kohya))
        self.assertTrue(any("alpha" in key for key in kohya))

    def test_convert_state_dict_to_kohya_adds_alpha_for_down_weights(self):
        down_weight = torch.ones(4, 2)
        state_dict = {
            "unet.down_blocks.0.attentions.0.to_q.lora_A.weight": down_weight,
            "unet.down_blocks.0.attentions.0.to_q.lora_B.weight": torch.ones(2, 4),
        }
        kohya = convert_state_dict_to_kohya(state_dict)
        alpha_keys = [key for key in kohya if key.endswith(".alpha")]
        self.assertEqual(len(alpha_keys), 1)
        self.assertEqual(kohya[alpha_keys[0]].item(), len(down_weight))

    def test_convert_state_dict_to_peft_unsupported_type_raises(self):
        state_dict = {"layer.weight": torch.ones(2, 2)}
        with self.assertRaises(ValueError):
            convert_state_dict_to_peft(state_dict, original_type=StateDictType.PEFT)

    def test_peft_unet_round_trip_preserves_weight_values(self):
        weight_a = torch.randn(2, 4)
        weight_b = torch.randn(4, 2)
        peft_state_dict = {
            "unet.down_blocks.0.attentions.0.to_q.lora_A.weight": weight_a.clone(),
            "unet.down_blocks.0.attentions.0.to_q.lora_B.weight": weight_b.clone(),
        }
        diffusers_state_dict = convert_state_dict_to_diffusers(peft_state_dict, original_type=StateDictType.PEFT)
        self.assertIn("unet.down_blocks.0.attentions.0.to_q.lora.down.weight", diffusers_state_dict)
        self.assertTrue(
            torch.allclose(
                diffusers_state_dict["unet.down_blocks.0.attentions.0.to_q.lora.down.weight"],
                weight_a,
            )
        )
        self.assertTrue(
            torch.allclose(
                diffusers_state_dict["unet.down_blocks.0.attentions.0.to_q.lora.up.weight"],
                weight_b,
            )
        )
