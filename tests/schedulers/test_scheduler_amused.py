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

from diffusers import AmusedScheduler
from diffusers.schedulers.scheduling_amused import (
    AmusedSchedulerOutput,
    gumbel_noise,
    mask_by_random_topk,
)


class AmusedSchedulerTest(unittest.TestCase):
    """
    Unit tests for AmusedScheduler masked-token denoising. This scheduler has a bespoke
    step API (discrete token IDs, Gumbel masking) and is not covered by SchedulerCommonTest.
    """

    mask_token_id = 0

    def get_default_config(self, **kwargs):
        config = {
            "mask_token_id": self.mask_token_id,
            "masking_schedule": "cosine",
        }
        config.update(**kwargs)
        return config

    def test_gumbel_noise_matches_input_shape(self):
        probs = torch.ones(2, 4)
        noise = gumbel_noise(probs)
        self.assertEqual(noise.shape, probs.shape)

    def test_gumbel_noise_is_reproducible_with_generator(self):
        probs = torch.ones(2, 4)
        generator = torch.Generator().manual_seed(0)
        noise_a = gumbel_noise(probs, generator=generator)
        generator = torch.Generator().manual_seed(0)
        noise_b = gumbel_noise(probs, generator=generator)
        self.assertTrue(torch.allclose(noise_a, noise_b))

    def test_mask_by_random_topk_masks_requested_count(self):
        probs = torch.tensor([[0.9, 0.8, 0.1, 0.05]])
        mask_len = torch.tensor([[2]])
        masking = mask_by_random_topk(mask_len, probs, temperature=0.0)
        self.assertEqual(masking.sum().item(), 2)

    def test_set_timesteps_builds_temperature_schedule(self):
        scheduler = AmusedScheduler(**self.get_default_config())
        scheduler.set_timesteps(num_inference_steps=4, temperature=(2, 0))
        self.assertEqual(scheduler.timesteps.shape, (4,))
        self.assertEqual(scheduler.temperatures.shape, (4,))
        self.assertAlmostEqual(scheduler.temperatures[0].item(), 2.0, places=5)
        self.assertAlmostEqual(scheduler.temperatures[-1].item(), 0.0, places=5)

    def test_step_final_timestep_unmasks_all_tokens(self):
        scheduler = AmusedScheduler(**self.get_default_config())
        scheduler.set_timesteps(num_inference_steps=2)

        sample = torch.tensor([[self.mask_token_id, self.mask_token_id, 5, 6]])
        model_output = torch.zeros(1, 4, 8)
        model_output[0, :, 3] = 10.0  # token id 3 wins for every position

        output = scheduler.step(model_output, timestep=0, sample=sample, generator=torch.Generator().manual_seed(0))
        self.assertIsInstance(output, AmusedSchedulerOutput)
        self.assertFalse((output.prev_sample == self.mask_token_id).any())

    def test_step_intermediate_timestep_keeps_some_masks(self):
        scheduler = AmusedScheduler(**self.get_default_config())
        scheduler.set_timesteps(num_inference_steps=4, temperature=0.0)

        sample = torch.full((1, 6), self.mask_token_id)
        model_output = torch.randn(1, 6, 4)

        timestep = scheduler.timesteps[1].item()
        output = scheduler.step(
            model_output,
            timestep=timestep,
            sample=sample,
            generator=torch.Generator().manual_seed(0),
        )
        self.assertTrue((output.prev_sample == self.mask_token_id).any())
        self.assertFalse((output.prev_sample == self.mask_token_id).all())

    def test_step_2d_input_is_reshaped_and_restored(self):
        scheduler = AmusedScheduler(**self.get_default_config())
        scheduler.set_timesteps(num_inference_steps=2)

        sample = torch.full((1, 2, 2), self.mask_token_id)
        model_output = torch.zeros(1, 4, 2, 2)
        model_output[:, :, 0, 0] = 10.0

        output = scheduler.step(model_output, timestep=0, sample=sample, generator=torch.Generator().manual_seed(0))
        self.assertEqual(output.prev_sample.shape, (1, 2, 2))

    def test_add_noise_respects_mask_ratio(self):
        scheduler = AmusedScheduler(**self.get_default_config(masking_schedule="linear"))
        scheduler.set_timesteps(num_inference_steps=4)

        sample = torch.arange(1, 17).reshape(1, 4, 4)
        # Early timesteps mask most tokens; use the second step for a partial mask.
        timestep = scheduler.timesteps[1].item()
        masked = scheduler.add_noise(sample, timesteps=timestep, generator=torch.Generator().manual_seed(0))
        self.assertTrue((masked == self.mask_token_id).any())
        self.assertFalse((masked == self.mask_token_id).all())

    def test_unknown_masking_schedule_raises_in_step(self):
        scheduler = AmusedScheduler(**self.get_default_config(masking_schedule="invalid"))
        scheduler.set_timesteps(num_inference_steps=4)

        sample = torch.tensor([[self.mask_token_id, 1, 2, 3]])
        model_output = torch.randn(1, 4, 4)
        with self.assertRaises(ValueError):
            scheduler.step(model_output, timestep=scheduler.timesteps[1].item(), sample=sample)

    def test_unknown_masking_schedule_raises_in_add_noise(self):
        scheduler = AmusedScheduler(**self.get_default_config(masking_schedule="invalid"))
        scheduler.set_timesteps(num_inference_steps=2)

        sample = torch.tensor([[1, 2, 3, 4]])
        with self.assertRaises(ValueError):
            scheduler.add_noise(sample, timesteps=scheduler.timesteps[0].item())
