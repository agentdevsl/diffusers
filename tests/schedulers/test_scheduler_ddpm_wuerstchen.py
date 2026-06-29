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

from diffusers import DDPMWuerstchenScheduler
from diffusers.schedulers.scheduling_ddpm_wuerstchen import DDPMWuerstchenSchedulerOutput


class DDPMWuerstchenSchedulerTest(unittest.TestCase):
    """
    Contract tests for DDPMWuerstchenScheduler — Stable Cascade prior/decoder use float timesteps
    in [1, 0] rather than integer indices. Pipeline tests only exercised it indirectly.
    """

    scheduler_class = DDPMWuerstchenScheduler

    def get_default_config(self, **kwargs):
        config = {"scaler": 1.0, "s": 0.008}
        config.update(**kwargs)
        return config

    def test_set_timesteps_float_schedule(self):
        scheduler = self.scheduler_class(**self.get_default_config())
        for nfe in [1, 4, 10]:
            scheduler.set_timesteps(num_inference_steps=nfe)
            self.assertEqual(scheduler.timesteps.shape, (nfe + 1,))
            self.assertAlmostEqual(scheduler.timesteps[0].item(), 1.0, places=5)
            self.assertAlmostEqual(scheduler.timesteps[-1].item(), 0.0, places=5)

    def test_scaler_modifies_alpha_cumprod(self):
        default = self.scheduler_class(**self.get_default_config())
        scaled = self.scheduler_class(**self.get_default_config(scaler=2.0))
        t = torch.tensor([0.5])
        default_alpha = default._alpha_cumprod(t, device="cpu")
        scaled_alpha = scaled._alpha_cumprod(t, device="cpu")
        self.assertFalse(torch.allclose(default_alpha, scaled_alpha))

    def test_step_shape_preserved(self):
        scheduler = self.scheduler_class(**self.get_default_config())
        scheduler.set_timesteps(num_inference_steps=4)

        sample = torch.randn(2, 8, 4, 4)
        model_output = torch.randn_like(sample)
        timestep = scheduler.timesteps[0:1].expand(sample.shape[0])

        output = scheduler.step(model_output, timestep, sample)
        self.assertIsInstance(output, DDPMWuerstchenSchedulerOutput)
        self.assertEqual(output.prev_sample.shape, sample.shape)
        self.assertEqual(output.prev_sample.dtype, sample.dtype)

    def test_step_deterministic_with_generator(self):
        scheduler = self.scheduler_class(**self.get_default_config())
        scheduler.set_timesteps(num_inference_steps=4)

        sample = torch.randn(1, 4, 8, 8)
        model_output = torch.randn_like(sample)
        timestep = scheduler.timesteps[0:1]
        generator = torch.Generator().manual_seed(0)

        first = scheduler.step(model_output, timestep, sample, generator=generator).prev_sample
        generator = torch.Generator().manual_seed(0)
        second = scheduler.step(model_output, timestep, sample, generator=generator).prev_sample
        torch.testing.assert_close(first, second)

    def test_previous_timestep_advances_schedule(self):
        scheduler = self.scheduler_class(**self.get_default_config())
        scheduler.set_timesteps(num_inference_steps=4)
        current = scheduler.timesteps[0:1]
        prev = scheduler.previous_timestep(current)
        self.assertAlmostEqual(prev[0].item(), scheduler.timesteps[1].item(), places=5)

    def test_add_noise_interpolates_sample_and_noise(self):
        scheduler = self.scheduler_class(**self.get_default_config())
        scheduler.set_timesteps(num_inference_steps=4)

        sample = torch.zeros(1, 4, 4, 4)
        noise = torch.ones_like(sample)
        timesteps = scheduler.timesteps[0:1]
        mixed = scheduler.add_noise(sample, noise, timesteps)
        self.assertTrue(mixed.min() >= 0.0)
        self.assertTrue(mixed.max() <= 1.0)
        self.assertFalse(torch.allclose(mixed, sample))
        self.assertFalse(torch.allclose(mixed, noise))

    def test_full_denoising_loop(self):
        scheduler = self.scheduler_class(**self.get_default_config())
        scheduler.set_timesteps(num_inference_steps=4)
        generator = torch.Generator().manual_seed(0)

        sample = torch.randn(1, 4, 8, 8)
        for t in scheduler.timesteps[:-1]:
            batch_t = t.expand(sample.shape[0])
            sample = scheduler.step(
                torch.randn_like(sample), batch_t, sample, generator=generator
            ).prev_sample
        self.assertEqual(sample.shape, (1, 4, 8, 8))
