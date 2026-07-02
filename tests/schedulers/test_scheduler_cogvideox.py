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

from diffusers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler
from diffusers.schedulers.scheduling_ddim_cogvideox import DDIMSchedulerOutput


class CogVideoXDDIMSchedulerTest(unittest.TestCase):
    """
    Contract tests for CogVideoXDDIMScheduler — used by CogView3+ and shares the SNR-shifted
    schedule with CogVideoX pipelines. No dedicated scheduler test file existed previously.
    """

    scheduler_class = CogVideoXDDIMScheduler

    def get_default_config(self, **kwargs):
        config = {
            "num_train_timesteps": 1000,
            "beta_start": 0.00085,
            "beta_end": 0.0120,
            "beta_schedule": "scaled_linear",
            "timestep_spacing": "leading",
        }
        config.update(**kwargs)
        return config

    def test_snr_shift_modifies_alphas_cumprod(self):
        shifted = self.scheduler_class(**self.get_default_config(snr_shift_scale=3.0))
        unshifted = self.scheduler_class(**self.get_default_config(snr_shift_scale=1.0))
        self.assertFalse(torch.allclose(shifted.alphas_cumprod, unshifted.alphas_cumprod))

    def test_set_timesteps_num_inference_steps_exceeds_train_timesteps_raises(self):
        scheduler = self.scheduler_class(**self.get_default_config())
        with self.assertRaises(ValueError):
            scheduler.set_timesteps(scheduler.config.num_train_timesteps + 1)

    def test_set_timesteps_produces_expected_count(self):
        scheduler = self.scheduler_class(**self.get_default_config())
        for nfe in [1, 4, 10, 50]:
            scheduler.set_timesteps(nfe)
            self.assertEqual(scheduler.num_inference_steps, nfe)
            self.assertEqual(scheduler.timesteps.shape, (nfe,))

    def test_step_shape_preserved(self):
        scheduler = self.scheduler_class(**self.get_default_config())
        scheduler.set_timesteps(num_inference_steps=4)

        sample = torch.randn(2, 4, 8, 8)
        model_output = torch.randn_like(sample)
        timestep = scheduler.timesteps[0]

        output = scheduler.step(model_output, timestep, sample, eta=0.0)
        self.assertIsInstance(output, DDIMSchedulerOutput)
        self.assertEqual(output.prev_sample.shape, sample.shape)
        self.assertEqual(output.prev_sample.dtype, sample.dtype)

    def test_step_deterministic_when_eta_zero(self):
        scheduler = self.scheduler_class(**self.get_default_config())
        scheduler.set_timesteps(num_inference_steps=4)

        sample = torch.randn(1, 4, 8, 8)
        model_output = torch.randn_like(sample)
        generator = torch.Generator().manual_seed(0)
        timestep = scheduler.timesteps[0]

        first = scheduler.step(model_output, timestep, sample, eta=0.0, generator=generator).prev_sample
        generator = torch.Generator().manual_seed(0)
        second = scheduler.step(model_output, timestep, sample, eta=0.0, generator=generator).prev_sample
        torch.testing.assert_close(first, second)

    def test_full_denoising_loop(self):
        scheduler = self.scheduler_class(**self.get_default_config())
        scheduler.set_timesteps(num_inference_steps=4)

        sample = torch.randn(1, 4, 8, 8)
        generator = torch.Generator().manual_seed(0)
        for t in scheduler.timesteps:
            sample = scheduler.step(
                torch.randn_like(sample), t, sample, eta=0.0, generator=generator
            ).prev_sample
        self.assertEqual(sample.shape, (1, 4, 8, 8))


class CogVideoXDPMSchedulerTest(unittest.TestCase):
    """
    Contract tests for CogVideoXDPMScheduler — CogVideoX pipelines branch on its multi-step API.
    """

    scheduler_class = CogVideoXDPMScheduler

    def get_default_config(self, **kwargs):
        config = {
            "num_train_timesteps": 1000,
            "beta_start": 0.00085,
            "beta_end": 0.0120,
            "beta_schedule": "scaled_linear",
            "timestep_spacing": "trailing",
        }
        config.update(**kwargs)
        return config

    def test_step_requires_set_timesteps(self):
        scheduler = self.scheduler_class(**self.get_default_config())
        sample = torch.randn(1, 4, 8, 8)
        with self.assertRaises(ValueError):
            scheduler.step(
                torch.randn_like(sample),
                None,
                scheduler.timesteps[0],
                None,
                sample,
                return_dict=False,
            )

    def test_set_timesteps_num_inference_steps_exceeds_train_timesteps_raises(self):
        scheduler = self.scheduler_class(**self.get_default_config())
        with self.assertRaises(ValueError):
            scheduler.set_timesteps(scheduler.config.num_train_timesteps + 1)

    def test_first_step_returns_pred_original_sample(self):
        scheduler = self.scheduler_class(**self.get_default_config())
        scheduler.set_timesteps(num_inference_steps=4)

        sample = torch.randn(1, 4, 8, 8)
        model_output = torch.randn_like(sample)
        timestep = scheduler.timesteps[0]

        prev_sample, pred_original_sample = scheduler.step(
            model_output,
            None,
            timestep,
            None,
            sample,
            return_dict=False,
        )
        self.assertEqual(prev_sample.shape, sample.shape)
        self.assertEqual(pred_original_sample.shape, sample.shape)

    def test_second_step_uses_old_pred_original_sample(self):
        scheduler = self.scheduler_class(**self.get_default_config())
        scheduler.set_timesteps(num_inference_steps=4)

        sample = torch.randn(1, 4, 8, 8)
        model_output = torch.randn_like(sample)
        generator = torch.Generator().manual_seed(0)

        _, old_pred = scheduler.step(
            model_output,
            None,
            scheduler.timesteps[0],
            None,
            sample,
            return_dict=False,
            generator=generator,
        )

        prev_sample, _ = scheduler.step(
            model_output,
            old_pred,
            scheduler.timesteps[1],
            scheduler.timesteps[0],
            sample,
            return_dict=False,
            generator=generator,
        )
        self.assertEqual(prev_sample.shape, sample.shape)

    def test_get_variables_and_get_mult(self):
        scheduler = self.scheduler_class(**self.get_default_config())
        scheduler.set_timesteps(num_inference_steps=4)

        t = scheduler.timesteps[1]
        t_prev = t - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
        t_back = scheduler.timesteps[0]

        alpha_prod_t = scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = scheduler.alphas_cumprod[t_prev]
        alpha_prod_t_back = scheduler.alphas_cumprod[t_back]

        h, r, lamb, lamb_next = scheduler.get_variables(alpha_prod_t, alpha_prod_t_prev, alpha_prod_t_back)
        mult = scheduler.get_mult(h, r, alpha_prod_t, alpha_prod_t_prev, alpha_prod_t_back)
        self.assertEqual(len(mult), 4)
        self.assertTrue(torch.isfinite(h))
        self.assertTrue(torch.isfinite(lamb))
