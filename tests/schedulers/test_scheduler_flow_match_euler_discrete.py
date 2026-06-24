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

from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteSchedulerOutput


class FlowMatchEulerDiscreteSchedulerTest(unittest.TestCase):
    """
    Contract tests for FlowMatchEulerDiscreteScheduler — the default scheduler for SD3, Flux, Wan,
    Qwen-Image, and many other pipelines. Pipeline tests only smoke-test it; these tests lock down
    validation and stepping behavior that is easy to break silently.
    """

    scheduler_class = FlowMatchEulerDiscreteScheduler

    def get_default_config(self, **kwargs):
        config = {
            "num_train_timesteps": 1000,
            "shift": 1.0,
        }
        config.update(**kwargs)
        return config

    def test_instantiation_with_defaults(self):
        scheduler = self.scheduler_class(**self.get_default_config())
        self.assertEqual(scheduler.config.num_train_timesteps, 1000)
        self.assertEqual(scheduler.config.shift, 1.0)

    def test_mutually_exclusive_sigma_schedules_raises(self):
        with self.assertRaises(ValueError):
            self.scheduler_class(
                **self.get_default_config(use_karras_sigmas=True, use_exponential_sigmas=True),
            )

    def test_invalid_time_shift_type_raises(self):
        with self.assertRaises(ValueError):
            self.scheduler_class(**self.get_default_config(time_shift_type="quadratic"))

    def test_set_timesteps_endpoints(self):
        scheduler = self.scheduler_class(**self.get_default_config())
        for nfe in [1, 2, 4, 8, 16]:
            scheduler.set_timesteps(num_inference_steps=nfe)
            self.assertEqual(scheduler.timesteps.shape, (nfe,))
            self.assertEqual(scheduler.sigmas.shape, (nfe + 1,))
            self.assertAlmostEqual(scheduler.timesteps[0].item(), 1000.0, places=4)
            self.assertAlmostEqual(scheduler.sigmas[-1].item(), 0.0, places=4)

    def test_set_timesteps_dynamic_shifting_requires_mu(self):
        scheduler = self.scheduler_class(**self.get_default_config(use_dynamic_shifting=True))
        with self.assertRaises(ValueError):
            scheduler.set_timesteps(num_inference_steps=4)

    def test_set_timesteps_dynamic_shifting_with_mu(self):
        scheduler = self.scheduler_class(**self.get_default_config(use_dynamic_shifting=True))
        scheduler.set_timesteps(num_inference_steps=4, mu=0.5)
        self.assertEqual(scheduler.num_inference_steps, 4)

    def test_set_timesteps_sigmas_timesteps_length_mismatch_raises(self):
        scheduler = self.scheduler_class(**self.get_default_config())
        with self.assertRaises(ValueError):
            scheduler.set_timesteps(sigmas=[0.9, 0.5, 0.1], timesteps=[900.0, 500.0])

    def test_set_timesteps_custom_sigmas(self):
        scheduler = self.scheduler_class(**self.get_default_config(shift=1.0))
        custom = [0.9, 0.7, 0.4, 0.1]
        scheduler.set_timesteps(sigmas=custom)
        self.assertEqual(scheduler.num_inference_steps, 4)
        self.assertEqual(scheduler.timesteps.shape, (4,))
        self.assertEqual(scheduler.sigmas.shape, (5,))
        self.assertAlmostEqual(scheduler.sigmas[-1].item(), 0.0, places=6)

    def test_step_shape_preserved(self):
        scheduler = self.scheduler_class(**self.get_default_config())
        scheduler.set_timesteps(num_inference_steps=4)

        sample = torch.randn(2, 16, 8, 8)
        model_output = torch.randn_like(sample)
        timestep = scheduler.timesteps[0:1]

        output = scheduler.step(model_output, timestep, sample)
        self.assertIsInstance(output, FlowMatchEulerDiscreteSchedulerOutput)
        self.assertEqual(output.prev_sample.shape, sample.shape)
        self.assertEqual(output.prev_sample.dtype, model_output.dtype)

    def test_step_rejects_integer_timestep_index(self):
        scheduler = self.scheduler_class(**self.get_default_config())
        scheduler.set_timesteps(num_inference_steps=4)

        sample = torch.randn(1, 4, 4, 4)
        model_output = torch.randn_like(sample)

        with self.assertRaises(ValueError):
            scheduler.step(model_output, 0, sample)

        with self.assertRaises(ValueError):
            scheduler.step(model_output, torch.tensor([0], dtype=torch.long), sample)

    def test_step_stochastic_sampling(self):
        scheduler = self.scheduler_class(**self.get_default_config(stochastic_sampling=True))
        scheduler.set_timesteps(num_inference_steps=4)

        sample = torch.randn(1, 4, 4, 4)
        model_output = torch.randn_like(sample)
        generator = torch.Generator().manual_seed(0)

        prev_sample = scheduler.step(
            model_output,
            scheduler.timesteps[0:1],
            sample,
            generator=generator,
        ).prev_sample
        self.assertEqual(prev_sample.shape, sample.shape)

    def test_scale_noise_interpolates_sample_and_noise(self):
        scheduler = self.scheduler_class(**self.get_default_config())
        scheduler.set_timesteps(timesteps=[500.0, 250.0])
        sample = torch.zeros(1, 4, 4, 4)
        noise = torch.ones_like(sample)
        mid_t = scheduler.timesteps[0:1]
        mixed = scheduler.scale_noise(sample, mid_t, noise)
        torch.testing.assert_close(mixed, 0.5 * noise, atol=1e-4, rtol=1e-4)

    def test_index_for_timestep_duplicate_handling(self):
        scheduler = self.scheduler_class(**self.get_default_config())
        scheduler.set_timesteps(timesteps=[900.0, 500.0, 500.0, 100.0])
        duplicate = scheduler.timesteps[1]
        self.assertEqual(scheduler.index_for_timestep(duplicate), 2)

    def test_set_begin_index_anchors_step_index(self):
        scheduler = self.scheduler_class(**self.get_default_config())
        scheduler.set_timesteps(num_inference_steps=4)
        scheduler.set_begin_index(2)
        sample = torch.randn(1, 4, 4, 4)
        scheduler.step(torch.randn_like(sample), scheduler.timesteps[0], sample)
        self.assertEqual(scheduler.step_index, 3)

    def test_full_denoising_loop(self):
        scheduler = self.scheduler_class(**self.get_default_config())
        scheduler.set_timesteps(num_inference_steps=4)

        sample = torch.randn(1, 4, 8, 8)
        for t in scheduler.timesteps:
            sample = scheduler.step(torch.randn_like(sample), t, sample).prev_sample
        self.assertEqual(sample.shape, (1, 4, 8, 8))
