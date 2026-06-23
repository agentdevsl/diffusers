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
    Contract tests for FlowMatchEulerDiscreteScheduler — shared by SD3, Flux, Wan, and many
    flow-matching pipelines. No dedicated scheduler test file existed previously.
    """

    scheduler_class = FlowMatchEulerDiscreteScheduler

    def get_default_config(self, **kwargs):
        config = {
            "num_train_timesteps": 1000,
            "shift": 1.0,
        }
        config.update(**kwargs)
        return config

    def test_set_timesteps_endpoints(self):
        scheduler = self.scheduler_class(**self.get_default_config())
        for nfe in [1, 2, 4, 8, 16]:
            scheduler.set_timesteps(num_inference_steps=nfe)
            self.assertEqual(scheduler.timesteps.shape, (nfe,))
            self.assertEqual(scheduler.sigmas.shape, (nfe + 1,))
            self.assertAlmostEqual(scheduler.sigmas[-1].item(), 0.0, places=6)

    def test_set_timesteps_dynamic_shifting_requires_mu(self):
        scheduler = self.scheduler_class(**self.get_default_config(use_dynamic_shifting=True))
        with self.assertRaises(ValueError):
            scheduler.set_timesteps(num_inference_steps=4)

    def test_set_timesteps_custom_sigmas_and_timesteps_length_mismatch(self):
        scheduler = self.scheduler_class(**self.get_default_config())
        with self.assertRaises(ValueError):
            scheduler.set_timesteps(sigmas=[1.0, 0.5, 0.0], timesteps=[900.0, 500.0])

    def test_step_shape_preserved(self):
        scheduler = self.scheduler_class(**self.get_default_config())
        scheduler.set_timesteps(num_inference_steps=4)

        sample = torch.randn(2, 4, 8, 8)
        model_output = torch.randn_like(sample)
        timestep = scheduler.timesteps[0:1]

        output = scheduler.step(model_output, timestep, sample)
        self.assertIsInstance(output, FlowMatchEulerDiscreteSchedulerOutput)
        self.assertEqual(output.prev_sample.shape, sample.shape)
        self.assertEqual(output.prev_sample.dtype, model_output.dtype)

    def test_step_rejects_integer_timestep(self):
        scheduler = self.scheduler_class(**self.get_default_config())
        scheduler.set_timesteps(num_inference_steps=4)
        sample = torch.randn(1, 4, 4, 4)
        with self.assertRaises(ValueError):
            scheduler.step(torch.randn_like(sample), 0, sample)

    def test_scale_noise_interpolates_sample_and_noise(self):
        scheduler = self.scheduler_class(**self.get_default_config())
        scheduler.set_timesteps(timesteps=[500.0, 250.0])
        sample = torch.zeros(1, 4, 4, 4)
        noise = torch.ones_like(sample)
        mid_t = scheduler.timesteps[0:1]
        mixed = scheduler.scale_noise(sample, mid_t, noise)
        # sigma=0.5 at t=500 with default 1000 training steps
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
