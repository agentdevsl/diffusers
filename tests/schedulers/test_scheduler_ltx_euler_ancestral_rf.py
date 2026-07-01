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

from diffusers import LTXEulerAncestralRFScheduler
from diffusers.schedulers.scheduling_ltx_euler_ancestral_rf import LTXEulerAncestralRFSchedulerOutput


class LTXEulerAncestralRFSchedulerTest(unittest.TestCase):
    """
    Contract tests for the LTX RF ancestral scheduler used by LTX-Video long-form pipelines.
    Mirrors the style of `test_scheduler_flow_map_euler_discrete.py` because this scheduler
    has a non-standard step API and cannot reuse `SchedulerCommonTest`.
    """

    scheduler_class = LTXEulerAncestralRFScheduler

    def get_default_config(self, **kwargs):
        config = {
            "num_train_timesteps": 1000,
            "eta": 1.0,
            "s_noise": 1.0,
        }
        config.update(**kwargs)
        return config

    def test_set_timesteps_auto_generates_schedule(self):
        scheduler = self.scheduler_class(**self.get_default_config())
        for nfe in [1, 2, 4, 8]:
            scheduler.set_timesteps(num_inference_steps=nfe)
            self.assertEqual(scheduler.num_inference_steps, nfe)
            self.assertEqual(scheduler.timesteps.shape, (nfe,))
            self.assertEqual(scheduler.sigmas.shape, (nfe + 1,))
            self.assertAlmostEqual(scheduler.sigmas[-1].item(), 0.0, places=6)

    def test_set_timesteps_requires_args(self):
        scheduler = self.scheduler_class(**self.get_default_config())
        with self.assertRaises(ValueError):
            scheduler.set_timesteps()

    def test_set_timesteps_explicit_sigmas(self):
        scheduler = self.scheduler_class(**self.get_default_config())
        custom_sigmas = [1.0, 0.75, 0.5, 0.25, 0.0]
        scheduler.set_timesteps(sigmas=custom_sigmas)
        self.assertEqual(scheduler.num_inference_steps, 4)
        for i, sigma in enumerate(custom_sigmas):
            self.assertAlmostEqual(scheduler.sigmas[i].item(), sigma, places=5)
        self.assertAlmostEqual(scheduler.timesteps[0].item(), 1000.0, places=4)

    def test_set_timesteps_rejects_non_1d_sigmas(self):
        scheduler = self.scheduler_class(**self.get_default_config())
        with self.assertRaises(ValueError):
            scheduler.set_timesteps(sigmas=[[1.0, 0.5], [0.5, 0.0]])

    def test_step_shape_preserved(self):
        scheduler = self.scheduler_class(**self.get_default_config())
        scheduler.set_timesteps(num_inference_steps=4)

        sample = torch.randn(2, 8, 4, 4)
        model_output = torch.randn_like(sample)
        timestep = scheduler.timesteps[0:1]

        output = scheduler.step(model_output, timestep, sample)
        self.assertIsInstance(output, LTXEulerAncestralRFSchedulerOutput)
        self.assertEqual(output.prev_sample.shape, sample.shape)
        self.assertEqual(output.prev_sample.dtype, sample.dtype)

    def test_step_deterministic_when_eta_zero(self):
        scheduler = self.scheduler_class(**self.get_default_config(eta=0.0))
        scheduler.set_timesteps(num_inference_steps=4)

        sample = torch.randn(1, 4, 8, 8)
        model_output = torch.randn_like(sample)
        generator = torch.Generator().manual_seed(0)

        first = scheduler.step(model_output, scheduler.timesteps[0:1], sample, generator=generator).prev_sample

        scheduler.set_timesteps(num_inference_steps=4)
        second = scheduler.step(model_output, scheduler.timesteps[0:1], sample, generator=generator).prev_sample
        torch.testing.assert_close(first, second)

    def test_step_rejects_integer_timestep(self):
        scheduler = self.scheduler_class(**self.get_default_config())
        scheduler.set_timesteps(num_inference_steps=4)
        sample = torch.randn(1, 4, 4, 4)
        with self.assertRaises(ValueError):
            scheduler.step(torch.randn_like(sample), 0, sample)

    def test_index_for_timestep_duplicate_handling(self):
        scheduler = self.scheduler_class(**self.get_default_config())
        scheduler.set_timesteps(sigmas=[1.0, 0.5, 0.5, 0.0])
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
        scheduler = self.scheduler_class(**self.get_default_config(eta=0.0))
        scheduler.set_timesteps(num_inference_steps=4)

        sample = torch.randn(1, 4, 8, 8)
        for t in scheduler.timesteps:
            sample = scheduler.step(torch.randn_like(sample), t, sample).prev_sample
        self.assertEqual(sample.shape, (1, 4, 8, 8))
