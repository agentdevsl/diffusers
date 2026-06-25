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

from diffusers import HeliosScheduler
from diffusers.schedulers.scheduling_helios import HeliosSchedulerOutput


class HeliosSchedulerTest(unittest.TestCase):
    """
    Unit tests for HeliosScheduler multi-stage scheduling. Pipeline tests only cover stages=1;
    these tests lock in per-stage sigma ranges and step behavior used by the default 3-stage config.
    """

    def get_default_config(self, **kwargs):
        config = {
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "stages": 3,
            "stage_range": [0, 1 / 3, 2 / 3, 1],
            "gamma": 1 / 3,
            "prediction_type": "flow_prediction",
            "scheduler_type": "euler",
            "use_dynamic_shifting": False,
        }
        config.update(**kwargs)
        return config

    def test_multi_stage_init_populates_per_stage_buffers(self):
        scheduler = HeliosScheduler(**self.get_default_config())
        self.assertEqual(len(scheduler.timesteps_per_stage), 3)
        self.assertEqual(len(scheduler.sigmas_per_stage), 3)
        self.assertEqual(len(scheduler.start_sigmas), 3)
        self.assertEqual(len(scheduler.end_sigmas), 3)
        for stage in range(3):
            self.assertGreater(scheduler.start_sigmas[stage], scheduler.end_sigmas[stage])

    def test_set_timesteps_multi_stage_per_stage_index(self):
        scheduler = HeliosScheduler(**self.get_default_config())
        for stage_index in range(3):
            scheduler.set_timesteps(num_inference_steps=8, stage_index=stage_index)
            self.assertEqual(scheduler.timesteps.shape, (8,))
            self.assertEqual(scheduler.sigmas.shape, (9,))
            self.assertAlmostEqual(scheduler.sigmas[-1].item(), 0.0, places=6)
            self.assertGreater(scheduler.timesteps[0].item(), scheduler.timesteps[-1].item())

    def test_set_timesteps_single_stage(self):
        scheduler = HeliosScheduler(**self.get_default_config(stages=1, stage_range=[0, 1]))
        scheduler.set_timesteps(num_inference_steps=4)
        self.assertEqual(scheduler.timesteps.shape, (4,))
        self.assertEqual(scheduler.sigmas.shape, (5,))

    def test_step_euler_updates_sample(self):
        scheduler = HeliosScheduler(**self.get_default_config(stages=1, stage_range=[0, 1]))
        scheduler.set_timesteps(num_inference_steps=4)

        sample = torch.randn(2, 4, 8, 8)
        model_output = torch.randn_like(sample)
        output = scheduler.step(model_output, scheduler.timesteps[0], sample)
        self.assertIsInstance(output, HeliosSchedulerOutput)
        self.assertEqual(output.prev_sample.shape, sample.shape)
        self.assertFalse(torch.allclose(output.prev_sample, sample))

    def test_convert_model_output_flow_prediction(self):
        scheduler = HeliosScheduler(**self.get_default_config(stages=1, stage_range=[0, 1], scheduler_type="unipc"))
        scheduler.set_timesteps(num_inference_steps=4)
        scheduler._step_index = 0

        sample = torch.randn(1, 4, 4, 4)
        model_output = torch.randn_like(sample)
        x0 = scheduler.convert_model_output(model_output, sample=sample)
        expected = sample - scheduler.sigmas[0] * model_output
        torch.testing.assert_close(x0, expected)

    def test_step_unipc_updates_sample(self):
        scheduler = HeliosScheduler(**self.get_default_config(stages=1, stage_range=[0, 1], scheduler_type="unipc"))
        scheduler.set_timesteps(num_inference_steps=4)

        sample = torch.randn(2, 4, 8, 8)
        model_output = torch.randn_like(sample)
        output = scheduler.step(model_output, scheduler.timesteps[0], sample)
        self.assertIsInstance(output, HeliosSchedulerOutput)
        self.assertEqual(output.prev_sample.shape, sample.shape)
        self.assertFalse(torch.allclose(output.prev_sample, sample))

    def test_step_unipc_full_loop(self):
        scheduler = HeliosScheduler(**self.get_default_config(stages=1, stage_range=[0, 1], scheduler_type="unipc"))
        scheduler.set_timesteps(num_inference_steps=4)

        sample = torch.randn(1, 4, 8, 8)
        for t in scheduler.timesteps:
            sample = scheduler.step(torch.randn_like(sample), t, sample).prev_sample
        self.assertEqual(sample.shape, (1, 4, 8, 8))

    def test_dynamic_shifting_rescales_timesteps(self):
        scheduler = HeliosScheduler(
            **self.get_default_config(stages=1, stage_range=[0, 1], use_dynamic_shifting=True)
        )
        scheduler.set_timesteps(num_inference_steps=4, mu=0.5)
        self.assertEqual(scheduler.timesteps.shape, (4,))
