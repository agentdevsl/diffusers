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
from transformers import AutoConfig, AutoTokenizer, T5EncoderModel

from diffusers import (
    AutoencoderKLLTXVideo,
    LTXEulerAncestralRFScheduler,
    LTXI2VLongMultiPromptPipeline,
    LTXVideoTransformer3DModel,
)
from diffusers.pipelines.ltx.pipeline_ltx_i2v_long_multi_prompt import (
    adain_normalize_latents,
    build_video_coords_for_window,
    get_latent_coords,
    inject_prev_tail_latents,
    linear_overlap_fuse,
    parse_prompt_segments,
    split_into_temporal_windows,
)

from ...testing_utils import enable_full_determinism, torch_device


enable_full_determinism()


class LTXI2VLongMultiPromptHelpersTest(unittest.TestCase):
    def test_split_into_temporal_windows_single_and_overlapping(self):
        windows = split_into_temporal_windows(latent_len=10, temporal_tile_size=4, temporal_overlap=1, compression=8)
        self.assertEqual(windows, [(0, 4), (3, 7), (6, 10)])

        single = split_into_temporal_windows(latent_len=3, temporal_tile_size=8, temporal_overlap=2, compression=8)
        self.assertEqual(single, [(0, 3)])

    def test_split_into_temporal_windows_rejects_invalid_tile_size(self):
        with self.assertRaises(ValueError):
            split_into_temporal_windows(latent_len=5, temporal_tile_size=0, temporal_overlap=0, compression=8)

    def test_parse_prompt_segments_bar_split(self):
        parts = parse_prompt_segments("walk | eat | sleep", prompt_segments=None)
        self.assertEqual(parts, ["walk", "eat", "sleep"])

    def test_parse_prompt_segments_window_mapping(self):
        segments = [
            {"start_window": 0, "end_window": 1, "text": "intro"},
            {"start_window": 2, "end_window": 3, "text": "outro"},
        ]
        texts = parse_prompt_segments("", prompt_segments=segments)
        self.assertEqual(texts, ["intro", "intro", "outro", "outro"])

    def test_parse_prompt_segments_fills_gaps_from_previous(self):
        segments = [
            {"start_window": 0, "end_window": 0, "text": "first"},
            {"start_window": 2, "end_window": 2, "text": "third"},
        ]
        texts = parse_prompt_segments("", prompt_segments=segments)
        self.assertEqual(texts, ["first", "first", "third"])

    def test_linear_overlap_fuse_blends_overlap_region(self):
        prev = torch.ones(1, 2, 4, 2, 2)
        new = torch.zeros(1, 2, 4, 2, 2)
        fused = linear_overlap_fuse(prev, new, overlap=2)
        self.assertEqual(fused.shape[2], 6)
        self.assertTrue(torch.all(fused[:, :, :2] == 1))
        self.assertTrue(torch.all(fused[:, :, -2:] == 0))
        self.assertTrue(torch.all(fused[:, :, 2:4] > 0))
        self.assertTrue(torch.all(fused[:, :, 2:4] < 1))

    def test_linear_overlap_fuse_no_overlap_concatenates(self):
        prev = torch.ones(1, 2, 2, 2, 2)
        new = torch.zeros(1, 2, 3, 2, 2)
        fused = linear_overlap_fuse(prev, new, overlap=1)
        self.assertEqual(fused.shape[2], 5)

    def test_adain_normalize_latents_noop_without_reference(self):
        latents = torch.randn(1, 4, 2, 2, 2)
        self.assertIs(adain_normalize_latents(latents, None, factor=0.5), latents)

    def test_adain_normalize_latents_matches_reference_stats(self):
        curr = torch.randn(1, 2, 3, 4, 4)
        ref = torch.randn(1, 2, 3, 4, 4) * 2 + 1
        normalized = adain_normalize_latents(curr, ref, factor=1.0)
        self.assertAlmostEqual(normalized.mean().item(), ref.mean().item(), places=4)
        self.assertAlmostEqual(normalized.std().item(), ref.std().item(), places=4)

    def test_inject_prev_tail_latents_appends_tail_and_mask(self):
        window = torch.randn(1, 4, 3, 2, 2)
        tail = torch.randn(1, 4, 2, 2, 2)
        mask = torch.ones(1, 1, 3, 2, 2)
        updated, updated_mask, overlap_len = inject_prev_tail_latents(
            window, tail, mask, overlap_lat=2, strength=0.75, prev_overlap_len=0
        )
        self.assertEqual(updated.shape[2], 5)
        self.assertEqual(updated_mask.shape[2], 5)
        self.assertEqual(overlap_len, 2)
        self.assertTrue(torch.all(updated[:, :, -2:] == tail))
        self.assertTrue(torch.all(updated_mask[:, :, -2:] == 0.25))

    def test_get_latent_coords_shape_and_time_shift(self):
        coords = get_latent_coords(
            latent_num_frames=2,
            latent_height=2,
            latent_width=2,
            batch_size=1,
            device=torch.device("cpu"),
            rope_interpolation_scale=(8, 4, 4),
            latent_idx=1,
        )
        self.assertEqual(coords.shape, (1, 3, 8))
        self.assertGreater(coords[0, 0, 0].item(), 0)

    def test_build_video_coords_for_window_applies_frame_rate(self):
        latents = torch.zeros(1, 4, 2, 2, 2)
        rope_scale = (8, 4, 4)
        coords = build_video_coords_for_window(
            latents=latents,
            overlap_len=0,
            guiding_len=0,
            negative_len=0,
            rope_interpolation_scale=rope_scale,
            frame_rate=25,
        )
        self.assertEqual(coords.shape, (1, 3, 8))
        self.assertAlmostEqual(coords[0, 0, 0].item(), 0.0, places=6)


class LTXI2VLongMultiPromptPipelineFastTests(unittest.TestCase):
    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = LTXVideoTransformer3DModel(
            in_channels=8,
            out_channels=8,
            patch_size=1,
            patch_size_t=1,
            num_attention_heads=4,
            attention_head_dim=8,
            cross_attention_dim=32,
            num_layers=1,
            caption_channels=32,
        )

        torch.manual_seed(0)
        vae = AutoencoderKLLTXVideo(
            in_channels=3,
            out_channels=3,
            latent_channels=8,
            block_out_channels=(8, 8, 8, 8),
            decoder_block_out_channels=(8, 8, 8, 8),
            layers_per_block=(1, 1, 1, 1, 1),
            decoder_layers_per_block=(1, 1, 1, 1, 1),
            spatio_temporal_scaling=(True, True, False, False),
            decoder_spatio_temporal_scaling=(True, True, False, False),
            decoder_inject_noise=(False, False, False, False, False),
            upsample_residual=(False, False, False, False),
            upsample_factor=(1, 1, 1, 1),
            timestep_conditioning=False,
            patch_size=1,
            patch_size_t=1,
            encoder_causal=True,
            decoder_causal=False,
        )
        vae.use_framewise_encoding = False
        vae.use_framewise_decoding = False

        torch.manual_seed(0)
        scheduler = LTXEulerAncestralRFScheduler(num_train_timesteps=1000)
        config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-t5")
        text_encoder = T5EncoderModel(config)
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

        return {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
        }

    def test_inference_latent_output_single_window(self):
        device = torch_device
        pipe = LTXI2VLongMultiPromptPipeline(**self.get_dummy_components())
        pipe.to(device)

        generator = torch.Generator(device=device).manual_seed(0)
        output = pipe(
            prompt="segment one | segment two",
            generator=generator,
            num_inference_steps=2,
            height=32,
            width=32,
            num_frames=9,
            temporal_tile_size=80,
            temporal_overlap=0,
            guidance_scale=1.0,
            output_type="latent",
            max_sequence_length=16,
        )

        frames = output.frames
        self.assertIsInstance(frames, torch.Tensor)
        self.assertEqual(frames.ndim, 5)
        self.assertEqual(frames.shape[0], 1)
        self.assertEqual(frames.shape[3], 32 // pipe.vae_spatial_compression_ratio)
        self.assertEqual(frames.shape[4], 32 // pipe.vae_spatial_compression_ratio)

    def test_inference_rejects_non_divisible_dimensions(self):
        pipe = LTXI2VLongMultiPromptPipeline(**self.get_dummy_components())
        pipe.to(torch_device)

        with self.assertRaises(ValueError):
            pipe(
                prompt="test",
                height=30,
                width=32,
                num_frames=9,
                num_inference_steps=1,
                output_type="latent",
            )
