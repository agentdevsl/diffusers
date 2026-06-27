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

import io
import json
import unittest
from unittest.mock import Mock

import torch
from PIL import Image

from diffusers.utils.remote_utils import (
    check_inputs_decode,
    detect_image_type,
    postprocess_decode,
    prepare_decode,
    prepare_encode,
)


class RemoteUtilsTest(unittest.TestCase):
    def test_detect_image_type(self):
        self.assertEqual(detect_image_type(b"\xff\xd8\xff"), "jpeg")
        self.assertEqual(detect_image_type(b"\x89PNG\r\n\x1a\n"), "png")
        self.assertEqual(detect_image_type(b"GIF89a"), "gif")
        self.assertEqual(detect_image_type(b"BM"), "bmp")
        self.assertEqual(detect_image_type(b"unknown"), "unknown")

    def test_check_inputs_decode_packed_latents_requires_hw(self):
        tensor = torch.randn(4, 8, 8)
        with self.assertRaises(ValueError):
            check_inputs_decode("http://example.com", tensor)

    def test_check_inputs_decode_processor_required(self):
        tensor = torch.randn(1, 4, 8, 8)
        with self.assertRaises(ValueError):
            check_inputs_decode(
                "http://example.com",
                tensor,
                processor=None,
                output_type="pt",
                return_type="pil",
                partial_postprocess=False,
            )

    def test_prepare_decode_sets_accept_header_for_jpeg(self):
        tensor = torch.randn(1, 4, 8, 8, dtype=torch.float16)
        payload = prepare_decode(tensor, output_type="pil", image_format="jpg")
        self.assertEqual(payload["headers"]["Accept"], "image/jpeg")
        self.assertEqual(payload["params"]["output_type"], "pil")
        self.assertEqual(payload["params"]["shape"], list(tensor.shape))

    def test_prepare_encode_tensor_includes_shape_and_dtype(self):
        tensor = torch.randn(1, 3, 8, 8, dtype=torch.float16)
        payload = prepare_encode(tensor, scaling_factor=0.18215)
        self.assertEqual(payload["params"]["shape"], list(tensor.shape))
        self.assertEqual(payload["params"]["dtype"], "float16")
        self.assertEqual(payload["params"]["scaling_factor"], 0.18215)

    def test_prepare_encode_pil_image(self):
        image = Image.new("RGB", (8, 8), color="red")
        payload = prepare_encode(image)
        self.assertIn(b"PNG", payload["data"][:8])

    def test_postprocess_decode_pil_without_processor(self):
        buffer = io.BytesIO()
        Image.new("RGB", (4, 4), color="blue").save(buffer, format="PNG")
        response = Mock()
        response.content = buffer.getvalue()

        output = postprocess_decode(response, processor=None, output_type="pil", return_type="pil")
        self.assertIsInstance(output, Image.Image)
        self.assertEqual(output.size, (4, 4))
        self.assertEqual(output.format, "png")

    def test_postprocess_decode_pt_tensor(self):
        tensor = torch.arange(16, dtype=torch.float32).reshape(1, 4, 2, 2)
        response = Mock()
        response.content = tensor.numpy().tobytes()
        response.headers = {
            "shape": json.dumps(list(tensor.shape)),
            "dtype": "float32",
        }

        output = postprocess_decode(
            response,
            processor=None,
            output_type="pt",
            return_type="pt",
            partial_postprocess=False,
        )
        torch.testing.assert_close(output, tensor)
