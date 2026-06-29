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
import os
import tempfile
import unittest
from unittest.mock import Mock, patch

import PIL.Image
import torch
from torch import nn

from diffusers.utils.loading_utils import get_module_from_name, get_submodule_by_name, load_image, load_video


class LoadingUtilsTest(unittest.TestCase):
    def test_load_image_pil_passthrough_converts_rgb(self):
        image = PIL.Image.new("RGBA", (4, 4), color=(255, 0, 0, 128))
        loaded = load_image(image)
        self.assertEqual(loaded.mode, "RGB")
        self.assertEqual(loaded.size, (4, 4))

    def test_load_image_local_path(self):
        image = PIL.Image.new("RGB", (8, 8), color="green")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name)
            path = tmp.name
        try:
            loaded = load_image(path)
            self.assertEqual(loaded.size, (8, 8))
            self.assertEqual(loaded.mode, "RGB")
        finally:
            os.remove(path)

    def test_load_image_invalid_path_raises(self):
        with self.assertRaises(ValueError):
            load_image("/path/that/does/not/exist.png")

    def test_load_image_invalid_scheme_raises(self):
        with self.assertRaises(ValueError):
            load_image("ftp://example.com/image.png")

    def test_load_image_invalid_type_raises(self):
        with self.assertRaises(ValueError):
            load_image(123)

    def test_load_image_custom_convert_method(self):
        image = PIL.Image.new("RGB", (4, 4), color="blue")

        def to_grayscale(img):
            return img.convert("L")

        loaded = load_image(image, convert_method=to_grayscale)
        self.assertEqual(loaded.mode, "L")

    @patch("diffusers.utils.loading_utils.requests.get")
    def test_load_image_from_url(self, mock_get):
        buffer = io.BytesIO()
        PIL.Image.new("RGB", (6, 6), color="red").save(buffer, format="PNG")
        buffer.seek(0)
        mock_response = Mock()
        mock_response.raw = buffer
        mock_get.return_value = mock_response

        loaded = load_image("https://example.com/image.png")
        self.assertEqual(loaded.size, (6, 6))
        self.assertEqual(loaded.mode, "RGB")

    def test_load_video_invalid_path_raises(self):
        with self.assertRaises(ValueError):
            load_video("/path/that/does/not/exist.mp4")

    def test_load_video_gif_frames(self):
        frames = [PIL.Image.new("RGB", (4, 4), color=(i * 40, 0, 0)) for i in range(3)]
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
            path = tmp.name
        try:
            frames[0].save(path, save_all=True, append_images=frames[1:], duration=100, loop=0)
            loaded = load_video(path)
            self.assertEqual(len(loaded), 3)
            self.assertEqual(loaded[0].size, (4, 4))
        finally:
            os.remove(path)

    def test_get_module_from_name_nested(self):
        module = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        found, name = get_module_from_name(module, "0.weight")
        self.assertIsInstance(found, nn.Linear)
        self.assertEqual(name, "weight")

    def test_get_module_from_name_missing_attribute_raises(self):
        module = nn.Linear(4, 4)
        with self.assertRaises(AttributeError):
            get_module_from_name(module, "missing.weight")

    def test_get_submodule_by_name_modulelist_index(self):
        module = nn.ModuleList([nn.Linear(2, 2), nn.Linear(3, 3)])
        found = get_submodule_by_name(module, "1")
        self.assertIsInstance(found, nn.Linear)
        self.assertEqual(found.in_features, 3)

    def test_get_submodule_by_name_dotted_path(self):
        module = nn.Sequential(
            nn.ModuleDict({"block": nn.Linear(4, 4)}),
        )
        found = get_submodule_by_name(module, "0.block")
        self.assertIsInstance(found, nn.Linear)
