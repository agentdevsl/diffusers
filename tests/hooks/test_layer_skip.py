# Copyright 2025 HuggingFace Inc.
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

import torch

from diffusers.hooks._helpers import TransformerBlockMetadata, TransformerBlockRegistry
from diffusers.hooks.layer_skip import LayerSkipConfig, apply_layer_skip
from diffusers.models import ModelMixin


class FluxLikeBlock(torch.nn.Module):
    def forward(self, hidden_states, encoder_hidden_states=None, **kwargs):
        return encoder_hidden_states + 1.0, hidden_states + 2.0


class FluxLikeTransformer(ModelMixin):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([FluxLikeBlock(), FluxLikeBlock()])

    def forward(self, hidden_states, encoder_hidden_states=None):
        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states
            )
        return encoder_hidden_states, hidden_states


def test_transformer_block_skip_hook_respects_return_order():
    TransformerBlockRegistry.register(
        FluxLikeBlock,
        TransformerBlockMetadata(return_hidden_states_index=1, return_encoder_hidden_states_index=0),
    )

    model = FluxLikeTransformer()
    hidden_states = torch.zeros(2, 3)
    encoder_hidden_states = torch.ones(2, 3)

    apply_layer_skip(model, LayerSkipConfig(indices=[0], fqn="transformer_blocks"))

    out_encoder, out_hidden = model(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states)

    # Block 0 is skipped (identity inputs), block 1 still runs (+1 / +2).
    torch.testing.assert_close(out_encoder, encoder_hidden_states + 1.0)
    torch.testing.assert_close(out_hidden, hidden_states + 2.0)
