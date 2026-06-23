# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import torch

from ..utils import get_logger
from ..utils.torch_utils import unwrap_module
from ._common import _ALL_TRANSFORMER_BLOCK_IDENTIFIERS
from ._helpers import TransformerBlockRegistry
from .hooks import BaseState, HookRegistry, ModelHook, StateManager


logger = get_logger(__name__)  # pylint: disable=invalid-name

_TEACACHE_LEADER_BLOCK_HOOK = "teacache_leader_block_hook"
_TEACACHE_BLOCK_HOOK = "teacache_block_hook"

FLUX_TEACACHE_COEFFICIENTS = [498.651651, -283.781631, 55.8554382, -3.82021401, 0.264230861]


def _evaluate_polynomial(coefficients: List[float], x: float) -> float:
    degree = len(coefficients) - 1
    result = 0.0
    for i, coefficient in enumerate(coefficients):
        result += coefficient * (x ** (degree - i))
    return result


def _compute_relative_l1_distance(current: torch.Tensor, previous: torch.Tensor) -> float:
    return (torch.abs(current - previous).mean() / torch.abs(previous).mean()).cpu().item()


@dataclass
class TeaCacheConfig:
    r"""
    Configuration for [TeaCache](https://huggingface.co/papers/2411.19108).

    Args:
        rel_l1_thresh (`float`, defaults to `0.4`):
            Threshold for the accumulated polynomial-rescaled relative L1 distance between consecutive modulated
            inputs. When the accumulated distance is below this threshold, the transformer block stack is skipped
            and the previous residual is replayed.
        num_inference_steps (`int`, defaults to `28`):
            The number of inference steps used in the pipeline. The first and last steps always compute regardless
            of the accumulated distance.
        coefficients (`List[float]`, *optional*):
            Polynomial coefficients for rescaling the relative L1 distance. Defaults to
            `FLUX_TEACACHE_COEFFICIENTS` for FLUX models.
    """

    rel_l1_thresh: float = 0.4
    num_inference_steps: int = 28
    coefficients: Optional[List[float]] = None

    def __post_init__(self):
        if self.coefficients is None:
            self.coefficients = list(FLUX_TEACACHE_COEFFICIENTS)


class TeaCacheState(BaseState):
    def __init__(self) -> None:
        super().__init__()
        self.previous_modulated_input: torch.Tensor = None
        self.accumulated_distance: float = 0.0
        self.previous_residual: torch.Tensor = None
        self.step_index: int = 0
        self.should_compute: bool = True
        self.head_block_input: Union[torch.Tensor, tuple[torch.Tensor, ...]] = None

    def reset(self):
        self.previous_modulated_input = None
        self.accumulated_distance = 0.0
        self.previous_residual = None
        self.step_index = 0
        self.should_compute = True
        self.head_block_input = None


_MODEL_CONFIG: dict[type, dict] = {}


def _register_flux_model_config():
    from ..models.transformers.transformer_flux import FluxTransformer2DModel

    def extract_modulated_input(block, hidden_states, args, kwargs):
        temb = kwargs.get("temb")
        modulated_input, _, _, _, _ = block.norm1(hidden_states, emb=temb)
        return modulated_input

    _MODEL_CONFIG[FluxTransformer2DModel] = {
        "coefficients": FLUX_TEACACHE_COEFFICIENTS,
        "extract_modulated_input": extract_modulated_input,
    }


def _get_model_config(model_class):
    if model_class in _MODEL_CONFIG:
        return _MODEL_CONFIG[model_class]

    if model_class.__name__ != "FluxTransformer2DModel":
        return None

    from ..models.transformers.transformer_flux import FluxTransformer2DModel

    if model_class is not FluxTransformer2DModel:
        return None

    _register_flux_model_config()
    return _MODEL_CONFIG.get(model_class)


class TeaCacheHeadHook(ModelHook):
    _is_stateful = True

    def __init__(
        self,
        state_manager: StateManager,
        config: TeaCacheConfig,
        extract_modulated_input: Callable,
        coefficients: List[float],
    ):
        self.state_manager = state_manager
        self.config = config
        self.extract_modulated_input = extract_modulated_input
        self.coefficients = coefficients
        self._metadata = None

    def initialize_hook(self, module):
        unwrapped_module = unwrap_module(module)
        self._metadata = TransformerBlockRegistry.get(unwrapped_module.__class__)
        return module

    @torch.compiler.disable
    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        if self.state_manager._current_context is None:
            self.state_manager.set_context("inference")

        arg_name = self._metadata.hidden_states_argument_name
        hidden_states = self._metadata._get_parameter_from_args_kwargs(arg_name, args, kwargs)

        state: TeaCacheState = self.state_manager.get_state()
        state.head_block_input = hidden_states

        modulated_input = self.extract_modulated_input(module, hidden_states, args, kwargs)

        should_compute = True
        current_step = state.step_index
        is_boundary_step = current_step == 0 or current_step == self.config.num_inference_steps - 1

        if is_boundary_step:
            state.accumulated_distance = 0.0
            should_compute = True
        elif state.previous_modulated_input is not None:
            rel_l1_distance = _compute_relative_l1_distance(modulated_input, state.previous_modulated_input)
            state.accumulated_distance += _evaluate_polynomial(self.coefficients, rel_l1_distance)

            if state.accumulated_distance < self.config.rel_l1_thresh and state.previous_residual is not None:
                should_compute = False
            else:
                state.accumulated_distance = 0.0
                should_compute = True

        state.previous_modulated_input = modulated_input
        state.should_compute = should_compute

        if not should_compute:
            logger.debug(f"TeaCache: Skipping step {state.step_index}")

            output = hidden_states
            res = state.previous_residual

            if res.device != output.device:
                res = res.to(output.device)

            if res.shape == output.shape:
                output = output + res
            elif (
                output.ndim == 3
                and res.ndim == 3
                and output.shape[0] == res.shape[0]
                and output.shape[2] == res.shape[2]
            ):
                diff = output.shape[1] - res.shape[1]
                if diff > 0:
                    output = output.clone()
                    output[:, diff:, :] = output[:, diff:, :] + res
                else:
                    logger.warning(
                        f"TeaCache: Dimension mismatch. Input {output.shape}, Residual {res.shape}. "
                        "Cannot apply residual safely. Returning input without residual."
                    )
            else:
                logger.warning(
                    f"TeaCache: Dimension mismatch. Input {output.shape}, Residual {res.shape}. "
                    "Cannot apply residual safely. Returning input without residual."
                )

            if self._metadata.return_encoder_hidden_states_index is not None:
                original_encoder_hidden_states = self._metadata._get_parameter_from_args_kwargs(
                    "encoder_hidden_states", args, kwargs
                )
                max_idx = max(
                    self._metadata.return_hidden_states_index, self._metadata.return_encoder_hidden_states_index
                )
                ret_list = [None] * (max_idx + 1)
                ret_list[self._metadata.return_hidden_states_index] = output
                ret_list[self._metadata.return_encoder_hidden_states_index] = original_encoder_hidden_states
                return tuple(ret_list)

            return output

        output = self.fn_ref.original_forward(*args, **kwargs)
        return output

    def reset_state(self, module):
        self.state_manager.reset()
        return module


class TeaCacheBlockHook(ModelHook):
    def __init__(self, state_manager: StateManager, is_tail: bool = False, config: TeaCacheConfig = None):
        super().__init__()
        self.state_manager = state_manager
        self.is_tail = is_tail
        self.config = config
        self._metadata = None

    def initialize_hook(self, module):
        unwrapped_module = unwrap_module(module)
        self._metadata = TransformerBlockRegistry.get(unwrapped_module.__class__)
        return module

    @torch.compiler.disable
    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        if self.state_manager._current_context is None:
            self.state_manager.set_context("inference")
        state: TeaCacheState = self.state_manager.get_state()

        if not state.should_compute:
            arg_name = self._metadata.hidden_states_argument_name
            hidden_states = self._metadata._get_parameter_from_args_kwargs(arg_name, args, kwargs)

            if self.is_tail:
                self._advance_step(state)

            if self._metadata.return_encoder_hidden_states_index is not None:
                encoder_hidden_states = self._metadata._get_parameter_from_args_kwargs(
                    "encoder_hidden_states", args, kwargs
                )
                max_idx = max(
                    self._metadata.return_hidden_states_index, self._metadata.return_encoder_hidden_states_index
                )
                ret_list = [None] * (max_idx + 1)
                ret_list[self._metadata.return_hidden_states_index] = hidden_states
                ret_list[self._metadata.return_encoder_hidden_states_index] = encoder_hidden_states
                return tuple(ret_list)

            return hidden_states

        output = self.fn_ref.original_forward(*args, **kwargs)

        if self.is_tail:
            if isinstance(output, tuple):
                out_hidden = output[self._metadata.return_hidden_states_index]
            else:
                out_hidden = output

            in_hidden = state.head_block_input

            if in_hidden is not None:
                if out_hidden.shape == in_hidden.shape:
                    residual = out_hidden - in_hidden
                elif out_hidden.ndim == 3 and in_hidden.ndim == 3 and out_hidden.shape[2] == in_hidden.shape[2]:
                    diff = in_hidden.shape[1] - out_hidden.shape[1]
                    if diff == 0:
                        residual = out_hidden - in_hidden
                    else:
                        residual = out_hidden - in_hidden
                else:
                    residual = out_hidden

                state.previous_residual = residual

            self._advance_step(state)

        return output

    def _advance_step(self, state: TeaCacheState):
        state.step_index += 1
        if state.step_index >= self.config.num_inference_steps:
            state.step_index = 0
            state.accumulated_distance = 0.0
            state.previous_residual = None
            state.previous_modulated_input = None


def apply_teacache(module: torch.nn.Module, config: TeaCacheConfig) -> None:
    """
    Applies TeaCache to a given module (typically a Transformer).

    Args:
        module (`torch.nn.Module`):
            The module to apply TeaCache to.
        config (`TeaCacheConfig`):
            The configuration for TeaCache.
    """
    unwrapped_module = unwrap_module(module)
    model_class = unwrapped_module.__class__

    model_config = _get_model_config(model_class)
    if model_config is None:
        raise ValueError(
            f"TeaCache is not supported for {model_class.__name__}. "
            "TeaCache v1 supports FLUX transformers (`FluxTransformer2DModel`) only."
        )

    coefficients = config.coefficients if config.coefficients is not None else model_config["coefficients"]
    extract_modulated_input = model_config["extract_modulated_input"]

    HookRegistry.check_if_exists_or_initialize(module)

    state_manager = StateManager(TeaCacheState, (), {})
    remaining_blocks = []

    for name, submodule in module.named_children():
        if name not in _ALL_TRANSFORMER_BLOCK_IDENTIFIERS or not isinstance(submodule, torch.nn.ModuleList):
            continue
        for index, block in enumerate(submodule):
            remaining_blocks.append((f"{name}.{index}", block))

    if not remaining_blocks:
        logger.warning("TeaCache: No transformer blocks found to apply hooks.")
        return

    if len(remaining_blocks) == 1:
        name, block = remaining_blocks[0]
        logger.info(f"TeaCache: Applying Head+Tail Hooks to single block '{name}'")
        _apply_teacache_block_hook(block, state_manager, config, is_tail=True)
        _apply_teacache_head_hook(block, state_manager, config, extract_modulated_input, coefficients)
        return

    head_block_name, head_block = remaining_blocks.pop(0)
    tail_block_name, tail_block = remaining_blocks.pop(-1)

    logger.info(f"TeaCache: Applying Head Hook to {head_block_name}")
    _apply_teacache_head_hook(head_block, state_manager, config, extract_modulated_input, coefficients)

    for name, block in remaining_blocks:
        _apply_teacache_block_hook(block, state_manager, config)

    logger.info(f"TeaCache: Applying Tail Hook to {tail_block_name}")
    _apply_teacache_block_hook(tail_block, state_manager, config, is_tail=True)


def _apply_teacache_head_hook(
    block: torch.nn.Module,
    state_manager: StateManager,
    config: TeaCacheConfig,
    extract_modulated_input: Callable,
    coefficients: List[float],
) -> None:
    registry = HookRegistry.check_if_exists_or_initialize(block)

    if registry.get_hook(_TEACACHE_LEADER_BLOCK_HOOK) is not None:
        registry.remove_hook(_TEACACHE_LEADER_BLOCK_HOOK)

    hook = TeaCacheHeadHook(state_manager, config, extract_modulated_input, coefficients)
    registry.register_hook(hook, _TEACACHE_LEADER_BLOCK_HOOK)


def _apply_teacache_block_hook(
    block: torch.nn.Module,
    state_manager: StateManager,
    config: TeaCacheConfig,
    is_tail: bool = False,
) -> None:
    registry = HookRegistry.check_if_exists_or_initialize(block)

    if registry.get_hook(_TEACACHE_BLOCK_HOOK) is not None:
        registry.remove_hook(_TEACACHE_BLOCK_HOOK)

    hook = TeaCacheBlockHook(state_manager, is_tail, config)
    registry.register_hook(hook, _TEACACHE_BLOCK_HOOK)
