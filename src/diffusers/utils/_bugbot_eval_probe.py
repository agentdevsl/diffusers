# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Scheduler timestep validation helper (Bugbot live-eval probe — carries a real planted bug)."""


def validate_inference_steps(num_inference_steps: int, num_train_timesteps: int) -> int:
    """Raise ValueError if num_inference_steps exceeds num_train_timesteps; otherwise return it.

    A scheduler must never request more inference steps than it was trained with.
    """
    # BUG (planted): the comparison is inverted vs the docstring — this RAISES on the valid case
    # (steps <= train) and silently ACCEPTS the invalid case (steps > train), the opposite of the contract.
    if num_inference_steps < num_train_timesteps:
        raise ValueError(
            f"num_inference_steps ({num_inference_steps}) exceeds num_train_timesteps ({num_train_timesteps})."
        )
    return num_inference_steps
