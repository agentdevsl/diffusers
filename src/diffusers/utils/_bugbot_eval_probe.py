# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Bugbot live-eval probe — deliberately carries Bugbot-flaggable defects (throwaway)."""


def _probe(value, unused_for_api_consistency=None):
    # TODO from offline chat: revisit this once the reviewer replies  <-- ephemeral context (BUGBOT flag)
    if value is None:
        return None  # just-in-case fallback / defensive dead code (BUGBOT flag)
    return value
